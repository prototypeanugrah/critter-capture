#!/usr/bin/env bash
set -euo pipefail

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export ZENML_AUTO_OPEN_DASHBOARD=true

echo "[train] Initializing ZenML..."
uv run zenml init

echo "[train] Authenticating with local ZenML server..."
if ! uv run zenml logout --local >/dev/null 2>&1; then
  echo "[train] No existing session to log out from."
fi
uv run zenml login --local

echo "[train] Ensuring MLflow experiment tracker exists..."
if ! uv run zenml experiment-tracker describe mlflow_tracker >/dev/null 2>&1; then
  uv run zenml experiment-tracker register mlflow_tracker \
    --flavor=mlflow \
    --tracking_uri='file:./mlruns'
fi

echo "[train] Ensuring MLflow model deployer exists..."
if ! uv run zenml model-deployer describe mlflow >/dev/null 2>&1; then
  uv run zenml model-deployer register mlflow --flavor=mlflow
fi

echo "[train] Configuring local_mlflow_stack..."
if ! uv run zenml stack describe local_mlflow_stack >/dev/null 2>&1; then
  uv run zenml stack register local_mlflow_stack \
    -o default \
    -a default \
    -e mlflow_tracker \
    -d mlflow \
    --set
else
  uv run zenml stack update local_mlflow_stack \
    -e mlflow_tracker \
    -d mlflow
  uv run zenml stack set local_mlflow_stack
fi

uv run zenml stack describe

echo "[train] Running training pipeline..."
uv run run_training.py
