#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(dirname "$0")/.."
MLFLOW_BACKEND_STORE_URI="sqlite:///${WORKDIR}/mlruns/mlflow.db"
MLFLOW_ARTIFACT_ROOT="${WORKDIR}/mlruns/artifacts"

mkdir -p "${WORKDIR}/mlruns"
cd "${WORKDIR}"

echo "Starting MLflow tracking server..."
MLFLOW_BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI}" \
MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT}" \
    uv run mlflow server \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
    --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}" \
    --host 127.0.0.1 \
    --port 5000
