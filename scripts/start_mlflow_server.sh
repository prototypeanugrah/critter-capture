#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_BACKEND_STORE_URI="file://${WORKDIR}/mlruns"
DEFAULT_REGISTRY_STORE_URI="sqlite:///${WORKDIR}/mlruns/mlflow.db"
DEFAULT_ARTIFACT_ROOT="${WORKDIR}/mlruns/artifacts"

MLFLOW_BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI:-$DEFAULT_BACKEND_STORE_URI}"
MLFLOW_REGISTRY_STORE_URI="${MLFLOW_REGISTRY_STORE_URI:-$DEFAULT_REGISTRY_STORE_URI}"
MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-$DEFAULT_ARTIFACT_ROOT}"

mkdir -p "${WORKDIR}/mlruns"
mkdir -p "${MLFLOW_ARTIFACT_ROOT}"
cd "${WORKDIR}"

echo "Starting MLflow tracking server..."
MLFLOW_BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI}" \
MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT}" \
MLFLOW_REGISTRY_STORE_URI="${MLFLOW_REGISTRY_STORE_URI}" \
    uv run mlflow server \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
    --registry-store-uri "${MLFLOW_REGISTRY_STORE_URI}" \
    --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}" \
    --host 127.0.0.1 \
    --port 5000
