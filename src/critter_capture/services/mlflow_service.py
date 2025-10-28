"""
Utilities for working with MLflow tracking and model registry.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import mlflow
from mlflow.entities import Run
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

LOGGER = logging.getLogger(__name__)


def configure_mlflow(
    tracking_uri: str,
    registry_uri: Optional[str] = None,
) -> None:
    """Configure MLflow tracking URIs."""

    mlflow.set_tracking_uri(tracking_uri)
    if registry_uri:
        mlflow.set_registry_uri(registry_uri)
    LOGGER.info("Configured MLflow with tracking URI %s", tracking_uri)


def start_run(
    experiment_name: str,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Run:
    """Start an MLflow run with the given experiment."""

    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=run_name, tags=tags)
    LOGGER.info("Started MLflow run %s", run.info.run_id)
    return run


def log_dict_artifact(
    data: Dict,
    artifact_path: str,
    filename: str,
) -> None:
    """Persist a dictionary artifact to the MLflow run."""

    temp_path = Path("outputs") / artifact_path
    temp_path.mkdir(parents=True, exist_ok=True)
    file_path = temp_path / filename
    file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    mlflow.log_artifact(str(file_path), artifact_path=artifact_path)


def log_config(config: Dict) -> None:
    log_dict_artifact(
        config,
        artifact_path="config",
        filename="config.json",
    )


def register_model(
    model_uri: str,
    model_name: str,
    run_id: str,
    stage: Optional[str] = None,
) -> str:
    try:
        # ``mlflow.register_model`` ensures the registered model exists before
        # creating a new version, which avoids flaky 5xx responses from the
        # registry when the model has not been created yet.
        result = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
        )
    except MlflowException as exc:
        tracking_uri = mlflow.get_tracking_uri()
        unsupported = getattr(exc, "error_code", None) == "UNSUPPORTED_OPERATION"
        msg = str(exc)
        if unsupported or "model registry functionality requires" in msg.lower():
            guidance = (
                "MLflow model registry operations require the tracking server to use "
                "a database-backed store (e.g., SQLite, MySQL, PostgreSQL). "
                f"The current tracking URI '{tracking_uri}' does not support model registry. "
                "Restart the MLflow server with a database backend (for example: "
                "`mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db "
                "--default-artifact-root mlruns/artifacts`) or disable deployment."
            )
            LOGGER.error(guidance)
            raise RuntimeError(guidance) from exc
        LOGGER.exception(
            "Failed to register model '%s' from run %s using URI %s",
            model_name,
            run_id,
            model_uri,
        )
        raise
    except Exception:
        LOGGER.exception(
            "Failed to register model '%s' from run %s using URI %s",
            model_name,
            run_id,
            model_uri,
        )
        raise

    if stage:
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage=stage,
            archive_existing=True,
        )
    LOGGER.info("Registered model %s version %s", model_name, result.version)
    return result.version


def update_model_stage(
    model_name: str,
    model_version: str | int,
    stage: str,
) -> None:
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=int(model_version),
        stage=stage,
        archive_existing=True,
    )
    LOGGER.info(
        "Moved model %s version %s to stage %s",
        model_name,
        model_version,
        stage,
    )


__all__ = [
    "configure_mlflow",
    "start_run",
    "log_config",
    "register_model",
    "update_model_stage",
]
