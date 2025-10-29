"""ZenML pipeline definitions that wrap the native Critter Capture workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from zenml.pipelines import pipeline
from zenml.steps import step

from critter_capture.pipelines.deployment import (
    DeploymentResult,
    run_deployment_pipeline,
)
from critter_capture.pipelines.inference import (
    InferenceResult,
    run_inference_pipeline,
)
from critter_capture.pipelines.training import (
    TrainingResult,
    run_training_pipeline,
)


@step(enable_cache=False, experiment_tracker="mlflow_tracker")
def training_step(
    config_path: str,
    environment: Optional[str],
) -> TrainingResult:
    """Execute the training workflow within a ZenML step."""

    return run_training_pipeline(Path(config_path), environment)


@step(enable_cache=False, experiment_tracker="mlflow_tracker")
def deployment_step(
    config_path: str,
    environment: Optional[str],
    run_id: Optional[str],
) -> DeploymentResult:
    """Execute the deployment workflow (optionally reusing a prior run)."""

    return run_deployment_pipeline(Path(config_path), environment, run_id)


@step(enable_cache=False, experiment_tracker="mlflow_tracker")
def inference_step(
    config_path: str,
    environment: Optional[str],
) -> InferenceResult:
    """Execute the inference validation workflow."""

    return run_inference_pipeline(Path(config_path), environment)


@pipeline
def zenml_training_pipeline(
    config_path: str,
    environment: Optional[str] = None,
) -> None:
    """ZenML pipeline orchestrating the training step."""

    training_step(config_path=config_path, environment=environment)


@pipeline
def zenml_deployment_pipeline(
    config_path: str,
    environment: Optional[str] = None,
    run_id: Optional[str] = None,
) -> None:
    """ZenML pipeline orchestrating the deployment step."""

    deployment_step(config_path=config_path, environment=environment, run_id=run_id)


@pipeline
def zenml_inference_pipeline(
    config_path: str,
    environment: Optional[str] = None,
) -> None:
    """ZenML pipeline orchestrating the inference step."""

    inference_step(config_path=config_path, environment=environment)


def run_training_pipeline_with_zenml(
    config_path: Path,
    environment: Optional[str] = None,
):
    """Helper to run the training pipeline via ZenML."""

    pipeline_instance = zenml_training_pipeline(
        config_path=str(config_path),
        environment=environment,
    )
    return pipeline_instance.run()


def run_deployment_pipeline_with_zenml(
    config_path: Path,
    environment: Optional[str] = None,
    run_id: Optional[str] = None,
):
    """Helper to run the deployment pipeline via ZenML."""

    pipeline_instance = zenml_deployment_pipeline(
        config_path=str(config_path),
        environment=environment,
        run_id=run_id,
    )
    return pipeline_instance.run()


def run_inference_pipeline_with_zenml(
    config_path: Path,
    environment: Optional[str] = None,
):
    """Helper to run the inference pipeline via ZenML."""

    pipeline_instance = zenml_inference_pipeline(
        config_path=str(config_path),
        environment=environment,
    )
    return pipeline_instance.run()


__all__ = [
    "deployment_step",
    "inference_step",
    "run_deployment_pipeline_with_zenml",
    "run_inference_pipeline_with_zenml",
    "run_training_pipeline_with_zenml",
    "training_step",
    "zenml_deployment_pipeline",
    "zenml_inference_pipeline",
    "zenml_training_pipeline",
]
