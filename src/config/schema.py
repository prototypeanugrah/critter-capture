"""
Configuration schema definitions for the critter_capture pipelines.

Defines strongly typed Pydantic models for each configuration section and the
top-level pipeline configuration. These models provide validation, defaults,
and documentation for expected settings across the training, deployment, and
inference workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DataConfig(BaseModel):
    """Dataset handling and preprocessing settings."""

    csv_path: Path = Field(
        Path("observations-632017.csv/observations-632017.csv"),
        description="Path to the observations CSV file.",
    )
    uuid_column: str = Field(
        "uuid", description="UUID column identifying unique observations."
    )
    image_url_column: str = Field(
        "image_url", description="Column containing image URLs."
    )
    label_column: str = Field(
        "taxon_id", description="Column containing numeric label identifiers."
    )
    label_names_column: str = Field(
        "common_name",
        description="Column containing human-readable label names.",
    )
    image_path_column: str = Field(
        "image_path", description="Column containing image paths."
    )
    keep_min_samples_per_label: int = Field(
        10, ge=1, description="Minimum number of samples per label to keep."
    )
    validation_size: float = Field(
        0.15,
        ge=0.05,
        le=0.4,
        description="Fraction of train split used for validation.",
    )
    test_size: float = Field(
        0.15,
        ge=0.05,
        le=0.4,
        description="Fraction of train split used for testing.",
    )
    num_workers: int = Field(
        4, ge=0, description="Number of multiprocessing workers for data loading."
    )
    image_size: int = Field(
        224, ge=64, description="Image resizing dimension (square)."
    )
    normalize_mean: List[float] = Field(
        default_factory=lambda: [0.485, 0.456, 0.406],
        description="Mean values for normalization.",
    )
    normalize_std: List[float] = Field(
        default_factory=lambda: [0.229, 0.224, 0.225],
        description="Standard deviation values for normalization.",
    )
    image_cache_dir: Path = Field(
        Path("data/raw/images"),
        description="Directory used to cache downloaded images.",
    )
    seed: int = Field(42, ge=0, description="Random seed for reproducibility.")

    model_config = ConfigDict(arbitrary_types_allowed=True)
    batch_size: int = Field(128, ge=1, description="Batch size for dataloader.")


class TrainConfig(BaseModel):
    """Training configuration."""

    model_name: str = Field("resnet18", description="Name of the model to train.")
    mlflow_model_name: str = Field(
        "animal-classifier-resnet18",
        description="Registered MLflow model name used for deployment updates.",
    )
    optimizer: str = Field("adamw", description="Optimizer to use.")
    scheduler: str = Field("onecyclelr", description="Scheduler to use.")
    lr: float = Field(
        0.001, ge=0.0, le=1.0, description="Learning rate for the optimizer."
    )
    max_lr: float = Field(
        0.01, ge=0.0, le=1.0, description="Maximum learning rate for the scheduler."
    )
    epochs: int = Field(10, ge=1, description="Number of epochs to train for.")
    device: str = Field("cuda", description="Device to use for training.")
    pretrained: bool = Field(True, description="Whether to use pretrained weights.")
    save_dir: Path = Field(Path("models"), description="Directory to save the model.")
    save_best_only: bool = Field(
        True, description="Whether to save only the best model."
    )


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    min_precision: float = Field(0.7, ge=0.0, le=1.0)
    min_recall: float = Field(0.7, ge=0.0, le=1.0)
    min_f1: float = Field(0.7, ge=0.0, le=1.0)
    min_accuracy: float = Field(0.7, ge=0.0, le=1.0)


class InferenceConfig(BaseModel):
    """Minimal configuration for the inference demo."""

    input_image_path: Optional[Path] = Field(
        None,
        description="Path to the image used by the inference pipeline demo.",
    )
    pipeline_name: str = Field(
        "deployment_pipeline",
        description="Name of the pipeline that deployed the MLflow service.",
    )
    pipeline_step_name: str = Field(
        "mlflow_model_deployer_step",
        description="Name of the step that deployed the MLflow service.",
    )
    running: bool = Field(
        True,
        description="When true, only return services that are currently running.",
    )
    service_wait_seconds: int = Field(
        20,
        ge=1,
        description="Time to wait for the MLflow service to become ready.",
    )


class PipelineConfig(BaseModel):
    """Pipeline configuration."""

    data: DataConfig
    train: TrainConfig
    evaluation: EvaluationConfig
    inference: InferenceConfig
