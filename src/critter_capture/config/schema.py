"""
Configuration schema definitions for the critter_capture pipelines.

Defines strongly typed Pydantic models for each configuration section and the
top-level pipeline configuration. These models provide validation, defaults,
and documentation for expected settings across the training, deployment, and
inference workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    augmentations: bool = Field(
        True, description="Enable light data augmentation for training."
    )
    image_cache_dir: Path = Field(
        Path("data/raw/images"),
        description="Directory used to cache downloaded images.",
    )
    sample_limit: Optional[int] = Field(
        None,
        ge=1,
        description="Optional limit on the number of records to load (useful for quick tests).",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ModelConfig(BaseModel):
    """Model architecture parameters."""

    num_classes: Optional[int] = Field(
        None, ge=1, description="Number of output labels."
    )
    in_channels: int = Field(3, ge=1, description="Input image channels.")
    conv_filters: List[int] = Field(
        default_factory=lambda: [32, 64, 128, 256, 256],
        description="Number of filters for each convolution layer.",
    )
    kernel_size: int = Field(3, description="Convolution kernel size.")
    dropout: float = Field(0.3, ge=0.0, le=0.8)
    hidden_dim: int = Field(
        512, ge=32, description="Number of units in the hidden layer."
    )
    use_batch_norm: bool = Field(True, description="Use batch normalization.")
    use_spectral_norm: bool = Field(False, description="Use spectral normalization.")

    @field_validator("conv_filters")
    def validate_conv_filters(cls, value: List[int]) -> List[int]:
        if len(value) != 5:
            raise ValueError(
                "conv_filters must contain exactly five entries for the five conv layers."
            )
        return value


class OptimizerConfig(BaseModel):
    """Optimizer configuration."""

    name: str = Field("adamw", description="Optimizer name.")
    lr: float = Field(1e-3, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    betas: Optional[List[float]] = None


class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""

    name: Optional[str] = Field(
        "cosine",
        description="Scheduler name or None.",
    )
    t_max: int = Field(
        10,
        ge=1,
        description="Cosine annealing period if applicable.",
    )
    min_lr: float = Field(1e-6, ge=0, description="Minimum learning rate.")


class TrainingConfig(BaseModel):
    """Training loop settings."""

    epochs: int = Field(30, ge=1)
    batch_size: int = Field(64, ge=4)
    gradient_clip_norm: Optional[float] = Field(5.0, ge=0)
    amp: bool = Field(
        True,
        description="Use automatic mixed precision if available.",
    )
    early_stopping_patience: int = Field(5, ge=1)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    seed: int = Field(42)
    device: str = Field(
        "cuda" if False else "cpu",
        description="Preferred device (cpu or cuda).",
    )


class TuneConfig(BaseModel):
    """Hyperparameter optimization configuration."""

    enabled: bool = True
    num_samples: int = Field(10, ge=1)
    max_epochs: int = Field(15, ge=1)
    grace_period: int = Field(5, ge=1)
    metric: str = Field(
        "val_macro_f1",
        description="Primary metric key reported to Ray Tune.",
    )
    mode: str = Field(
        "max",
        description="Optimization direction for the primary metric ('max' or 'min').",
    )
    search_space: Dict[str, Any] = Field(
        default_factory=lambda: {
            "lr": {"loguniform": [1e-4, 1e-2]},
            "weight_decay": {"loguniform": [1e-6, 1e-3]},
            "dropout": {"uniform": [0.2, 0.5]},
            "batch_size": {"choice": [32, 48, 64]},
        }
    )
    resources_per_trial: Dict[str, Any] = Field(
        default_factory=lambda: {"cpu": 2, "gpu": 0}
    )

    @field_validator("mode")
    def validate_mode(cls, value: str) -> str:
        lowered = value.lower()
        if lowered not in {"max", "min"}:
            raise ValueError("tuning.mode must be either 'max' or 'min'.")
        return lowered


class EvaluationCriteria(BaseModel):
    """Evaluation thresholds used for deployment gating."""

    min_precision: float = Field(0.7, ge=0, le=1)
    min_recall: float = Field(0.7, ge=0, le=1)
    metric_key: str = Field("val_macro_f1")


class DeploymentConfig(BaseModel):
    """Deployment specific configuration."""

    enable: bool = True
    mode: str = Field("process", description="Deployment mode: process or external.")
    evaluation: EvaluationCriteria = Field(default_factory=EvaluationCriteria)
    mlflow_model_name: str = Field("AnimalSpeciesClassifier")
    serving_host: str = Field("0.0.0.0")
    serving_port: int = Field(5001, ge=1024, le=65535)
    server_env: Dict[str, str] = Field(default_factory=dict)
    healthcheck_timeout: int = Field(60, ge=1)
    external_service_url: Optional[str] = Field(
        None, description="Optional externally managed serving endpoint."
    )

    @field_validator("mode")
    def validate_mode(cls, value: str) -> str:
        allowed = {"process", "external"}
        mode = value.lower()
        if mode not in allowed:
            raise ValueError(f"deployment.mode must be one of {allowed}")
        return mode


class InferenceConfig(BaseModel):
    """Inference pipeline configuration."""

    batch_size: int = Field(16, ge=1)
    max_concurrency: int = Field(4, ge=1)
    latency_threshold_ms: float = Field(500.0, ge=1)
    tolerance_precision_delta: float = Field(0.05, ge=0)
    tolerance_recall_delta: float = Field(0.05, ge=0)


class LoggingConfig(BaseModel):
    """Logging settings."""

    level: str = Field("INFO")
    log_dir: Path = Field(Path("logs"))
    log_file: str = Field("pipeline.log")


class StorageConfig(BaseModel):
    """Persistent storage configuration for artifacts and feedback."""

    mlflow_tracking_uri: str = Field("http://127.0.0.1:5000")
    mlflow_registry_uri: Optional[str] = None
    s3_bucket: str = Field("animal-species-feedback")
    s3_region: Optional[str] = None


class PipelineConfig(BaseModel):
    """Top-level configuration object."""

    environment: str = Field(
        "local", description="Configuration environment identifier."
    )
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    tuning: TuneConfig
    deployment: DeploymentConfig
    inference: InferenceConfig
    logging: LoggingConfig
    storage: StorageConfig
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("data")
    def ensure_paths(cls, value: DataConfig) -> DataConfig:
        value.csv_path = Path(value.csv_path)
        value.image_cache_dir = Path(value.image_cache_dir)
        return value

    @field_validator("logging")
    def ensure_log_dir(cls, value: LoggingConfig) -> LoggingConfig:
        value.log_dir = Path(value.log_dir)
        return value

    @field_validator("training")
    def resolve_device(cls, value: TrainingConfig) -> TrainingConfig:
        device = value.device.lower()
        if device not in {"cpu", "cuda"}:
            raise ValueError("training.device must be 'cpu' or 'cuda'.")
        return value
