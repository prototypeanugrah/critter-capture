import logging
from pathlib import Path

import mlflow
import torch
import yaml
from mlflow.models import infer_signature
from torch.utils.data import DataLoader
from zenml import step
from zenml.integrations.pytorch.materializers import PyTorchModuleMaterializer

from src.config import DataConfig, TrainConfig
from src.models.resnet18 import AnimalClassifierResNet18

LOGGER = logging.getLogger(__name__)


def _find_config_file(filename: str) -> Path:
    """Find the config.yaml file in common locations."""
    # Try current directory first (for temp files from deploy script)
    if Path(filename).exists():
        return Path(filename)

    # Try project root
    project_root = Path(__file__).parent.parent.parent
    if (project_root / filename).exists():
        return project_root / filename

    # Try src/steps/config.yaml
    if (project_root / "src" / "steps" / filename).exists():
        return project_root / "src" / "steps" / filename

    # Default fallback
    return Path(filename)


def calculate_mfb_class_weights(
    train_dataloader: DataLoader,
    num_classes: int,
) -> torch.Tensor:
    """
    Calculate Median Frequency Balancing class weights from training data.

    Args:
        train_dataloader: Training dataloader (NOT validation or test)
        num_classes: Number of classes

    Returns:
        torch.Tensor: Class weights for each class
    """
    # Count samples per class in TRAINING SET ONLY
    class_counts = torch.zeros(num_classes, dtype=torch.long)

    for _, labels in train_dataloader:
        for label in labels:
            class_counts[label] += 1

    # Calculate median frequency
    median_freq = torch.median(class_counts.float())

    # Calculate weights: weight_i = median_freq / freq_i
    class_weights = median_freq / class_counts.float()

    # Handle any zero counts (shouldn't happen in practice with good data)
    class_weights[class_counts == 0] = 0.0

    return class_weights


@step(
    output_materializers=PyTorchModuleMaterializer,
    experiment_tracker="mlflow_tracker",
    enable_cache=False,
)
def train_model(
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
) -> AnimalClassifierResNet18:
    """
    Train the model on the prepared data.

    This step:
    - Receives prepared data from the prepare_data_step
    - Creates DataLoaders from the DatasetBundle
    - Initializes the model
    - Trains the model with train and validation data
    - Saves the best model checkpoint
    - Logs the best model to MLflow for downstream deployment

    Args:
        data_bundle (DatasetBundle): Bundle containing train, validation, and test datasets
        data_config (DataConfig): Configuration for data loading
        train_config (TrainConfig): Configuration for training

    Returns:
        TrainedModelArtifact: References to the trained model artifact
    """
    LOGGER.info("Building data loaders...")

    LOGGER.info("Preparing data for training...")
    config_file = "config.yaml"
    config_path = _find_config_file(config_file)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        data_config = DataConfig(**config["data"])
        train_config = TrainConfig(**config["train"])

    mlflow.log_params(train_config.model_dump(mode="json"))

    LOGGER.info(
        "Initializing model with %d classes...",
        data_config.num_classes,
    )

    class_weights = calculate_mfb_class_weights(
        train_dataloader=train_dataloader,
        num_classes=data_config.num_classes,
    )
    model = AnimalClassifierResNet18(
        num_classes=data_config.num_classes,
        optimizer=train_config.optimizer,
        pretrained=train_config.pretrained,
        lr=train_config.lr,
        max_lr=train_config.max_lr,
        epochs=train_config.epochs,
        device=train_config.device,
        train_loader=train_dataloader,
        class_weights=class_weights,
    )

    LOGGER.info("Starting model training...")
    model.fit(
        train_loader=train_dataloader,
        val_loader=validation_dataloader,
        save_dir=train_config.save_dir,
        save_best_only=train_config.save_best_only,
    )

    model_path = Path(train_config.save_dir) / "best_model.pth"
    config["inference"]["model_path"] = model_path
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    LOGGER.info(
        "Loading best checkpoint from %s for MLflow logging",
        model_path,
    )
    model.load(model_path)

    example_input = torch.randn(
        1,
        3,
        data_config.image_size,
        data_config.image_size,
        device="cpu",
    )
    model.eval()
    with torch.no_grad():
        example_output = model(example_input.to(model.device)).cpu().numpy()

    example_input_numpy = example_input.cpu().numpy()
    signature = infer_signature(
        example_input_numpy,
        example_output,
    )

    artifact_path = train_config.mlflow_model_name
    LOGGER.info(
        "Logging trained model to MLflow at artifact path '%s'",
        artifact_path,
    )
    model_info = mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path=artifact_path,
        input_example=example_input_numpy,
        signature=signature,
        registered_model_name=train_config.mlflow_model_name,
    )

    version = model_info.registered_model_version
    config["inference"]["model_version"] = version
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    return model
