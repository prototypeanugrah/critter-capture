import logging
from pathlib import Path
from typing import Tuple

import mlflow
import yaml
from torch.utils.data import DataLoader
from zenml import step
from zenml.integrations.pytorch.materializers import PyTorchDataLoaderMaterializer

from src.config import DataConfig
from src.data.dataset import build_data_loaders, prepare_data_for_training

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


@step(
    enable_cache=False,
    experiment_tracker="mlflow_tracker",
    output_materializers=PyTorchDataLoaderMaterializer,
)
def load_data() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare data for training by loading, cleaning, and splitting into train/val/test sets.

    This step:
    - Loads the CSV data
    - Cleans and filters the data
    - Creates stratified train/validation/test splits
    - Returns a DatasetBundle with all three datasets

    Args:
        data_config (DataConfig): Configuration for data preparation

    Returns:
        DatasetBundle: Bundle containing train, validation, and test datasets
    """

    config_file = "config.yaml"
    LOGGER.info("Preparing data for training...")
    config_path = _find_config_file(config_file)
    LOGGER.info("Loading config from %s", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        data_config = DataConfig(**config["data"])

    mlflow.log_params(data_config.model_dump(mode="json"))

    dataset_bundle = prepare_data_for_training(data_config)
    dataloader = build_data_loaders(
        dataset_bundle,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
    )

    num_classes = len(dataset_bundle.label_names)
    # add num_classes to the config
    config["data"]["num_classes"] = num_classes
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    # print(config)

    LOGGER.info(
        "Data preparation complete. Train: %d, Val: %d, Test: %d samples",
        len(dataloader["train_dataloader"]),
        len(dataloader["validation_dataloader"]),
        len(dataloader["test_dataloader"]),
    )

    return (
        dataloader["train_dataloader"],
        dataloader["validation_dataloader"],
        dataloader["test_dataloader"],
    )
