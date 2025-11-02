import logging
from pathlib import Path
from typing import Tuple

import mlflow
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from zenml import step

from src.config import DataConfig
from src.evaluators import Evaluator
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


@step(
    enable_cache=False,
    experiment_tracker="mlflow_tracker",
    # output_materializers=EvaluateModelOutputMaterializer,
)
def evaluate_model(
    model: AnimalClassifierResNet18,
    test_loader: DataLoader,
) -> Tuple[float, float, float, float]:
    """
    Evaluate the model on the test data.

    This step:
    - Receives the same data bundle from prepare_data_step (ensures same test set)
    - Loads the trained model from the provided path
    - Creates a DataLoader for the test dataset
    - Generates predictions for all test samples
    - Calculates evaluation metrics (accuracy, precision, recall, f1)
    - Logs metrics to MLflow

    Args:
        model (torch.nn.Module): The trained model
        data_bundle (DatasetBundle): Bundle containing the test dataset (from prepare step)
        data_config (DataConfig): Configuration for data loading
        train_config (TrainConfig): Configuration for model initialization

    Returns:
        EvaluateModelOutput: Evaluation metrics containing accuracy, precision, recall, and f1 score
    """

    config_path = _find_config_file("config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        data_config = DataConfig(**config["data"])

    evaluator = Evaluator()

    LOGGER.info("Generating predictions on test data...")
    try:
        # Get predictions and true labels
        all_predictions = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(model.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        y_pred = np.array(all_predictions)
        y_true = np.array(all_labels)

        # Calculate the metrics
        metrics = evaluator.compute_classification_metrics(
            y_pred=y_pred,
            y_true=y_true,
        )

        mlflow.log_metrics(
            {
                "test_accuracy": metrics.accuracy,
                "test_precision": metrics.precision,
                "test_recall": metrics.recall,
                "test_f1": metrics.f1,
            }
        )

        LOGGER.info("Evaluation completed successfully.")
        return (
            metrics.accuracy,
            metrics.precision,
            metrics.recall,
            metrics.f1,
        )
    except Exception as e:
        LOGGER.error(
            "Exception %s occurred in evaluate_model step: %s",
            type(e).__name__,
            str(e),
        )
        raise e
