import logging
from pathlib import Path
from typing import Tuple

import yaml
from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from src.config import TrainConfig
from src.models.resnet18 import AnimalClassifierResNet18
from src.steps.create_data import load_data
from src.steps.deploy_model import deployment_trigger
from src.steps.evaluate_model import evaluate_model
from src.steps.train_model import train_model

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


@pipeline()
def deployment_pipeline() -> Tuple[AnimalClassifierResNet18, str, bool]:
    """
    Complete deployment pipeline for the animal classifier.

    This pipeline orchestrates the following steps:
    1. Data preparation: Load and split data once
    2. Model training: Train model using prepared data
    3. Model evaluation: Evaluate model on test data from same split
    4. Deploy model if evaluation metrics are good enough
    """

    config_path = _find_config_file("test_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        train_config = TrainConfig(**config["train"])

    # Load data
    train_dataloader, validation_dataloader, test_dataloader = load_data()

    # Train model
    model = train_model(
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
    )

    # Evaluate model
    accuracy, precision, recall, f1 = evaluate_model(
        model=model,
        test_loader=test_dataloader,
    )

    # Step 4: Deploy model if evaluation metrics are good enough
    deployment_decision = deployment_trigger(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )

    model_deployment_service = mlflow_model_deployer_step(
        model=model,
        model_name=train_config.mlflow_model_name,
        deploy_decision=deployment_decision,
        workers=2,
    )

    LOGGER.info("Model deployed successfully: %s", model_deployment_service)
