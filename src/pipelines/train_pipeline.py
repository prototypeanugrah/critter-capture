from zenml import pipeline

from src.steps.create_data import load_data
from src.steps.evaluate_model import evaluate_model
from src.steps.train_model import train_model


@pipeline(enable_cache=False, experiment_tracker="mlflow_tracker")
def train_pipeline():
    """
    Complete training pipeline for the animal classifier.

    This pipeline orchestrates the following steps:
    1. Data preparation: Load and split data once
    2. Model training: Train model using prepared data
    3. Model evaluation: Evaluate model on test data from same split

    Args:
        data_config (DataConfig): Configuration for data preparation and loading
        train_config (TrainConfig): Configuration for model training

    Note:
        Data is prepared once and passed to both training and evaluation steps,
        ensuring consistency. Evaluation metrics are logged to MLflow and can be
        viewed in the MLflow UI.
    """

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
