import logging

from src.pipelines.train_pipeline import train_pipeline

LOGGER = logging.getLogger(__name__)


def run_training():
    """
    Run the training pipeline with the given configuration.

    Args:
        config (Path): Path to the configuration YAML file
    """
    LOGGER.info("Starting training pipeline...")
    pipeline_run = train_pipeline()

    run_id = getattr(pipeline_run, "id", None)
    if run_id:
        LOGGER.info("Training pipeline completed successfully! Run ID: %s", run_id)
    else:
        LOGGER.info("Training pipeline completed successfully!")
    LOGGER.info(
        "Evaluation metrics have been logged to MLflow. View them in the MLflow UI."
    )


if __name__ == "__main__":
    run_training()
