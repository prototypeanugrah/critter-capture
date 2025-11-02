import logging

from src.pipelines.inference_pipeline import inference_pipeline

LOGGER = logging.getLogger(__name__)


def run_inference() -> None:
    """
    Run the inference pipeline using the provided configuration.

    Args:
        config (Path): Configuration file path.
    """
    LOGGER.info("Starting inference pipeline...")
    pipeline_run = inference_pipeline()

    run_id = getattr(pipeline_run, "id", None)
    if run_id:
        LOGGER.info("Inference pipeline completed. Run ID: %s", run_id)
    else:
        LOGGER.info("Inference pipeline completed.")
    LOGGER.info("Inspect the 'predictor' step artifacts/logs for prediction details.")


if __name__ == "__main__":
    run_inference()
