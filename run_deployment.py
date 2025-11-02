import logging

from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

from src.pipelines.deployment_pipeline import deployment_pipeline

LOGGER = logging.getLogger(__name__)


def run_deployment() -> None:
    """
    Run the deployment pipeline with the given configuration.

    Args:
        config (Path): Path to the configuration YAML file
    """

    LOGGER.info("Starting deployment pipeline...")
    deployment_pipeline()

    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    service = model_deployer.find_model_server(
        pipeline_name="deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        running=False,
    )

    if service:
        first_service = service[0]
        if first_service:
            print(
                "The MLflow prediction server is running locally as a daemon process "
                "and accepts inference requests at:\n"
                f"    {first_service.prediction_url}\n"
            )

    # run_id = getattr(pipeline_run, "id", None)
    # if run_id:
    #     LOGGER.info("Deployment pipeline completed successfully! Run ID: %s", run_id)
    # else:
    #     LOGGER.info("Deployment pipeline completed successfully!")


if __name__ == "__main__":
    run_deployment()
