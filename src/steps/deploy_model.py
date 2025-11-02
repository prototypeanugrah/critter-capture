import logging
from pathlib import Path

import yaml
from zenml.materializers import BuiltInMaterializer
from zenml.steps import step

from src.config import EvaluationConfig

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
    output_materializers=BuiltInMaterializer,
    enable_cache=False,
    experiment_tracker="mlflow_tracker",
)
def deployment_trigger(
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
) -> bool:
    """Determine if the evaluated model is good enough to deploy.

    Args:
        metrics (EvaluateModelOutput): Aggregated evaluation metrics.

    Returns:
        bool: True to deploy the model, False otherwise.
    """

    config_path = _find_config_file("config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        evaluation_config = EvaluationConfig(**config["evaluation"])

    precision_threshold_met = precision >= evaluation_config.min_precision
    recall_threshold_met = recall >= evaluation_config.min_recall
    f1_threshold_met = f1 >= evaluation_config.min_f1
    accuracy_threshold_met = accuracy >= evaluation_config.min_accuracy

    LOGGER.info(
        "Deployment gate results - precision: %.4f (>= %.4f: %s), "
        "recall: %.4f (>= %.4f: %s), f1: %.4f (>= %.4f: %s), "
        "accuracy: %.4f (>= %.4f: %s)",
        precision,
        evaluation_config.min_precision,
        precision_threshold_met,
        recall,
        evaluation_config.min_recall,
        recall_threshold_met,
        f1,
        evaluation_config.min_f1,
        f1_threshold_met,
        accuracy,
        evaluation_config.min_accuracy,
        accuracy_threshold_met,
    )

    conditions_met = (
        precision_threshold_met
        and recall_threshold_met
        and f1_threshold_met
        and accuracy_threshold_met
    )

    LOGGER.info("Deployment decision: %s", "approved" if conditions_met else "rejected")
    return conditions_met
