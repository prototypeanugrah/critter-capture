"""
Deployment pipeline implementation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mlflow.tracking import MlflowClient

from critter_capture.config import load_config
from critter_capture.metrics.classification import ClassificationMetrics
from critter_capture.pipelines.base import PipelineBase, PipelineContext
from critter_capture.pipelines.training import TrainingPipeline, TrainingResult
from critter_capture.services import (
    configure_mlflow,
    register_model,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class DeploymentDecision:
    approved: bool
    reason: str
    model_uri: Optional[str] = None
    model_version: Optional[str] = None


@dataclass
class DeploymentResult:
    training: TrainingResult
    decision: DeploymentDecision
    service_url: Optional[str]
    metadata_path: Optional[Path]
    model_version: Optional[str]


def _resolve_checkpoint_path(download_path: Path) -> Path:
    """Return a stable path to the checkpoint file inside the MLflow artifact."""

    if download_path.is_file():
        return download_path

    candidates = sorted(
        (
            path
            for suffix in (".pt", ".pth", ".bin")
            for path in download_path.rglob(f"*{suffix}")
        ),
        key=lambda p: p.name,
    )
    if candidates:
        return candidates[0]

    fallback_files = sorted(path for path in download_path.rglob("*") if path.is_file())
    if fallback_files:
        return fallback_files[0]

    raise FileNotFoundError(
        f"Failed to locate checkpoint file in downloaded artifact at {download_path}"
    )


def load_training_result_from_run(
    run_id: str,
    config_path: Path,
    environment: str | None,
) -> TrainingResult:
    cfg = load_config(config_path, environment)

    client = MlflowClient()
    cache_dir = Path("outputs/mlflow_cache") / run_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(
        client.download_artifacts(run_id, "metadata/metadata.json", str(cache_dir))
    )
    raw_model_path = Path(client.download_artifacts(run_id, "model", str(cache_dir)))
    checkpoint_path = _resolve_checkpoint_path(raw_model_path)

    metadata = json.loads(metadata_path.read_text())
    best_metrics = ClassificationMetrics(**metadata["validation_metrics"])
    test_metrics = ClassificationMetrics(**metadata["test_metrics"])

    return TrainingResult(
        run_id=run_id,
        model_path=checkpoint_path,
        best_metrics=best_metrics,
        test_metrics=test_metrics,
        label_names=metadata["label_names"],
        best_params=metadata["best_params"],
        config=cfg,
        model_variant=metadata["model_variant"],
    )


class DeploymentPipeline(PipelineBase):
    """Extends the training pipeline with deployment orchestration."""

    def __init__(
        self,
        context: PipelineContext,
        config_path: Path,
        environment: Optional[str],
        run_id: Optional[str] = None,
    ) -> None:
        super().__init__(context)
        self._config_path = config_path
        self._environment = environment
        self._run_id = run_id

    def run(self) -> DeploymentResult:
        cfg = self.context.config
        configure_mlflow(
            cfg.storage.mlflow_tracking_uri, cfg.storage.mlflow_registry_uri
        )

        if self._run_id is None:
            training_pipeline = TrainingPipeline(
                context=self.context,
                config_path=self._config_path,
                environment=self._environment,
            )
            training_result = training_pipeline.run()
        else:
            training_result = load_training_result_from_run(
                run_id=self._run_id,
                config_path=self._config_path,
                environment=self._environment,
            )

        decision = self._evaluate(training_result)
        service_url = None
        metadata_path = None

        if decision.approved and cfg.deployment.enable:
            model_version = self._deploy(training_result, decision)
        else:
            LOGGER.info("Deployment skipped: %s", decision.reason)

        return DeploymentResult(
            training=training_result,
            decision=decision,
            service_url=service_url,
            metadata_path=metadata_path,
            model_version=model_version,
        )

    def _evaluate(self, training_result: TrainingResult) -> DeploymentDecision:
        cfg = self.context.config
        criteria = cfg.deployment.evaluation
        metrics = training_result.best_metrics

        precision_ok = metrics.macro_precision >= criteria.min_precision
        recall_ok = metrics.macro_recall >= criteria.min_recall
        accuracy_ok = metrics.accuracy >= criteria.min_accuracy
        approved = precision_ok and recall_ok and accuracy_ok

        reason = (
            "All deployment criteria satisfied."
            if approved
            else "Deployment criteria not met."
        )
        LOGGER.info(
            "Deployment decision: approved=%s precision=%.3f recall=%.3f accuracy=%.3f thresholds=(%.3f, %.3f, %.3f)",
            approved,
            metrics.macro_precision,
            metrics.macro_recall,
            metrics.accuracy,
            criteria.min_precision,
            criteria.min_recall,
            criteria.min_accuracy,
        )

        decision = DeploymentDecision(approved=approved, reason=reason)

        client = MlflowClient()
        client.set_tag(
            training_result.run_id,
            "model_variant",
            training_result.model_variant,
        )
        client.set_tag(
            training_result.run_id,
            "deployment_decision",
            "approved" if approved else "rejected",
        )
        client.set_tag(training_result.run_id, "deployment_reason", reason)

        outputs_dir = Path("outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        decision_path = outputs_dir / "deployment_decision.json"
        decision_payload = {
            "approved": approved,
            "reason": reason,
            "model_variant": training_result.model_variant,
            "metrics": {
                "macro_precision": metrics.macro_precision,
                "macro_recall": metrics.macro_recall,
                "macro_f1": metrics.macro_f1,
                "accuracy": metrics.accuracy,
            },
            "thresholds": {
                "min_precision": criteria.min_precision,
                "min_recall": criteria.min_recall,
                "min_accuracy": criteria.min_accuracy,
            },
        }
        decision_path.write_text(
            json.dumps(decision_payload, indent=2), encoding="utf-8"
        )
        client.log_artifact(
            run_id=training_result.run_id,
            local_path=str(decision_path),
            artifact_path="deployment",
        )

        return decision

    def _deploy(
        self, training_result: TrainingResult, decision: DeploymentDecision
    ) -> str:
        cfg = self.context.config
        model_variant = training_result.model_variant
        run_model_uri = f"runs:/{training_result.run_id}/model_artifact"

        if decision.approved:
            model_version = register_model(
                model_uri=run_model_uri,
                model_name=f"{cfg.deployment.mlflow_model_name}_{model_variant}",
            )
        return str(model_version)

        # registry_model_uri = f"models:/{f"{cfg.deployment.mlflow_model_name}_{model_variant}"}/{version}"
        # deployment_stage = "Production"
        # update_model_stage(
        #     cfg.deployment.mlflow_model_name,
        #     version,
        #     deployment_stage,
        # )

        # decision.model_uri = registry_model_uri
        # decision.model_version = str(version)

        # mode = cfg.deployment.mode
        # service_url = (
        #     cfg.deployment.external_service_url
        #     or f"http://{cfg.deployment.serving_host}:{cfg.deployment.serving_port}/invocations"
        # )
        # deployed_model_uri = (
        #     f"models:/{cfg.deployment.mlflow_model_name}/{deployment_stage}"
        #     if deployment_stage
        #     else registry_model_uri
        # )

        # if mode == "external":
        #     LOGGER.info(
        #         "External deployment mode detected. Registered model version %s for serving endpoint %s.",
        #         version,
        #         service_url,
        #     )
        #     client = MlflowClient()
        #     client.set_tag(training_result.run_id, "deployment_mode", mode)
        #     client.set_tag(
        #         training_result.run_id, "deployed_model_version", str(version)
        #     )
        #     client.set_tag(
        #         training_result.run_id, "deployed_model_uri", deployed_model_uri
        #     )
        #     client.set_tag(training_result.run_id, "deployed_endpoint", service_url)
        #     return service_url, None

        # metadata_path = Path("outputs/deployment/serving_process.json")
        # metadata_path.parent.mkdir(parents=True, exist_ok=True)
        # stop_existing_process(metadata_path)

        # server_env = dict(cfg.deployment.server_env or {})
        # server_env.setdefault("MLFLOW_TRACKING_URI", cfg.storage.mlflow_tracking_uri)
        # if cfg.storage.mlflow_registry_uri:
        #     server_env.setdefault(
        #         "MLFLOW_REGISTRY_URI", cfg.storage.mlflow_registry_uri
        #     )
        # else:
        #     server_env.setdefault(
        #         "MLFLOW_REGISTRY_URI", cfg.storage.mlflow_tracking_uri
        #     )

        # LOGGER.info(
        #     "Starting local serving for model URI %s (alias %s)",
        #     registry_model_uri,
        #     deployed_model_uri,
        # )

        # process = start_mlflow_server(
        #     model_uri=deployed_model_uri,
        #     host=cfg.deployment.serving_host,
        #     port=cfg.deployment.serving_port,
        #     env=server_env,
        # )
        # save_process_metadata(process, metadata_path)

        # health_url = (
        #     f"http://{cfg.deployment.serving_host}:{cfg.deployment.serving_port}/ping"
        # )
        # if not wait_for_healthcheck(
        #     health_url, timeout=cfg.deployment.healthcheck_timeout
        # ):
        #     LOGGER.error("Deployment failed health check.")
        #     raise RuntimeError("Deployment health check failed.")

        # LOGGER.info("Model deployed at %s", service_url)

        # client = MlflowClient()
        # client.set_tag(training_result.run_id, "deployed_model_version", str(version))
        # client.set_tag(training_result.run_id, "deployed_model_uri", deployed_model_uri)
        # client.set_tag(training_result.run_id, "deployed_endpoint", service_url)
        # client.set_tag(training_result.run_id, "deployment_mode", mode)

        # return service_url, metadata_path


def run_deployment_pipeline(
    config_path: Path,
    environment: Optional[str] = None,
    run_id: Optional[str] = None,
) -> DeploymentResult:
    from critter_capture.pipelines.base import build_context

    context = build_context(config_path, environment)
    pipeline = DeploymentPipeline(
        context=context,
        config_path=config_path,
        environment=environment,
        run_id=run_id,
    )
    return pipeline.run()


__all__ = [
    "DeploymentPipeline",
    "DeploymentDecision",
    "DeploymentResult",
    "run_deployment_pipeline",
]
