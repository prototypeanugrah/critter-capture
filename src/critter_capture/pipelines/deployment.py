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

from critter_capture.pipelines.base import PipelineBase, PipelineContext
from critter_capture.pipelines.training import TrainingPipeline, TrainingResult
from critter_capture.services import (
    configure_mlflow,
    register_model,
    save_process_metadata,
    start_mlflow_server,
    stop_existing_process,
    update_model_stage,
    wait_for_healthcheck,
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


class DeploymentPipeline(PipelineBase):
    """Extends the training pipeline with deployment orchestration."""

    def __init__(
        self,
        context: PipelineContext,
        config_path: Path,
        environment: Optional[str],
    ) -> None:
        super().__init__(context)
        self._config_path = config_path
        self._environment = environment

    def run(self) -> DeploymentResult:
        cfg = self.context.config
        configure_mlflow(
            cfg.storage.mlflow_tracking_uri, cfg.storage.mlflow_registry_uri
        )

        training_pipeline = TrainingPipeline(
            context=self.context,
            config_path=self._config_path,
            environment=self._environment,
        )
        training_result = training_pipeline.run()

        decision = self._evaluate(training_result)
        service_url = None
        metadata_path = None

        if decision.approved and cfg.deployment.enable:
            service_url, metadata_path = self._deploy(training_result, decision)
        else:
            LOGGER.info("Deployment skipped: %s", decision.reason)

        return DeploymentResult(
            training=training_result,
            decision=decision,
            service_url=service_url,
            metadata_path=metadata_path,
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
    ) -> tuple[str, Path]:
        cfg = self.context.config
        model_uri = f"runs:/{training_result.run_id}/model_artifact"
        version = register_model(
            model_uri=model_uri,
            model_name=cfg.deployment.mlflow_model_name,
            run_id=training_result.run_id,
            stage=None,
        )

        deployment_stage = "Production"
        update_model_stage(
            cfg.deployment.mlflow_model_name,
            version,
            deployment_stage,
        )

        decision.model_uri = model_uri
        decision.model_version = str(version)

        mode = cfg.deployment.mode
        service_url = (
            cfg.deployment.external_service_url
            or f"http://{cfg.deployment.serving_host}:{cfg.deployment.serving_port}/invocations"
        )

        if mode == "external":
            LOGGER.info(
                "External deployment mode detected. Registered model version %s for serving endpoint %s.",
                version,
                service_url,
            )
            client = MlflowClient()
            client.set_tag(training_result.run_id, "deployment_mode", mode)
            client.set_tag(
                training_result.run_id, "deployed_model_version", str(version)
            )
            client.set_tag(training_result.run_id, "deployed_endpoint", service_url)
            return service_url, None

        metadata_path = Path("outputs/deployment/serving_process.json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        stop_existing_process(metadata_path)

        process = start_mlflow_server(
            model_uri=model_uri,
            host=cfg.deployment.serving_host,
            port=cfg.deployment.serving_port,
            env=cfg.deployment.server_env or None,
        )
        save_process_metadata(process, metadata_path)

        health_url = (
            f"http://{cfg.deployment.serving_host}:{cfg.deployment.serving_port}/ping"
        )
        if not wait_for_healthcheck(
            health_url, timeout=cfg.deployment.healthcheck_timeout
        ):
            LOGGER.error("Deployment failed health check.")
            raise RuntimeError("Deployment health check failed.")

        LOGGER.info("Model deployed at %s", service_url)

        client = MlflowClient()
        client.set_tag(training_result.run_id, "deployed_model_version", str(version))
        client.set_tag(training_result.run_id, "deployed_endpoint", service_url)
        client.set_tag(training_result.run_id, "deployment_mode", mode)

        return service_url, metadata_path


def run_deployment_pipeline(
    config_path: Path, environment: Optional[str] = None
) -> DeploymentResult:
    from critter_capture.pipelines.base import build_context

    context = build_context(config_path, environment)
    pipeline = DeploymentPipeline(
        context=context, config_path=config_path, environment=environment
    )
    return pipeline.run()


__all__ = [
    "DeploymentPipeline",
    "DeploymentDecision",
    "DeploymentResult",
    "run_deployment_pipeline",
]
