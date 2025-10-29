"""
Inference pipeline that exercises the deployed model endpoint.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import numpy as np
import torch
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader

from critter_capture.data import build_dataloaders, prepare_datasets
from critter_capture.metrics.classification import (
    ClassificationMetrics,
    compute_classification_metrics,
)
from critter_capture.pipelines.base import PipelineBase, PipelineContext
from critter_capture.services import configure_mlflow

LOGGER = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    metrics: ClassificationMetrics
    predictions_path: Path
    service_url: str
    run_id: str


class InferencePipeline(PipelineBase):
    """Pipeline to validate inference against the deployment endpoint."""

    def __init__(
        self,
        context: PipelineContext,
        config_path: Path,
        environment: Optional[str],
    ) -> None:
        super().__init__(context)
        self._config_path = config_path
        self._environment = environment

    def run(self) -> InferenceResult:
        cfg = self.context.config
        configure_mlflow(
            cfg.storage.mlflow_tracking_uri, cfg.storage.mlflow_registry_uri
        )

        bundle = prepare_datasets(cfg.data, seed=cfg.training.seed)
        if cfg.model.num_classes is None or cfg.model.num_classes != len(
            bundle.label_names
        ):
            cfg.model.num_classes = len(bundle.label_names)
        dataloaders = build_dataloaders(
            bundle,
            batch_size=cfg.inference.batch_size,
            num_workers=cfg.data.num_workers,
        )
        test_loader = dataloaders["test"]

        service_url = f"http://{cfg.deployment.serving_host}:{cfg.deployment.serving_port}/invocations"

        LOGGER.info("Running inference pipeline against %s", service_url)
        predictions = asyncio.run(self._run_inference(test_loader, service_url))

        y_true = predictions["y_true"]
        y_scores = predictions["y_scores"]
        y_pred = np.argmax(y_scores, axis=1)

        metrics = compute_classification_metrics(y_true, y_pred, bundle.label_names)

        experiment_name = cfg.inference.experiment_name
        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(
            run_name="inference",
            tags={"pipeline": "inference", "environment": cfg.environment},
        )
        with run:
            mlflow.log_metric("inference_accuracy", metrics.accuracy)
            mlflow.log_metric("inference_macro_precision", metrics.macro_precision)
            mlflow.log_metric("inference_macro_recall", metrics.macro_recall)
            mlflow.log_metric("inference_macro_f1", metrics.macro_f1)
            mlflow.log_metric("inference_latency_ms_p95", predictions["latency_p95"])
            mlflow.log_dict(asdict(metrics), "metrics/inference_metrics.json")

            outputs_dir = Path("outputs/inference")
            outputs_dir.mkdir(parents=True, exist_ok=True)
            preds_path = outputs_dir / "predictions.npz"
            np.savez_compressed(
                preds_path,
                y_true=y_true,
                y_scores=y_scores,
                y_pred=y_pred,
                latencies=predictions["latencies"],
            )
            mlflow.log_artifact(str(preds_path), artifact_path="predictions")

            client = MlflowClient()
            client.set_tag(run.info.run_id, "service_url", service_url)

        return InferenceResult(
            metrics=metrics,
            predictions_path=preds_path,
            service_url=service_url,
            run_id=run.info.run_id,
        )

    async def _run_inference(
        self,
        dataloader: DataLoader,
        url: str,
    ) -> Dict[str, np.ndarray]:
        import time

        import httpx

        latencies: List[float] = []
        outputs: List[np.ndarray] = []
        truths: List[np.ndarray] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for batch in dataloader:
                images: torch.Tensor = batch["image"]
                targets: torch.Tensor = batch["target"]
                payload = {"instances": images.numpy().tolist()}

                start = time.time()
                response = await client.post(url, json=payload)
                latency = (time.time() - start) * 1000.0
                latencies.append(latency)

                response.raise_for_status()
                data = response.json()
                scores = np.array(data["predictions"], dtype=np.float32)
                outputs.append(scores)
                truths.append(targets.numpy())

        y_scores = np.concatenate(outputs, axis=0)
        y_true = np.concatenate(truths, axis=0)
        latency_p95 = float(np.percentile(latencies, 95)) if latencies else 0.0

        return {
            "y_scores": y_scores,
            "y_true": y_true,
            "latencies": np.array(latencies),
            "latency_p95": latency_p95,
        }


def run_inference_pipeline(
    config_path: Path, environment: Optional[str] = None
) -> InferenceResult:
    from critter_capture.pipelines.base import build_context

    context = build_context(config_path, environment)
    pipeline = InferencePipeline(
        context=context, config_path=config_path, environment=environment
    )
    return pipeline.run()


__all__ = ["InferencePipeline", "InferenceResult", "run_inference_pipeline"]
