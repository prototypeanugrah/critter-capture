#!/usr/bin/env python
"""Run the ResNet-50 baseline evaluation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow

from critter_capture.baselines import run_resnet50_baseline
from critter_capture.data import build_dataloaders, compute_class_weights, prepare_datasets
from critter_capture.pipelines.base import build_context
from critter_capture.services import configure_mlflow
from critter_capture.utils import resolve_device

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a torchvision ResNet-50 baseline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipeline.yaml"),
        help="Path to the pipeline configuration file.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Configuration environment override (e.g. local, prod).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Existing MLflow run ID to append baseline metrics to.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of baseline fine-tuning epochs (default: min(training epochs, 5)).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate for the classifier head (default: best hp or config lr).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay for the classifier head (default: best hp or config value).",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=None,
        help="Optional momentum for SGD on the classifier head.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/baseline"),
        help="Directory to store baseline artifacts.",
    )
    return parser.parse_args()


def _load_best_params(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        return {}
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to parse metadata at %s: %s", metadata_path, exc)
        return {}
    return metadata.get("best_params", {})


def main() -> None:
    args = _parse_args()

    context = build_context(args.config, args.env)
    cfg = context.config
    configure_mlflow(
        cfg.storage.mlflow_tracking_uri, cfg.storage.mlflow_registry_uri
    )

    bundle = prepare_datasets(cfg.data, seed=cfg.training.seed)
    cfg.model.num_classes = len(bundle.label_names)

    metadata_path = Path("outputs/metadata.json")
    best_params = _load_best_params(metadata_path)

    batch_size = int(best_params.get("batch_size", cfg.training.batch_size))
    lr = args.lr if args.lr is not None else float(
        best_params.get("lr", cfg.training.optimizer.lr)
    )
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else float(best_params.get("weight_decay", cfg.training.optimizer.weight_decay))
    )
    epochs = args.epochs if args.epochs is not None else min(cfg.training.epochs, 5)

    dataloaders = build_dataloaders(
        bundle,
        batch_size=batch_size,
        num_workers=cfg.data.num_workers,
    )
    device = resolve_device(cfg.training.device)
    class_weights = compute_class_weights(bundle.train)

    result = run_resnet50_baseline(
        dataloaders=dataloaders,
        device=device,
        num_classes=len(bundle.label_names),
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        momentum=args.momentum,
        output_dir=args.output_dir,
        class_weights=class_weights,
    )

    run = None
    tags = {"pipeline": "baseline", "environment": cfg.environment}
    try:
        if args.run_id:
            run = mlflow.start_run(run_id=args.run_id)
            mlflow.set_tags({"baseline_logged": "true"})
        else:
            run = mlflow.start_run(run_name="baseline", tags=tags)

        baseline_params = {
            "baseline_epochs": result.epochs,
            "baseline_best_epoch": result.best_epoch,
            "baseline_lr": lr,
            "baseline_weight_decay": weight_decay,
            "baseline_batch_size": batch_size,
        }
        if args.run_id:
            existing_params = mlflow.get_run(run.info.run_id).data.params
            params_to_log = {
                key: value
                for key, value in baseline_params.items()
                if key not in existing_params
            }
            for key, value in baseline_params.items():
                if key in existing_params and str(value) != existing_params[key]:
                    LOGGER.warning(
                        "Skipping logging for param %s; existing value %s differs from %s",
                        key,
                        existing_params[key],
                        value,
                    )
            if params_to_log:
                mlflow.log_params(params_to_log)
        else:
            mlflow.log_params(baseline_params)
        mlflow.log_metric("baseline_val_loss", result.val_loss)
        mlflow.log_metric("baseline_val_accuracy", result.val_metrics.accuracy)
        mlflow.log_metric(
            "baseline_val_macro_precision", result.val_metrics.macro_precision
        )
        mlflow.log_metric(
            "baseline_val_macro_recall", result.val_metrics.macro_recall
        )
        mlflow.log_metric("baseline_val_macro_f1", result.val_metrics.macro_f1)
        mlflow.log_metric("baseline_test_loss", result.test_loss)
        mlflow.log_metric(
            "baseline_test_accuracy", result.test_metrics.accuracy
        )
        mlflow.log_metric(
            "baseline_test_macro_precision",
            result.test_metrics.macro_precision,
        )
        mlflow.log_metric(
            "baseline_test_macro_recall", result.test_metrics.macro_recall
        )
        mlflow.log_metric(
            "baseline_test_macro_f1", result.test_metrics.macro_f1
        )

        mlflow.log_artifact(str(result.model_path), artifact_path="baseline")
        mlflow.log_artifact(str(result.metadata_path), artifact_path="baseline")
        mlflow.log_artifacts(
            str(result.predictions_dir), artifact_path="baseline/predictions"
        )
        mlflow.log_dict(
            {"class_weights": class_weights.tolist()},
            "baseline/class_weights.json",
        )
    finally:
        if run is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
