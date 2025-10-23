"""
Training pipeline implementation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from mlflow import pytorch as mlflow_pytorch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from critter_capture.config import PipelineConfig
from critter_capture.data import DatasetBundle, build_dataloaders, prepare_datasets
from critter_capture.metrics.classification import ClassificationMetrics
from critter_capture.models import AnimalSpeciesCNN
from critter_capture.pipelines.base import PipelineBase, PipelineContext
from critter_capture.pipelines.training_loop import evaluate, log_epoch_metrics, train_one_epoch
from critter_capture.services import (
    build_scheduler,
    configure_mlflow,
    init_ray,
    log_config,
    run_tune,
    shutdown_ray,
    start_run,
)
from critter_capture.utils import resolve_device

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    run_id: str
    model_path: Path
    best_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    label_names: list[str]
    best_params: Dict[str, Any]
    config: PipelineConfig


def _build_model(config: PipelineConfig, overrides: Dict[str, Any]) -> AnimalSpeciesCNN:
    model_cfg = config.model.copy()
    if "dropout" in overrides:
        model_cfg.dropout = overrides["dropout"]
    model = AnimalSpeciesCNN(
        in_channels=model_cfg.in_channels,
        num_classes=model_cfg.num_classes,
        conv_filters=model_cfg.conv_filters,
        hidden_dim=model_cfg.hidden_dim,
        dropout=model_cfg.dropout,
        kernel_size=model_cfg.kernel_size,
        use_batch_norm=model_cfg.use_batch_norm,
        use_spectral_norm=model_cfg.use_spectral_norm,
    )
    return model


def _build_optimizer(model: nn.Module, config: PipelineConfig, overrides: Dict[str, Any]) -> torch.optim.Optimizer:
    opt_cfg = config.training.optimizer
    lr = overrides.get("lr", opt_cfg.lr)
    weight_decay = overrides.get("weight_decay", opt_cfg.weight_decay)

    if opt_cfg.name.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_cfg.name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_cfg.name}")

    return optimizer


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: PipelineConfig,
    overrides: Dict[str, Any],
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    sched_cfg = config.training.scheduler
    name = sched_cfg.name.lower() if sched_cfg.name else None

    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=sched_cfg.t_max, eta_min=sched_cfg.min_lr)
    if name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    return None


class TrainingPipeline(PipelineBase):
    """Run model training with optional hyperparameter tuning."""

    def __init__(self, context: PipelineContext, config_path: Path, environment: Optional[str] = None) -> None:
        super().__init__(context)
        self._config_path = config_path
        self._environment = environment

    def run(self) -> TrainingResult:
        cfg = self.context.config
        configure_mlflow(cfg.storage.mlflow_tracking_uri, cfg.storage.mlflow_registry_uri)

        bundle = prepare_datasets(cfg.data, seed=cfg.training.seed)
        cfg.model.num_classes = len(bundle.label_names)
        best_params = self._run_tuning_if_enabled(bundle)

        LOGGER.info("Training final model with params: %s", best_params)
        result = self._train_final(bundle, best_params)

        return result

    def _run_tuning_if_enabled(self, bundle: DatasetBundle) -> Dict[str, Any]:
        cfg = self.context.config
        tuning_cfg = cfg.tuning

        if not tuning_cfg.enabled:
            LOGGER.info("Hyperparameter tuning disabled in configuration.")
            return {}

        init_ray()

        def trainable(params: Dict[str, Any]) -> None:
            from ray import air

            torch.set_float32_matmul_precision("medium")

            local_cfg = self.context.config
            device = resolve_device(local_cfg.training.device)

            trial_bundle = prepare_datasets(local_cfg.data, seed=local_cfg.training.seed)
            local_cfg.model.num_classes = len(trial_bundle.label_names)
            dataloaders = build_dataloaders(
                trial_bundle,
                batch_size=int(params.get("batch_size", local_cfg.training.batch_size)),
                num_workers=local_cfg.data.num_workers,
            )
            model = _build_model(local_cfg, params).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = _build_optimizer(model, local_cfg, params)
            scheduler = _build_scheduler(optimizer, local_cfg, params)

            scaler = torch.cuda.amp.GradScaler(enabled=local_cfg.training.amp and device.type == "cuda")

            max_epochs = self.context.config.tuning.max_epochs
            best_macro_f1 = 0.0
            patience = self.context.config.training.early_stopping_patience
            patience_counter = 0

            for _ in range(1, max_epochs + 1):
                train_one_epoch(
                    model=model,
                    dataloader=dataloaders["train"],
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    scaler=scaler,
                    clip_norm=self.context.config.training.gradient_clip_norm,
                )
                val_loss, val_metrics, _ = evaluate(
                    model=model,
                    dataloader=dataloaders["validation"],
                    criterion=criterion,
                    device=device,
                )

                if scheduler and isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_metrics.macro_f1)
                elif scheduler:
                    scheduler.step()

                air.session.report(
                    {
                        "val_macro_f1": val_metrics.macro_f1,
                        "val_macro_precision": val_metrics.macro_precision,
                        "val_macro_recall": val_metrics.macro_recall,
                        "val_accuracy": val_metrics.accuracy,
                        "val_loss": val_loss,
                    }
                )

                if val_metrics.macro_f1 > best_macro_f1:
                    best_macro_f1 = val_metrics.macro_f1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        scheduler = build_scheduler(cfg.tuning.max_epochs, cfg.tuning.grace_period)
        result_grid = run_tune(
            trainable=trainable,
            search_space=cfg.tuning.search_space,
            num_samples=cfg.tuning.num_samples,
            scheduler=scheduler,
            resources_per_trial=cfg.tuning.resources_per_trial,
        )
        shutdown_ray()

        best_result = result_grid.get_best_result(metric="val_macro_f1", mode="max")
        LOGGER.info("Best tuning result: f1=%.4f config=%s", best_result.metrics["val_macro_f1"], best_result.config)
        return dict(best_result.config)

    def _train_final(self, bundle: DatasetBundle, hyperparams: Dict[str, Any]) -> TrainingResult:
        cfg = self.context.config
        batch_size = int(hyperparams.get("batch_size", cfg.training.batch_size))

        dataloaders = build_dataloaders(bundle, batch_size=batch_size, num_workers=cfg.data.num_workers)

        device = resolve_device(cfg.training.device)
        model = _build_model(cfg, hyperparams).to(device)
        optimizer = _build_optimizer(model, cfg, hyperparams)
        scheduler = _build_scheduler(optimizer, cfg, hyperparams)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.amp and device.type == "cuda")

        experiment_name = "animal_species_multiclass"
        tags = {"pipeline": "training", "environment": cfg.environment}
        run = start_run(experiment_name=experiment_name, run_name="training", tags=tags)

        with run:
            mlflow_pytorch.autolog(log_models=False)
            Path("outputs").mkdir(parents=True, exist_ok=True)
            mlflow.log_params(
                {
                    "batch_size": batch_size,
                    "dropout": hyperparams.get("dropout", cfg.model.dropout),
                    "lr": hyperparams.get("lr", cfg.training.optimizer.lr),
                    "weight_decay": hyperparams.get("weight_decay", cfg.training.optimizer.weight_decay),
                }
            )
            log_config(json.loads(cfg.json()))

            best_val_f1 = float("-inf")
            patience_counter = 0
            epochs = cfg.training.epochs

            for epoch in range(1, epochs + 1):
                LOGGER.info("Epoch %d/%d", epoch, epochs)
                train_loss = train_one_epoch(
                    model=model,
                    dataloader=dataloaders["train"],
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    scaler=scaler,
                    clip_norm=cfg.training.gradient_clip_norm,
                )
                val_loss, val_metrics, _ = evaluate(
                    model=model,
                    dataloader=dataloaders["validation"],
                    criterion=criterion,
                    device=device,
                )

                mlflow.log_metric("train_loss", train_loss, step=epoch)
                log_epoch_metrics("val", val_loss, val_metrics, epoch)

                if scheduler and isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_metrics.macro_f1)
                elif scheduler:
                    scheduler.step()

                if val_metrics.macro_f1 > best_val_f1:
                    best_val_f1 = val_metrics.macro_f1
                    patience_counter = 0
                    torch.save(model.state_dict(), "outputs/best_model.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= cfg.training.early_stopping_patience:
                        LOGGER.info("Early stopping triggered at epoch %d", epoch)
                        break

            checkpoint_path = Path("outputs/best_model.pt")
            if not checkpoint_path.exists():
                torch.save(model.state_dict(), checkpoint_path)

            model.load_state_dict(torch.load(checkpoint_path, map_location=device))

            val_loss, val_metrics, val_outputs = evaluate(
                model=model,
                dataloader=dataloaders["validation"],
                criterion=criterion,
                device=device,
            )
            test_loss, test_metrics, test_outputs = evaluate(
                model=model,
                dataloader=dataloaders["test"],
                criterion=criterion,
                device=device,
            )

            mlflow.log_metric("final_val_accuracy", val_metrics.accuracy)
            mlflow.log_metric("final_val_macro_precision", val_metrics.macro_precision)
            mlflow.log_metric("final_val_macro_recall", val_metrics.macro_recall)
            mlflow.log_metric("final_val_macro_f1", val_metrics.macro_f1)
            mlflow.log_metric("final_test_accuracy", test_metrics.accuracy)
            mlflow.log_metric("final_test_macro_precision", test_metrics.macro_precision)
            mlflow.log_metric("final_test_macro_recall", test_metrics.macro_recall)
            mlflow.log_metric("final_test_macro_f1", test_metrics.macro_f1)
            mlflow.log_artifact("outputs/best_model.pt", artifact_path="model")
            mlflow_pytorch.log_model(model, artifact_path="model_artifact")

            metadata = {
                "label_names": bundle.label_names,
                "label_ids": list(bundle.label_ids),
                "validation_metrics": asdict(val_metrics),
                "test_metrics": asdict(test_metrics),
                "best_params": hyperparams,
            }
            Path("outputs/predictions").mkdir(parents=True, exist_ok=True)
            np.save("outputs/predictions/val_y_true.npy", val_outputs["y_true"])
            np.save("outputs/predictions/val_y_scores.npy", val_outputs["y_scores"])
            np.save("outputs/predictions/val_y_pred.npy", val_outputs["y_pred"])
            np.save("outputs/predictions/test_y_true.npy", test_outputs["y_true"])
            np.save("outputs/predictions/test_y_scores.npy", test_outputs["y_scores"])
            np.save("outputs/predictions/test_y_pred.npy", test_outputs["y_pred"])
            mlflow.log_artifacts("outputs/predictions", artifact_path="predictions")

            metadata_path = Path("outputs/metadata.json")
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(metadata_path), artifact_path="metadata")
            mlflow.log_dict(hyperparams, "hyperparams/best_params.json")

            run_id = run.info.run_id

        result = TrainingResult(
            run_id=run_id,
            model_path=Path("outputs/best_model.pt"),
            best_metrics=val_metrics,
            test_metrics=test_metrics,
            label_names=list(bundle.label_names),
            best_params=hyperparams,
            config=cfg,
        )

        return result


def run_training_pipeline(config_path: Path, environment: Optional[str] = None) -> TrainingResult:
    from critter_capture.pipelines.base import build_context

    context = build_context(config_path, environment)
    pipeline = TrainingPipeline(context=context, config_path=config_path, environment=environment)
    return pipeline.run()


__all__ = ["TrainingPipeline", "TrainingResult", "run_training_pipeline"]
