"""
Training pipeline implementation.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from mlflow import pytorch as mlflow_pytorch
from ray.air import session
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet50

from critter_capture.baselines import run_resnet18_baseline, run_resnet50_baseline
from critter_capture.config import PipelineConfig
from critter_capture.data import (
    DatasetBundle,
    build_dataloaders,
    compute_class_weights,
    prepare_datasets,
)
from critter_capture.metrics.classification import ClassificationMetrics
from critter_capture.models import AnimalSpeciesCNN
from critter_capture.pipelines.base import (
    PipelineBase,
    PipelineContext,
    build_context,
)
from critter_capture.pipelines.training_loop import (
    evaluate,
    log_epoch_metrics,
    train_one_epoch,
)
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
    """
    Result of the training pipeline.

    Args:
        run_id (str): The ID of the run.
        model_path (Path): The path to the model.
        best_metrics (ClassificationMetrics): The best metrics.
        test_metrics (ClassificationMetrics): The test metrics.
        label_names (list[str]): The label names.
        best_params (Dict[str, Any]): The best hyperparameters.
        config (PipelineConfig): The configuration.
        model_variant (str): Identifier for the trained model type.
    """

    run_id: str
    model_path: Path
    best_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    label_names: list[str]
    best_params: Dict[str, Any]
    config: PipelineConfig
    model_variant: str


def _build_model(
    config: PipelineConfig,
    overrides: Dict[str, Any],
) -> AnimalSpeciesCNN:
    model_cfg = config.model.copy()
    if "dropout" in overrides:
        model_cfg.dropout = overrides["dropout"]
    model = AnimalSpeciesCNN(
        in_channels=model_cfg.in_channels,
        num_classes=model_cfg.num_classes,
        conv_filters=model_cfg.conv_filters,
        conv_layers_per_block=model_cfg.conv_layers_per_block,
        hidden_dim=model_cfg.hidden_dim,
        dropout=model_cfg.dropout,
        second_hidden_dim=model_cfg.second_hidden_dim,
        kernel_size=model_cfg.kernel_size,
        use_batch_norm=model_cfg.use_batch_norm,
        use_spectral_norm=model_cfg.use_spectral_norm,
    )
    return model


def _build_optimizer(
    model: nn.Module, config: PipelineConfig, overrides: Dict[str, Any]
) -> torch.optim.Optimizer:
    opt_cfg = config.training.optimizer
    lr = overrides.get("lr", opt_cfg.lr)
    weight_decay = overrides.get("weight_decay", opt_cfg.weight_decay)

    if opt_cfg.name.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif opt_cfg.name.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_cfg.name}")

    return optimizer


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: PipelineConfig,
    loader: DataLoader,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    sched_cfg = config.training.scheduler
    name = sched_cfg.name.lower() if sched_cfg.name else None

    if name == "onecycle":
        max_lr = sched_cfg.max_lr
        total_steps = config.training.epochs * len(loader)
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
        )
    return None


def _build_baseline_model_for_logging(
    baseline_name: str,
    num_classes: int,
) -> nn.Module:
    """Construct a baseline model instance for saving/logging purposes."""

    if baseline_name == "resnet18":
        model = resnet18(weights=None)
    elif baseline_name == "resnet50":
        model = resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported baseline: {baseline_name}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


class TrainingPipeline(PipelineBase):
    """Run model training with optional hyperparameter tuning."""

    def __init__(
        self,
        context: PipelineContext,
        config_path: Path,
        environment: Optional[str] = None,
    ) -> None:
        super().__init__(context)
        self._config_path = config_path
        self._environment = environment

    def run(self) -> TrainingResult:
        cfg = self.context.config
        configure_mlflow(
            cfg.storage.mlflow_tracking_uri, cfg.storage.mlflow_registry_uri
        )

        bundle = prepare_datasets(cfg.data, seed=cfg.training.seed)
        cfg.model.num_classes = len(bundle.label_names)
        best_params = self._run_tuning_if_enabled()

        LOGGER.info("Training final model with params: %s", best_params)
        result = self._train_final(bundle, best_params)

        return result

    def _run_tuning_if_enabled(self) -> Dict[str, Any]:
        """Run hyperparameter tuning if enabled."""
        cfg = self.context.config
        tuning_cfg = cfg.tuning

        if not tuning_cfg.enabled:
            LOGGER.info("Hyperparameter tuning disabled in configuration.")
            return {}

        init_ray()

        def trainable(params: Dict[str, Any]) -> None:
            torch.set_float32_matmul_precision("medium")

            local_cfg = self.context.config
            device = resolve_device(local_cfg.training.device)

            trial_bundle = prepare_datasets(
                local_cfg.data, seed=local_cfg.training.seed
            )
            local_cfg.model.num_classes = len(trial_bundle.label_names)
            class_weights = compute_class_weights(trial_bundle.train)
            dataloaders = build_dataloaders(
                trial_bundle,
                batch_size=int(
                    params.get(
                        "batch_size",
                        local_cfg.training.batch_size,
                    )
                ),
                num_workers=local_cfg.data.num_workers,
            )
            model = _build_model(local_cfg, params).to(device)
            criterion_train = nn.CrossEntropyLoss(weight=class_weights.to(device))
            criterion_eval = nn.CrossEntropyLoss()
            optimizer = _build_optimizer(model, local_cfg, params)
            scheduler = _build_scheduler(optimizer, local_cfg, dataloaders["train"])

            scaler = torch.amp.GradScaler(
                device=device.type,
                enabled=local_cfg.training.amp and device.type == "cuda",
            )

            max_epochs = self.context.config.tuning.max_epochs
            best_val_acc = float("-inf")
            patience = self.context.config.training.early_stopping_patience
            patience_counter = 0

            for _ in range(1, max_epochs + 1):
                train_loss, train_metrics = train_one_epoch(
                    model=model,
                    dataloader=dataloaders["train"],
                    criterion=criterion_train,
                    optimizer=optimizer,
                    device=device,
                    scaler=scaler,
                    clip_norm=self.context.config.training.gradient_clip_norm,
                    scheduler=scheduler,
                )
                val_loss, val_metrics, _ = evaluate(
                    model=model,
                    dataloader=dataloaders["validation"],
                    criterion=criterion_eval,
                    device=device,
                )

                session.report(
                    {
                        "val_macro_f1": val_metrics.macro_f1,
                        "val_macro_precision": val_metrics.macro_precision,
                        "val_macro_recall": val_metrics.macro_recall,
                        "val_accuracy": val_metrics.accuracy,
                        "val_loss": val_loss,
                        "train_loss": train_loss,
                        "train_accuracy": train_metrics.accuracy,
                    }
                )

                if val_metrics.accuracy > best_val_acc:
                    best_val_acc = val_metrics.accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        metric_key = cfg.tuning.metric
        metric_mode = cfg.tuning.mode

        scheduler = build_scheduler(
            cfg.tuning.max_epochs,
            cfg.tuning.grace_period,
            metric_key,
            metric_mode,
        )
        result_grid = run_tune(
            trainable=trainable,
            search_space=cfg.tuning.search_space,
            num_samples=cfg.tuning.num_samples,
            scheduler=scheduler,
            resources_per_trial=cfg.tuning.resources_per_trial,
            metric=metric_key,
            mode=metric_mode,
        )
        shutdown_ray()

        best_result = result_grid.get_best_result(
            metric=metric_key,
            mode=metric_mode,
        )
        LOGGER.info(
            "Best tuning result: f1=%.4f config=%s",
            best_result.metrics["val_macro_f1"],
            best_result.config,
        )
        return dict(best_result.config)

    def _train_final(
        self, bundle: DatasetBundle, hyperparams: Dict[str, Any]
    ) -> TrainingResult:
        """Train the final model.

        Args:
            bundle (DatasetBundle): The dataset bundle.
            hyperparams (Dict[str, Any]): The hyperparameters.

        Returns:
            TrainingResult: The result of the training pipeline.
        """
        cfg = self.context.config
        batch_size = int(
            hyperparams.get(
                "batch_size",
                cfg.training.batch_size,
            )
        )

        dataloaders = build_dataloaders(
            bundle,
            batch_size=batch_size,
            num_workers=cfg.data.num_workers,
        )
        class_weights = compute_class_weights(bundle.train)

        device = resolve_device(cfg.training.device)
        class_weights_device = class_weights.to(device)
        baseline_lr = float(hyperparams.get("lr", cfg.training.optimizer.lr))
        baseline_weight_decay = float(
            hyperparams.get(
                "weight_decay",
                cfg.training.optimizer.weight_decay,
            )
        )

        experiment_name = "animal_species_multiclass"
        tags = {"pipeline": "training", "environment": cfg.environment}
        run = start_run(
            experiment_name=experiment_name,
            run_name="training",
            tags=tags,
        )

        with run:
            Path("outputs").mkdir(parents=True, exist_ok=True)
            mlflow.log_params(
                {
                    "batch_size": batch_size,
                    "dropout": hyperparams.get("dropout", cfg.model.dropout),
                    "lr": hyperparams.get("lr", cfg.training.optimizer.lr),
                    "weight_decay": hyperparams.get(
                        "weight_decay", cfg.training.optimizer.weight_decay
                    ),
                }
            )
            log_config(json.loads(cfg.json()))

            if cfg.training.baseline == "resnet18":
                baseline_result = run_resnet18_baseline(
                    dataloaders=dataloaders,
                    device=device,
                    num_classes=len(bundle.label_names),
                    epochs=min(cfg.training.epochs, 5),
                    lr=baseline_lr,
                    weight_decay=baseline_weight_decay,
                    scheduler_cfg=cfg.training.scheduler,
                    output_dir=Path("outputs/baseline"),
                    class_weights=class_weights,
                    mlflow_logging=True,
                    mlflow_prefix="baseline_resnet18",
                )
            elif cfg.training.baseline == "resnet50":
                baseline_result = run_resnet50_baseline(
                    dataloaders=dataloaders,
                    device=device,
                    num_classes=len(bundle.label_names),
                    epochs=min(cfg.training.epochs, 5),
                    lr=baseline_lr,
                    weight_decay=baseline_weight_decay,
                    scheduler_cfg=cfg.training.scheduler,
                    output_dir=Path("outputs/baseline"),
                    class_weights=class_weights,
                    mlflow_logging=True,
                    mlflow_prefix="baseline_resnet50",
                )
            else:
                raise ValueError(f"Unsupported baseline: {cfg.training.baseline}")

            mlflow.log_params(
                {
                    "baseline_epochs": baseline_result.epochs,
                    "baseline_best_epoch": baseline_result.best_epoch,
                    "baseline_lr": baseline_lr,
                    "baseline_weight_decay": baseline_weight_decay,
                    "baseline_batch_size": batch_size,
                }
            )
            # metric_step = (
            #     baseline_result.best_epoch
            #     if baseline_result.best_epoch != -1
            #     else baseline_result.epochs
            # )
            # mlflow.log_metric(
            #     "baseline_val_loss",
            #     baseline_result.val_loss,
            #     step=metric_step,
            # )
            # mlflow.log_metric(
            #     "baseline_val_accuracy",
            #     baseline_result.val_metrics.accuracy,
            #     step=metric_step,
            # )
            # mlflow.log_metric(
            #     "baseline_val_macro_precision",
            #     baseline_result.val_metrics.macro_precision,
            #     step=metric_step,
            # )
            # mlflow.log_metric(
            #     "baseline_val_macro_recall",
            #     baseline_result.val_metrics.macro_recall,
            #     step=metric_step,
            # )
            # mlflow.log_metric(
            #     "baseline_val_macro_f1",
            #     baseline_result.val_metrics.macro_f1,
            #     step=metric_step,
            # )
            # mlflow.log_metric(
            #     "baseline_test_loss",
            #     baseline_result.test_loss,
            #     step=baseline_result.epochs,
            # )
            # mlflow.log_metric(
            #     "baseline_test_accuracy",
            #     baseline_result.test_metrics.accuracy,
            #     step=baseline_result.epochs,
            # )
            # mlflow.log_metric(
            #     "baseline_test_macro_precision",
            #     baseline_result.test_metrics.macro_precision,
            #     step=baseline_result.epochs,
            # )
            # mlflow.log_metric(
            #     "baseline_test_macro_recall",
            #     baseline_result.test_metrics.macro_recall,
            #     step=baseline_result.epochs,
            # )
            # mlflow.log_metric(
            #     "baseline_test_macro_f1",
            #     baseline_result.test_metrics.macro_f1,
            #     step=baseline_result.epochs,
            # )
            mlflow.log_param("full_training", cfg.training.full_training)

            predictions_dir = Path("outputs/predictions")
            predictions_dir.mkdir(parents=True, exist_ok=True)
            for existing_file in predictions_dir.glob("*.npy"):
                existing_file.unlink()

            final_model: nn.Module
            final_model_path: Path
            final_val_metrics: ClassificationMetrics
            final_test_metrics: ClassificationMetrics
            val_outputs: Optional[Dict[str, np.ndarray]] = None
            test_outputs: Optional[Dict[str, np.ndarray]] = None
            final_model_variant = f"baseline_{cfg.training.baseline}"

            if cfg.training.full_training:
                mlflow_pytorch.autolog(log_models=False)
                model = _build_model(cfg, hyperparams).to(device)
                optimizer = _build_optimizer(model, cfg, hyperparams)
                scheduler = _build_scheduler(optimizer, cfg, dataloaders["train"])
                criterion_train = nn.CrossEntropyLoss(weight=class_weights_device)
                criterion_eval = nn.CrossEntropyLoss()
                scaler = torch.amp.GradScaler(
                    device=device.type,
                    enabled=cfg.training.amp and device.type == "cuda",
                )

                best_val_acc = float("-inf")
                patience_counter = 0
                epochs = cfg.training.epochs

                for epoch in range(1, epochs + 1):
                    LOGGER.info("Epoch %d/%d", epoch, epochs)
                    train_loss, train_metrics = train_one_epoch(
                        model=model,
                        dataloader=dataloaders["train"],
                        criterion=criterion_train,
                        optimizer=optimizer,
                        device=device,
                        scaler=scaler,
                        clip_norm=cfg.training.gradient_clip_norm,
                        scheduler=scheduler,
                    )
                    val_loss, val_metrics, _ = evaluate(
                        model=model,
                        dataloader=dataloaders["validation"],
                        criterion=criterion_eval,
                        device=device,
                    )

                    log_epoch_metrics("train", train_loss, train_metrics, epoch)
                    log_epoch_metrics("val", val_loss, val_metrics, epoch)

                    if val_metrics.accuracy > best_val_acc:
                        best_val_acc = val_metrics.accuracy
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

                model.load_state_dict(
                    torch.load(
                        checkpoint_path,
                        map_location=device,
                    )
                )

                val_loss, final_val_metrics, val_outputs = evaluate(
                    model=model,
                    dataloader=dataloaders["validation"],
                    criterion=criterion_eval,
                    device=device,
                )
                _, final_test_metrics, test_outputs = evaluate(
                    model=model,
                    dataloader=dataloaders["test"],
                    criterion=criterion_eval,
                    device=device,
                )

                final_model = model
                final_model_path = checkpoint_path
                final_model_variant = "cnn"
            else:
                baseline_state = torch.load(
                    baseline_result.model_path,
                    map_location="cpu",
                )
                final_model = _build_baseline_model_for_logging(
                    cfg.training.baseline,
                    len(bundle.label_names),
                )
                final_model.load_state_dict(baseline_state)
                final_model_path = Path("outputs/best_model.pt")
                torch.save(final_model.state_dict(), final_model_path)
                final_val_metrics = baseline_result.val_metrics
                final_test_metrics = baseline_result.test_metrics

                for filename in [
                    "val_y_true.npy",
                    "val_y_scores.npy",
                    "val_y_pred.npy",
                    "test_y_true.npy",
                    "test_y_scores.npy",
                    "test_y_pred.npy",
                ]:
                    source_file = baseline_result.predictions_dir / filename
                    if source_file.exists():
                        shutil.copy2(source_file, predictions_dir / filename)

            if val_outputs is not None and test_outputs is not None:
                np.save(predictions_dir / "val_y_true.npy", val_outputs["y_true"])
                np.save(predictions_dir / "val_y_scores.npy", val_outputs["y_scores"])
                np.save(predictions_dir / "val_y_pred.npy", val_outputs["y_pred"])
                np.save(predictions_dir / "test_y_true.npy", test_outputs["y_true"])
                np.save(
                    predictions_dir / "test_y_scores.npy",
                    test_outputs["y_scores"],
                )
                np.save(predictions_dir / "test_y_pred.npy", test_outputs["y_pred"])

            mlflow.log_param("model_variant", final_model_variant)
            mlflow.set_tag("model_variant", final_model_variant)

            # Log final validation and test metrics
            LOGGER.info("Logging final validation and test metrics")
            mlflow.log_metric(
                "final_val_accuracy",
                final_val_metrics.accuracy,
            )
            mlflow.log_metric(
                "final_val_macro_precision",
                final_val_metrics.macro_precision,
            )
            mlflow.log_metric(
                "final_val_macro_recall",
                final_val_metrics.macro_recall,
            )
            mlflow.log_metric(
                "final_val_macro_f1",
                final_val_metrics.macro_f1,
            )
            mlflow.log_metric(
                "final_test_accuracy",
                final_test_metrics.accuracy,
            )
            mlflow.log_metric(
                "final_test_macro_precision",
                final_test_metrics.macro_precision,
            )
            mlflow.log_metric(
                "final_test_macro_recall",
                final_test_metrics.macro_recall,
            )
            mlflow.log_metric(
                "final_test_macro_f1",
                final_test_metrics.macro_f1,
            )

            # Log final model and artifacts
            LOGGER.info("Logging final model and artifacts")
            mlflow.log_artifact(str(final_model_path), artifact_path="model")
            mlflow_pytorch.log_model(
                final_model.to("cpu"), artifact_path="model_artifact"
            )

            LOGGER.info("Logging predictions")
            mlflow.log_artifacts(str(predictions_dir), artifact_path="predictions")

            LOGGER.info("Logging metadata for final model")
            metadata = {
                "label_names": bundle.label_names,
                "label_ids": list(bundle.label_ids),
                "validation_metrics": asdict(final_val_metrics),
                "test_metrics": asdict(final_test_metrics),
                "best_params": hyperparams,
                "class_weights": class_weights.tolist(),
                "model_variant": final_model_variant,
                "full_training": cfg.training.full_training,
                "baseline": {
                    "name": cfg.training.baseline,
                    "val_loss": baseline_result.val_loss,
                    "test_loss": baseline_result.test_loss,
                    "val_metrics": asdict(baseline_result.val_metrics),
                    "test_metrics": asdict(baseline_result.test_metrics),
                    "best_epoch": baseline_result.best_epoch,
                    "epochs": baseline_result.epochs,
                    "lr": baseline_lr,
                    "weight_decay": baseline_weight_decay,
                },
            }

            metadata_path = Path("outputs/metadata.json")
            metadata_path.write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(metadata_path), artifact_path="metadata")
            mlflow.log_dict(hyperparams, "hyperparams/best_params.json")

            run_id = run.info.run_id

        result = TrainingResult(
            run_id=run_id,
            model_path=final_model_path,
            best_metrics=final_val_metrics,
            test_metrics=final_test_metrics,
            label_names=list(bundle.label_names),
            best_params=hyperparams,
            config=cfg,
            model_variant=final_model_variant,
        )

        return result


def run_training_pipeline(
    config_path: Path, environment: Optional[str] = None
) -> TrainingResult:
    """Run the training pipeline.

    Args:
        config_path (Path): The path to the configuration file.
        environment (Optional[str], optional): The environment to use. Defaults to None.

    Returns:
        TrainingResult: The result of the training pipeline.
    """

    context = build_context(config_path, environment)
    pipeline = TrainingPipeline(
        context=context, config_path=config_path, environment=environment
    )
    return pipeline.run()


__all__ = ["TrainingPipeline", "TrainingResult", "run_training_pipeline"]
