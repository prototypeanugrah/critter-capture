"""Utilities for running a ResNet-50 baseline."""

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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50

from critter_capture.metrics.classification import ClassificationMetrics
from critter_capture.pipelines.training_loop import evaluate

LOGGER = logging.getLogger(__name__)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_cfg: Any | None,
    epochs: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if scheduler_cfg is None:
        return None

    name = getattr(scheduler_cfg, "name", None)
    if not name:
        return None

    name = name.lower()
    if name == "cosine":
        t_max = getattr(scheduler_cfg, "t_max", epochs)
        eta_min = getattr(scheduler_cfg, "min_lr", 0.0)
        return CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )

    if name == "plateau":
        patience = getattr(scheduler_cfg, "patience", 3)
        factor = getattr(scheduler_cfg, "factor", 0.5)
        mode = getattr(scheduler_cfg, "mode", "max")
        return ReduceLROnPlateau(
            optimizer,
            mode=mode,
            patience=patience,
            factor=factor,
        )

    LOGGER.warning(
        "Unsupported scheduler '%s' provided to baseline; skipping scheduler setup.",
        name,
    )
    return None


@dataclass
class ResNet50BaselineResult:
    """Results from running the ResNet-50 baseline."""

    model_path: Path
    predictions_dir: Path
    metadata_path: Path
    val_loss: float
    val_metrics: ClassificationMetrics
    test_loss: float
    test_metrics: ClassificationMetrics
    best_epoch: int
    epochs: int


def _build_model(num_classes: int) -> nn.Module:
    """Load torchvision ResNet-50 with ImageNet weights and reset classifier."""

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    classifier = nn.Linear(in_features, num_classes)
    nn.init.xavier_uniform_(classifier.weight)
    if classifier.bias is not None:
        nn.init.zeros_(classifier.bias)

    model.fc = classifier
    return model


def run_resnet50_baseline(
    dataloaders: Dict[str, DataLoader],
    device: torch.device,
    num_classes: int,
    *,
    epochs: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    scheduler_cfg: Any | None = None,
    output_dir: Path | None = None,
    class_weights: torch.Tensor | None = None,
    mlflow_logging: bool = False,
    mlflow_prefix: str = "baseline",
) -> ResNet50BaselineResult:
    """Fine-tune the classifier head of a pretrained ResNet-50 and evaluate it."""

    if (
        "train" not in dataloaders
        or "validation" not in dataloaders
        or "test" not in dataloaders
    ):
        raise ValueError(
            "Expected dataloaders dict with 'train', 'validation', and 'test' keys."
        )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["validation"]
    test_loader = dataloaders["test"]

    if output_dir is None:
        output_dir = Path("outputs/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    model = _build_model(num_classes=num_classes).to(device)
    weight_tensor = class_weights.to(device) if class_weights is not None else None
    criterion_train = nn.CrossEntropyLoss(weight=weight_tensor)
    criterion_eval = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.fc.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = _build_scheduler(optimizer, scheduler_cfg, epochs)

    best_val_acc = float("-inf")
    best_epoch = -1
    best_state: Dict[str, torch.Tensor] | None = None
    best_val = (float("inf"), None, None)  # loss, metrics, outputs

    log_metrics = mlflow_logging
    if log_metrics and mlflow.active_run() is None:
        LOGGER.warning(
            "MLflow logging requested for baseline run but no active MLflow run found. "
            "Skipping live logging."
        )
        log_metrics = False

    LOGGER.info(
        "Starting ResNet-50 baseline fine-tuning for %d epochs (lr=%.4g, weight_decay=%.4g)",
        epochs,
        lr,
        weight_decay,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            inputs = batch["image"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion_train(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / max(len(train_loader), 1)

        val_loss, val_metrics, val_outputs = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion_eval,
            device=device,
        )

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics.accuracy)
            else:
                scheduler.step()

        LOGGER.info(
            "Baseline epoch %d/%d - train_loss=%.4f, val_loss=%.4f, val_accuracy=%.4f",
            epoch,
            epochs,
            train_loss,
            val_loss,
            val_metrics.accuracy,
        )

        if log_metrics:
            mlflow.log_metric(f"{mlflow_prefix}_train_loss", train_loss, step=epoch)
            mlflow.log_metric(f"{mlflow_prefix}_val_loss", val_loss, step=epoch)
            mlflow.log_metric(
                f"{mlflow_prefix}_val_accuracy", val_metrics.accuracy, step=epoch
            )
            mlflow.log_metric(
                f"{mlflow_prefix}_val_macro_precision",
                val_metrics.macro_precision,
                step=epoch,
            )
            mlflow.log_metric(
                f"{mlflow_prefix}_val_macro_recall",
                val_metrics.macro_recall,
                step=epoch,
            )
            mlflow.log_metric(
                f"{mlflow_prefix}_val_macro_f1", val_metrics.macro_f1, step=epoch
            )

        if val_metrics.accuracy >= best_val_acc:
            best_val_acc = val_metrics.accuracy
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            }
            best_val = (val_loss, val_metrics, val_outputs)

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        LOGGER.warning(
            "Baseline fine-tuning did not improve validation accuracy; using final epoch weights."
        )

    val_loss, val_metrics, val_outputs = best_val
    if val_metrics is None or val_outputs is None:
        val_loss, val_metrics, val_outputs = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion_eval,
            device=device,
        )

    test_loss, test_metrics, test_outputs = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion_eval,
        device=device,
    )

    if log_metrics:
        mlflow.log_metric(f"{mlflow_prefix}_test_loss", test_loss, step=epochs)
        mlflow.log_metric(
            f"{mlflow_prefix}_test_accuracy", test_metrics.accuracy, step=epochs
        )
        mlflow.log_metric(
            f"{mlflow_prefix}_test_macro_precision",
            test_metrics.macro_precision,
            step=epochs,
        )
        mlflow.log_metric(
            f"{mlflow_prefix}_test_macro_recall",
            test_metrics.macro_recall,
            step=epochs,
        )
        mlflow.log_metric(
            f"{mlflow_prefix}_test_macro_f1", test_metrics.macro_f1, step=epochs
        )

    np.save(predictions_dir / "val_y_true.npy", val_outputs["y_true"])
    np.save(predictions_dir / "val_y_scores.npy", val_outputs["y_scores"])
    np.save(predictions_dir / "val_y_pred.npy", val_outputs["y_pred"])
    np.save(predictions_dir / "test_y_true.npy", test_outputs["y_true"])
    np.save(predictions_dir / "test_y_scores.npy", test_outputs["y_scores"])
    np.save(predictions_dir / "test_y_pred.npy", test_outputs["y_pred"])

    model_path = output_dir / "resnet50_baseline.pt"
    torch.save(model.state_dict(), model_path)

    class_weights_list = (
        class_weights.cpu().tolist() if class_weights is not None else None
    )

    metadata = {
        "epochs": epochs,
        "best_epoch": best_epoch,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "val_loss": val_loss,
        "val_metrics": asdict(val_metrics),
        "test_loss": test_loss,
        "test_metrics": asdict(test_metrics),
        "class_weights": class_weights_list,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return ResNet50BaselineResult(
        model_path=model_path,
        predictions_dir=predictions_dir,
        metadata_path=metadata_path,
        val_loss=val_loss,
        val_metrics=val_metrics,
        test_loss=test_loss,
        test_metrics=test_metrics,
        best_epoch=best_epoch,
        epochs=epochs,
    )


__all__ = ["ResNet50BaselineResult", "run_resnet50_baseline"]
