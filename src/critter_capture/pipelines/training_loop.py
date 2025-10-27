"""Model training and evaluation loop utilities."""

from __future__ import annotations

import logging
import re
from typing import Dict, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from critter_capture.metrics.classification import (
    ClassificationMetrics,
    compute_classification_metrics,
)

LOGGER = logging.getLogger(__name__)


# Retained for backwards compatibility if needed.


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    clip_norm: float | None = None,
) -> float:
    """Train the model for a single epoch and return the average loss."""

    model.train()
    running_loss = 0.0

    use_amp = scaler is not None and scaler.is_enabled()

    for batch in tqdm(dataloader, desc="train", leave=False):
        inputs = batch["image"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(inputs)
            loss = criterion(logits, targets)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if clip_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / max(len(dataloader), 1)
    return average_loss


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, ClassificationMetrics, Dict[str, np.ndarray]]:
    """Evaluate the model and compute metrics."""

    model.eval()
    running_loss = 0.0

    y_true = []
    y_scores = []
    y_pred = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="eval", leave=False):
            inputs = batch["image"].to(device)
            targets = batch["target"].to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)
            running_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            y_true.append(targets.cpu().numpy())
            y_scores.append(probs.cpu().numpy())
            y_pred.append(torch.argmax(probs, dim=1).cpu().numpy())

    avg_loss = running_loss / max(len(dataloader), 1)
    y_true_arr = np.concatenate(y_true, axis=0)
    y_scores_arr = np.concatenate(y_scores, axis=0)
    y_pred_arr = np.concatenate(y_pred, axis=0)

    metrics = compute_classification_metrics(
        y_true=y_true_arr,
        y_pred=y_pred_arr,
        label_names=dataloader.dataset.label_names,  # type: ignore[attr-defined]
    )

    return (
        avg_loss,
        metrics,
        {"y_true": y_true_arr, "y_scores": y_scores_arr, "y_pred": y_pred_arr},
    )


def log_epoch_metrics(
    phase: str, loss: float, metrics: ClassificationMetrics, epoch: int
) -> None:
    """Log metrics to MLflow."""

    mlflow.log_metric(f"{phase}_loss", loss, step=epoch)
    mlflow.log_metric(f"{phase}_accuracy", metrics.accuracy, step=epoch)
    mlflow.log_metric(
        f"{phase}_macro_precision",
        metrics.macro_precision,
        step=epoch,
    )
    mlflow.log_metric(
        f"{phase}_macro_recall",
        metrics.macro_recall,
        step=epoch,
    )
    mlflow.log_metric(f"{phase}_macro_f1", metrics.macro_f1, step=epoch)
    for label_name, stats in metrics.per_class.items():
        safe_label = re.sub(
            r"[^0-9a-zA-Z_\-./:]+", "_", label_name.lower().replace(" ", "_")
        )
        for stat_name, value in stats.items():
            mlflow.log_metric(
                f"{phase}_{safe_label}_{stat_name}",
                value,
                step=epoch,
            )


__all__ = ["evaluate", "log_epoch_metrics", "train_one_epoch"]
