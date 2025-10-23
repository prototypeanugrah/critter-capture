"""Dataset splitting utilities for stratified multi-class datasets."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)


def stratified_train_val_test_split(
    labels: np.ndarray,
    validation_size: float,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create stratified train/validation/test splits for multi-class data."""

    if validation_size + test_size >= 0.9:
        raise ValueError("Combined validation and test size should leave reasonable train data.")

    indices = np.arange(len(labels))
    stratify_labels = labels if np.min(np.bincount(labels)) >= 2 else None
    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=validation_size + test_size,
        random_state=seed,
        stratify=stratify_labels,
    )

    holdout_labels = labels[holdout_idx]
    stratify_holdout = holdout_labels if np.min(np.bincount(holdout_labels)) >= 2 else None
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=test_size / (validation_size + test_size),
        random_state=seed,
        stratify=stratify_holdout,
    )

    LOGGER.info(
        "Created stratified splits: train=%d, val=%d, test=%d",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )

    return train_idx, val_idx, test_idx


__all__ = ["stratified_train_val_test_split"]
