"""Utility helpers for deterministic seeding."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy, and torch for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(preferred: str = "cuda") -> torch.device:
    """Resolve the torch device based on availability and preference."""

    preferred_lower = preferred.lower()
    if preferred_lower == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif preferred_lower == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


__all__ = ["seed_everything", "resolve_device"]
