"""Baseline model utilities."""

from .resnet18 import ResNet18BaselineResult, run_resnet18_baseline
from .resnet50 import ResNet50BaselineResult, run_resnet50_baseline

__all__ = [
    "ResNet18BaselineResult",
    "run_resnet18_baseline",
    "ResNet50BaselineResult",
    "run_resnet50_baseline",
]
