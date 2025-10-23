"""Convolutional neural network for multi-class classification."""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


def _make_conv_block(in_channels: int, out_channels: int, kernel_size: int, use_batch_norm: bool) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
    ]
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(kernel_size=2))
    return nn.Sequential(*layers)


class AnimalSpeciesCNN(nn.Module):
    """Simple CNN with five convolutional blocks and a classifier head."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_filters: Sequence[int],
        hidden_dim: int,
        dropout: float,
        kernel_size: int = 3,
        use_batch_norm: bool = True,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()

        filters = list(conv_filters)
        blocks = []
        prev_channels = in_channels
        for out_channels in filters:
            block = _make_conv_block(prev_channels, out_channels, kernel_size, use_batch_norm)
            if use_spectral_norm:
                block[0] = nn.utils.spectral_norm(block[0])
            blocks.append(block)
            prev_channels = out_channels

        self.feature_extractor = nn.Sequential(*blocks)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters[-1], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        pooled = self.avg_pool(features)
        logits = self.classifier(pooled)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

__all__ = ["AnimalSpeciesCNN"]
