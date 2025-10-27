"""Convolutional neural network for multi-class classification."""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


def _make_conv_block(
    in_channels: int,
    out_channels: int,
    num_layers: int,
    kernel_size: int,
    use_batch_norm: bool,
    use_spectral_norm: bool,
) -> nn.Sequential:
    """Construct a VGG-style convolutional block."""

    layers: list[nn.Module] = []
    current_in = in_channels
    for _ in range(num_layers):
        conv = nn.Conv2d(
            current_in,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        if use_spectral_norm:
            conv = nn.utils.spectral_norm(conv)
        layers.append(conv)
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        current_in = out_channels

    layers.append(nn.MaxPool2d(kernel_size=2))
    return nn.Sequential(*layers)


class AnimalSpeciesCNN(nn.Module):
    """Configurable VGG-style CNN for multi-class classification."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_filters: Sequence[int],
        hidden_dim: int,
        dropout: float,
        conv_layers_per_block: Sequence[int] | None = None,
        second_hidden_dim: int | None = None,
        kernel_size: int = 3,
        use_batch_norm: bool = True,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()

        filters = list(conv_filters)
        if not filters:
            raise ValueError("conv_filters must contain at least one entry.")

        if conv_layers_per_block is None:
            conv_layers_per_block = [2, 2, 3, 3, 3][: len(filters)]

        block_depths = list(conv_layers_per_block)
        if len(block_depths) != len(filters):
            raise ValueError(
                "conv_layers_per_block must have the same length as conv_filters."
            )

        blocks = []
        prev_channels = in_channels
        for out_channels, depth in zip(filters, block_depths):
            block = _make_conv_block(
                prev_channels,
                out_channels,
                depth,
                kernel_size,
                use_batch_norm,
                use_spectral_norm,
            )
            blocks.append(block)
            prev_channels = out_channels

        self.feature_extractor = nn.Sequential(*blocks)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        classifier_layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(filters[-1], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ]

        current_dim = hidden_dim
        if second_hidden_dim:
            classifier_layers.extend(
                [
                    nn.Linear(current_dim, second_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = second_hidden_dim

        classifier_layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

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
