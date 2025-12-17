#!/usr/bin/env python3

from __future__ import annotations

import torch
from torch import nn


class StringCNN(nn.Module):
    """
    Simple CNN for string classification from log-mel patches.

    Input:  [B, 1, n_mels, num_frames]
    Output: [B, num_strings] logits
    """

    def __init__(
        self,
        num_strings: int = 6,
        n_mels: int = 48,
        dropout_p: float = 0.3,
    ) -> None:
        super().__init__()

        # Downsample mostly along time, lightly along freq.
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # freq/2, time/2

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # freq/4, time/4

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # freq/8, time/8
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(64, num_strings)

        # stored for reproducibility / checkpoint metadata
        self.n_mels = int(n_mels)
        self.num_strings = int(num_strings)
        self.dropout_p = float(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)
