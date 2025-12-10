#!/usr/bin/env python

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoteCNN(nn.Module):
    """
    CNN for note+octave classification from log-mel spectrograms.

    Expected input: [B, 1, n_mels, T] (e.g. [B, 1, 64, 32])
    Output: [B, num_classes] logits over MIDI-note classes.
    """

    def __init__(self, num_classes: int, n_mels: int = 64):
        super().__init__()

        # [B, 1, n_mels, T]
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Global average pooling over (freq, time)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, n_mels, T]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2)  # downsample

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.bn3(self.conv3(x)))

        # Global average pooling over spatial dims
        x = x.mean(dim=[2, 3])  # [B, 128]

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward_with_embedding(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optional helper if embeddings are ever desired.
        Returns:
          logits: [B, num_classes]
          embedding: [B, 128]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.bn3(self.conv3(x)))

        x = x.mean(dim=[2, 3])  # [B, 128]
        embedding = F.relu(self.fc1(x))
        logits = self.fc2(embedding)
        return logits, embedding
