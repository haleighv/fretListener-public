# src/multihead_cnn.py

from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class GuitarMultiHeadCNN(nn.Module):
    """
    CNN encoder over log-mel spectrograms with three classification heads:
      - note head: 12 pitch classes (C..B)
      - string head: 6 string classes (1..6)
      - fret head: 6 fret classes (0..5)

    Input shape:
        [B, 1, n_mels, T]  (e.g. [B, 1, 64, ~31])
    """

    def __init__(
        self,
        n_mels: int = 64,
        num_notes: int = 12,
        num_strings: int = 6,
        num_frets: int = 6,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # Convolutional encoder
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # halves both freq and time
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # global pooling → [B, 128, 1, 1]
        )

        # Shared fully-connected layer
        self.fc_shared = nn.Linear(128, hidden_dim)

        # Heads
        self.fc_note = nn.Linear(hidden_dim, num_notes)
        self.fc_string = nn.Linear(hidden_dim, num_strings)
        self.fc_fret = nn.Linear(hidden_dim, num_frets)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, 1, n_mels, T]

        Returns dict with logits for each head:
            {
                "note":   [B, num_notes],
                "string": [B, num_strings],
                "fret":   [B, num_frets],
            }
        """
        # CNN encoder
        h = self.conv_block1(x)
        h = self.conv_block2(h)
        h = self.conv_block3(h)  # [B, 128, 1, 1]

        h = h.view(h.size(0), -1)  # [B, 128]
        h = self.fc_shared(h)
        h = F.relu(h, inplace=True)

        logits_note = self.fc_note(h)
        logits_string = self.fc_string(h)
        logits_fret = self.fc_fret(h)

        return {
            "note": logits_note,
            "string": logits_string,
            "fret": logits_fret,
        }
