# src/fusion_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionFretNet(nn.Module):
    """
    Fusion model:
      - Waveform branch: 1D CNN on raw audio
      - Spectrogram branch: 2D CNN on log-mel spectrogram
      - Concatenate embeddings and classify fret/string class
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # Waveform branch: input [B, 1, T]
        self.wave_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.wave_pool = nn.AdaptiveAvgPool1d(1)  # -> [B, 128, 1]

        # Spectrogram branch: input [B, 1, n_mels, time]
        self.spec_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # Global average pool to [B, 128, 1, 1] to avoid MPS size issues
        self.spec_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final classifier
        wave_feat_dim = 128           # from wave_pool
        spec_feat_dim = 128           # from spec_pool -> [B,128,1,1] -> 128
        fusion_dim = wave_feat_dim + spec_feat_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, wave: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        """
        wave: [B, 1, T]
        spec: [B, 1, n_mels, time]
        """
        # Waveform path
        w = self.wave_branch(wave)          # [B, 128, T']
        w = self.wave_pool(w)               # [B, 128, 1]
        w = w.squeeze(-1)                   # [B, 128]

        # Spectrogram path
        s = self.spec_branch(spec)          # [B, 128, H, W]
        s = self.spec_pool(s)               # [B, 128, 1, 1]
        s = s.view(s.size(0), -1)           # [B, 128]

        fused = torch.cat([w, s], dim=1)    # [B, 256]
        logits = self.classifier(fused)
        return logits
