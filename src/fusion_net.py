import torch
import torch.nn as nn

class FusionNet(nn.Module):
    def __init__(self, emb_dim=128, num_strings=6, num_frets=8):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(emb_dim + 1, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, num_strings * num_frets)
        )

    def forward(self, emb, pitch_midi):
        pitch_norm = pitch_midi.float().unsqueeze(1) / 127.0
        x = torch.cat([emb, pitch_norm], dim=1)
        return self.net(x)
