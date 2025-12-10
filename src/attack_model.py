import torch
import torch.nn as nn

class AttackPositionNet(nn.Module):
    def __init__(self, n_mels=64, num_pos=48):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(128, 128)
        self.head_pos = nn.Linear(128, num_pos)

    def forward(self, x, return_embedding=False):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        emb = torch.relu(self.fc(x))

        if return_embedding:
            return emb

        return self.head_pos(emb)
