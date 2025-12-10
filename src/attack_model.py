import torch
import torch.nn as nn
import torch.nn.functional as F

class AttackPositionNet(nn.Module):
    def __init__(self, num_classes: int = 48):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_embed = nn.Linear(128, 128)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x, return_embedding: bool = False):
        z = self.features(x)
        z = self.global_pool(z)
        z = z.view(z.size(0), -1)

        embedding = F.relu(self.fc_embed(z), inplace=True)
        logits = self.fc_out(embedding)

        if return_embedding:
            return logits, embedding
        return logits
