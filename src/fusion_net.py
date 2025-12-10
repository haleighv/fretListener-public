import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(129, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 48)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
