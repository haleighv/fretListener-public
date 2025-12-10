import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from fusion_net import FusionNet

class FusionDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = np.stack(df["embedding"].apply(lambda s: np.fromstring(s, sep=" ")))
        self.pitch = df["pitch_midi"].values[:, None]
        self.y = df["pos"].values
        self.X = np.concatenate([self.X, self.pitch], axis=1)

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.float32), torch.tensor(self.y[i])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--checkpoint", default="fusion_net.pt")
    args = parser.parse_args()

    ds = FusionDataset(args.csv)
    dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)

    device = torch.device("mps")
    model = FusionNet().to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(30):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        print("Epoch", epoch)

    torch.save({"model_state": model.state_dict()}, args.checkpoint)

if __name__ == "__main__":
    main()
