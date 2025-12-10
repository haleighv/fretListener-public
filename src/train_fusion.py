#!/usr/bin/env python3

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from fusion_net import FusionNet

# =====================================================
# REAL PATHS (MATCHES YOUR TREE)
# =====================================================

ROOT = Path(__file__).resolve().parent.parent
FUSION_CSV = ROOT / "data" / "pilot_1" / "fusion_dataset.csv"
OUT_MODEL = ROOT / "src" / "pilot_1_fusion.pt"


# =====================================================
# DATASET
# =====================================================

class FusionDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        # embedding is stored as a string repr of a list → parse safely
        self.X_embed = np.stack(df["embedding"].apply(ast.literal_eval).values)
        self.X_pitch = df["pitch_midi"].values.reshape(-1, 1).astype(np.float32)
        self.y = df["pos"].values.astype(np.int64)

        # final feature vector = [embedding ... , pitch_midi]
        self.X = np.hstack([self.X_embed, self.X_pitch])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


# =====================================================
# TRAINING
# =====================================================

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # ---------- Load CSV ----------
    if not FUSION_CSV.exists():
        raise FileNotFoundError(f"Fusion CSV not found at: {FUSION_CSV}")

    df = pd.read_csv(FUSION_CSV)
    print("Loaded fusion dataset:", len(df))
    print("Columns:", list(df.columns))

    # train/test split
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["pos"], random_state=42
    )

    train_ds = FusionDataset(train_df)
    test_ds = FusionDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128)

    input_dim = train_ds.X.shape[1]

    # ---------- Model ----------
    model = FusionNet(input_dim=input_dim, num_classes=48).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # =================================================
    # EPOCH LOOP
    # =================================================
    for epoch in range(30):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # ---------- Eval ----------
        model.eval()
        all_preds = []
        all_true = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(y.numpy())

        acc = accuracy_score(all_true, all_preds)
        print(f"Epoch {epoch+1:02d} | loss={total_loss:.4f} | acc={acc:.3f}")

    # =================================================
    # SAVE CHECKPOINT
    # =================================================
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
        },
        OUT_MODEL,
    )

    print(f"\n✅ Saved fusion model → {OUT_MODEL}")


if __name__ == "__main__":
    main()
