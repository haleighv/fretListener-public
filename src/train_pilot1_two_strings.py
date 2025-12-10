#!/usr/bin/env python

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# ============================
# CONFIG
# ============================

DATA_ROOT = Path("pilot1_two_strings")

# Use the balanced metadata produced by balance_two_string_metadata.py
META_CSV = DATA_ROOT / "metadata_balanced.csv"
FEAT_DIR = DATA_ROOT / "features"

BATCH_SIZE = 32
EPOCHS = 80
LR = 3e-4
WEIGHT_DECAY = 3e-4

NUM_STRINGS = 2
NUM_FRETS = 8  # frets 0..7


# ============================
# DATASET
# ============================

class TwoStringAttackDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feat_dir: Path):
        self.df = df.reset_index(drop=True)
        self.feat_dir = feat_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat_file = row["feat_file"]
        string_idx = int(row["string_idx"])
        fret = int(row["fret"])

        feat_path = self.feat_dir / feat_file
        x = np.load(feat_path)  # (n_mels, n_frames)

        x = torch.from_numpy(x).unsqueeze(0)  # (1, n_mels, n_frames)
        y_string = torch.tensor(string_idx, dtype=torch.long)
        y_fret = torch.tensor(fret, dtype=torch.long)
        return x, y_string, y_fret


# ============================
# MODEL
# ============================

class PositionCNN(nn.Module):
    def __init__(self, n_mels: int, n_frames: int,
                 num_strings: int, num_frets: int, dropout: float = 0.2):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.dropout = nn.Dropout(dropout)
        self.fc_shared = nn.Linear(64, 64)

        self.head_string = nn.Linear(64, num_strings)
        self.head_fret = nn.Linear(64, num_frets)

    def forward(self, x):
        h = self.backbone(x)        # (B, 64, 1, 1)
        h = h.view(x.size(0), -1)   # (B, 64)
        h = self.dropout(h)
        h = torch.relu(self.fc_shared(h))
        logits_string = self.head_string(h)
        logits_fret = self.head_fret(h)
        return logits_string, logits_fret


# ============================
# TRAIN / EVAL UTILS
# ============================

def split_dataframe(df: pd.DataFrame, train_frac=0.7, val_frac=0.15, seed=42):
    rng = np.random.RandomState(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)

    n = len(df)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    return df_train, df_val, df_test


def evaluate(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct_string = 0
    correct_fret = 0
    total = 0

    with torch.no_grad():
        for x, y_string, y_fret in loader:
            x = x.to(device)
            y_string = y_string.to(device)
            y_fret = y_fret.to(device)

            logits_string, logits_fret = model(x)
            loss_string = criterion(logits_string, y_string)
            loss_fret = criterion(logits_fret, y_fret)
            loss = loss_string + loss_fret

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

            preds_string = logits_string.argmax(dim=1)
            preds_fret = logits_fret.argmax(dim=1)

            correct_string += (preds_string == y_string).sum().item()
            correct_fret += (preds_fret == y_fret).sum().item()

    avg_loss = total_loss / total
    acc_string = correct_string / total
    acc_fret = correct_fret / total
    return avg_loss, acc_string, acc_fret


def compute_confusions(model, loader, device):
    model.eval()
    conf_str = np.zeros((NUM_STRINGS, NUM_STRINGS), dtype=int)
    conf_fret = np.zeros((NUM_FRETS, NUM_FRETS), dtype=int)

    with torch.no_grad():
        for x, y_string, y_fret in loader:
            x = x.to(device)
            y_string = y_string.to(device)
            y_fret = y_fret.to(device)

            logits_string, logits_fret = model(x)
            preds_string = logits_string.argmax(dim=1)
            preds_fret = logits_fret.argmax(dim=1)

            for t_s, p_s, t_f, p_f in zip(y_string, preds_string, y_fret, preds_fret):
                conf_str[int(t_s), int(p_s)] += 1
                conf_fret[int(t_f), int(p_f)] += 1

    return conf_str, conf_fret


# ============================
# MAIN
# ============================

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    if not META_CSV.exists():
        raise FileNotFoundError(f"Metadata file not found: {META_CSV}")

    df = pd.read_csv(META_CSV)
    print(f"Total samples (balanced): {len(df)}")
    print("Unique string_idx:", sorted(df["string_idx"].unique()))
    print("Unique frets:", sorted(df["fret"].unique()))

    first_feat = np.load(FEAT_DIR / df.iloc[0]["feat_file"])
    n_mels, n_frames = first_feat.shape
    print("Feature shape:", first_feat.shape)

    df_train, df_val, df_test = split_dataframe(df, train_frac=0.7, val_frac=0.15, seed=42)
    print(f"Train/Val/Test sizes: {len(df_train)}, {len(df_val)}, {len(df_test)}")

    train_ds = TwoStringAttackDataset(df_train, FEAT_DIR)
    val_ds = TwoStringAttackDataset(df_val, FEAT_DIR)
    test_ds = TwoStringAttackDataset(df_test, FEAT_DIR)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = PositionCNN(
        n_mels=n_mels,
        n_frames=n_frames,
        num_strings=NUM_STRINGS,
        num_frets=NUM_FRETS,
        dropout=0.2,  # tweaked
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct_string = 0
        correct_fret = 0

        for x, y_string, y_fret in train_loader:
            x = x.to(device)
            y_string = y_string.to(device)
            y_fret = y_fret.to(device)

            optimizer.zero_grad()
            logits_string, logits_fret = model(x)

            loss_string = criterion(logits_string, y_string)
            loss_fret = criterion(logits_fret, y_fret)
            loss = loss_string + loss_fret

            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total += bs

            preds_string = logits_string.argmax(dim=1)
            preds_fret = logits_fret.argmax(dim=1)
            correct_string += (preds_string == y_string).sum().item()
            correct_fret += (preds_fret == y_fret).sum().item()

        train_loss = total_loss / total
        train_acc_string = correct_string / total
        train_acc_fret = correct_fret / total

        val_loss, val_acc_string, val_acc_fret = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} "
            f"train_acc_str={train_acc_string:.3f} "
            f"train_acc_fret={train_acc_fret:.3f} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc_str={val_acc_string:.3f} "
            f"val_acc_fret={val_acc_fret:.3f}"
        )

    test_loss, test_acc_string, test_acc_fret = evaluate(model, test_loader, device)
    print(
        f"Test loss={test_loss:.4f} "
        f"acc_str={test_acc_string:.3f} "
        f"acc_fret={test_acc_fret:.3f}"
    )

    conf_str, conf_fret = compute_confusions(model, test_loader, device)

    print("\n=== STRING confusion (rows=true, cols=pred, 0..1) ===")
    for i in range(NUM_STRINGS):
        row = " ".join(f"{conf_str[i, j]:4d}" for j in range(NUM_STRINGS))
        print(f"true {i}: {row}")

    print("\n=== FRET confusion (rows=true, cols=pred, 0..7) ===")
    for i in range(NUM_FRETS):
        row = " ".join(f"{conf_fret[i, j]:4d}" for j in range(NUM_FRETS))
        print(f"true {i}: {row}")

    ckpt_path = DATA_ROOT / "pilot1_two_strings_position_cnn_balanced.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_mels": n_mels,
            "n_frames": n_frames,
            "num_strings": NUM_STRINGS,
            "num_frets": NUM_FRETS,
        },
        ckpt_path,
    )
    print(f"\nSaved model checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
