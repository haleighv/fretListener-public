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

DATA_ROOT = Path("pilot2_all_strings")
META_CSV = DATA_ROOT / "metadata_full.csv"
FEAT_DIR = DATA_ROOT / "features"

BATCH_SIZE = 64
EPOCHS = 80
LR = 3e-4
WEIGHT_DECAY = 3e-4

NUM_STRINGS = 6   # strings 1..6 → idx 0..5
NUM_FRETS = 8     # frets 0..7
NUM_POS = NUM_STRINGS * NUM_FRETS  # 48 joint (string, fret) classes

# Attack window: longer than before
MAX_ATTACK_FRAMES = 40


# ============================
# DATASET WITH AUGMENTATION
# ============================

class AttackPosition48Dataset(Dataset):
    """
    Each sample:
      - Input: log-mel patch cropped to first MAX_ATTACK_FRAMES frames.
      - Optional augmentation on log-mel.
      - Label: position_idx ∈ [0, 47] encoding (string, fret).
    """

    def __init__(self, df: pd.DataFrame, feat_dir: Path, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.feat_dir = feat_dir
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        feat_file = row["feat_file"]
        string_idx = int(row["string_idx"])  # 0..5
        fret = int(row["fret"])              # 0..7

        feat_path = self.feat_dir / feat_file
        x = np.load(feat_path)  # (n_mels, n_frames)

        # Attack-only cropping
        if x.shape[1] > MAX_ATTACK_FRAMES:
            x = x[:, :MAX_ATTACK_FRAMES]

        # Simple log-mel augmentations
        if self.augment:
            x = self._augment_logmel(x)

        # Per-sample normalization
        x = (x - x.mean()) / (x.std() + 1e-6)

        x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)  # (1, M, T)

        pos_idx = string_idx * NUM_FRETS + fret
        y_pos = torch.tensor(pos_idx, dtype=torch.long)

        return x, y_pos

    def _augment_logmel(self, x: np.ndarray) -> np.ndarray:
        """
        x: (n_mels, n_frames)
        Augmentations:
          - Small gain jitter in dB
          - Small additive Gaussian noise
          - Tiny time shift (±2 frames)
        """
        x_aug = x.copy()

        # Gain jitter ±3 dB
        if np.random.rand() < 0.8:
            gain_db = np.random.uniform(-3.0, 3.0)
            x_aug = x_aug + gain_db

        # Additive noise
        if np.random.rand() < 0.8:
            noise_std = np.random.uniform(0.1, 0.3)
            noise = np.random.randn(*x_aug.shape) * noise_std
            x_aug = x_aug + noise

        # Tiny time shift (circular)
        if np.random.rand() < 0.5 and x_aug.shape[1] > 4:
            shift = np.random.randint(-2, 3)
            if shift != 0:
                x_aug = np.roll(x_aug, shift=shift, axis=1)

        return x_aug


# ============================
# MODEL
# ============================

class Position48CNN(nn.Module):
    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, 128)
        self.head_pos = nn.Linear(128, NUM_POS)

    def forward(self, x):
        h = self.backbone(x)
        h = h.view(x.size(0), -1)
        h = self.dropout(h)
        h = torch.relu(self.fc(h))
        logits_pos = self.head_pos(h)
        return logits_pos


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


def decode_pos_idx(pos_idx_tensor: torch.Tensor):
    string_idx = pos_idx_tensor // NUM_FRETS
    fret = pos_idx_tensor % NUM_FRETS
    return string_idx, fret


def evaluate(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    total = 0
    correct_pos = 0
    correct_string = 0
    correct_fret = 0

    with torch.no_grad():
        for x, y_pos in loader:
            x = x.to(device)
            y_pos = y_pos.to(device)

            logits_pos = model(x)
            loss = criterion(logits_pos, y_pos)

            bs = x.size(0)
            total_loss += loss.item() * bs
            total += bs

            preds_pos = logits_pos.argmax(dim=1)
            correct_pos += (preds_pos == y_pos).sum().item()

            true_str, true_fret = decode_pos_idx(y_pos)
            pred_str, pred_fret = decode_pos_idx(preds_pos)
            correct_string += (pred_str == true_str).sum().item()
            correct_fret += (pred_fret == true_fret).sum().item()

    avg_loss = total_loss / total
    acc_pos = correct_pos / total
    acc_str = correct_string / total
    acc_fret = correct_fret / total

    return avg_loss, acc_pos, acc_str, acc_fret


def compute_confusions(model, loader, device):
    model.eval()
    conf_str = np.zeros((NUM_STRINGS, NUM_STRINGS), dtype=int)
    conf_fret = np.zeros((NUM_FRETS, NUM_FRETS), dtype=int)

    with torch.no_grad():
        for x, y_pos in loader:
            x = x.to(device)
            y_pos = y_pos.to(device)

            logits_pos = model(x)
            preds_pos = logits_pos.argmax(dim=1)

            true_str, true_fret = decode_pos_idx(y_pos)
            pred_str, pred_fret = decode_pos_idx(preds_pos)

            for ts, ps, tf, pf in zip(true_str, pred_str, true_fret, pred_fret):
                conf_str[int(ts), int(ps)] += 1
                conf_fret[int(tf), int(pf)] += 1

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
    print(f"Total samples: {len(df)}")
    print("Unique string_idx:", sorted(df["string_idx"].unique()))
    print("Unique frets:", sorted(df["fret"].unique()))

    first_feat = np.load(FEAT_DIR / df.iloc[0]["feat_file"])
    print("Original feature shape:", first_feat.shape)

    df_train, df_val, df_test = split_dataframe(df, train_frac=0.7, val_frac=0.15, seed=42)
    print(f"Train/Val/Test sizes: {len(df_train)}, {len(df_val)}, {len(df_test)}")

    train_ds = AttackPosition48Dataset(df_train, FEAT_DIR, augment=True)
    val_ds = AttackPosition48Dataset(df_val, FEAT_DIR, augment=False)
    test_ds = AttackPosition48Dataset(df_test, FEAT_DIR, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = Position48CNN(dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct_pos = 0
        correct_str = 0
        correct_fret = 0

        for x, y_pos in train_loader:
            x = x.to(device)
            y_pos = y_pos.to(device)

            optimizer.zero_grad()
            logits_pos = model(x)
            loss = criterion(logits_pos, y_pos)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total += bs

            preds_pos = logits_pos.argmax(dim=1)
            correct_pos += (preds_pos == y_pos).sum().item()

            true_str, true_fret = decode_pos_idx(y_pos)
            pred_str, pred_fret = decode_pos_idx(preds_pos)
            correct_str += (pred_str == true_str).sum().item()
            correct_fret += (pred_fret == true_fret).sum().item()

        train_loss = total_loss / total
        train_acc_pos = correct_pos / total
        train_acc_str = correct_str / total
        train_acc_fret = correct_fret / total

        val_loss, val_acc_pos, val_acc_str, val_acc_fret = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} "
            f"train_acc_pos={train_acc_pos:.3f} "
            f"train_acc_str={train_acc_str:.3f} "
            f"train_acc_fret={train_acc_fret:.3f} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc_pos={val_acc_pos:.3f} "
            f"val_acc_str={val_acc_str:.3f} "
            f"val_acc_fret={val_acc_fret:.3f}"
        )

    test_loss, test_acc_pos, test_acc_str, test_acc_fret = evaluate(model, test_loader, device)
    print(
        f"\nTest loss={test_loss:.4f} "
        f"acc_pos={test_acc_pos:.3f} "
        f"acc_str={test_acc_str:.3f} "
        f"acc_fret={test_acc_fret:.3f}"
    )

    conf_str, conf_fret = compute_confusions(model, test_loader, device)

    print("\n=== STRING confusion (rows=true, cols=pred, 0..5) ===")
    for i in range(NUM_STRINGS):
        row = " ".join(f"{conf_str[i, j]:4d}" for j in range(NUM_STRINGS))
        print(f"true {i}: {row}")

    print("\n=== FRET confusion (rows=true, cols=pred, 0..7) ===")
    for i in range(NUM_FRETS):
        row = " ".join(f"{conf_fret[i, j]:4d}" for j in range(NUM_FRETS))
        print(f"true {i}: {row}")

    ckpt_path = DATA_ROOT / "pilot2_attack_position_48class_aug.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_strings": NUM_STRINGS,
            "num_frets": NUM_FRETS,
            "num_pos": NUM_POS,
            "max_attack_frames": MAX_ATTACK_FRAMES,
        },
        ckpt_path,
    )
    print(f"\nSaved model checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
