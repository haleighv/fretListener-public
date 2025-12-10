#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path

# =========================================================
# PATHS (MATCH YOUR ENV)
# =========================================================

ROOT = Path(__file__).resolve().parent.parent
FUSION_MODEL_PATH = Path("src/pilot_1_fusion.pt")
FUSION_DATASET_PATH = ROOT / "data" / "pilot_1" / "fusion_dataset.csv"

# =========================================================
# MODEL DEFINITION (MUST MATCH TRAINING)
# =========================================================

class FusionNet(nn.Module):
    def __init__(self, attack_dim, pitch_dim=1, hidden=128, num_pos=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(attack_dim + pitch_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_pos),
        )

    def forward(self, x):
        return self.net(x)

# =========================================================
# POSITION DECODING
# =========================================================

POSITION_LOOKUP = [(s, f) for s in range(6) for f in range(8)]

def decode_position(idx):
    s, f = POSITION_LOOKUP[idx]
    return s, f

# =========================================================
# MAIN
# =========================================================

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # ---------------- LOAD DATA ----------------

    df = pd.read_csv(FUSION_DATASET_PATH)
    print("Loaded fusion dataset:", len(df), "samples")
    print("Fusion columns:", list(df.columns))

    # ---------------- DECODE EMBEDDINGS ----------------

    if "embedding" not in df.columns:
        raise RuntimeError("Missing 'embedding' column in fusion dataset")

    # Embeddings are stored as stringified python lists
    X_attack = np.stack(df["embedding"].apply(eval).values).astype(np.float32)

    # ---------------- PITCH ----------------

    if "pitch_midi" not in df.columns:
        raise RuntimeError("Missing 'pitch_midi' column in fusion dataset")

    X_pitch = df["pitch_midi"].to_numpy().astype(np.float32)[:, None]

    # ---------------- CONCAT FEATURES ----------------

    X = np.concatenate([X_attack, X_pitch], axis=1)

    y_pos = df["pos"].to_numpy().astype(int)
    y_string = df["string"].to_numpy().astype(int)
    y_fret = df["fret"].to_numpy().astype(int)

    X = torch.from_numpy(X).float().to(device)
    y_pos = torch.from_numpy(y_pos).long().to(device)

    # ---------------- LOAD MODEL ----------------

    checkpoint = torch.load(FUSION_MODEL_PATH, map_location=device)
    attack_dim = checkpoint["attack_dim"]

    model = FusionNet(attack_dim=attack_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ---------------- INFERENCE ----------------

    with torch.no_grad():
        logits = model(X)
        preds = torch.argmax(logits, dim=1)

    pred_pos = preds.cpu().numpy()

    pred_string = np.array([decode_position(i)[0] for i in pred_pos])
    pred_fret = np.array([decode_position(i)[1] for i in pred_pos])

    acc_pos = (pred_pos == y_pos.cpu().numpy()).mean()
    acc_str = (pred_string == y_string).mean()
    acc_fret = (pred_fret == y_fret).mean()

    print("\n========= FUSION RESULTS =========")
    print(f"Position Accuracy : {acc_pos:.3f}")
    print(f"String Accuracy   : {acc_str:.3f}")
    print(f"Fret Accuracy     : {acc_fret:.3f}")

    # ---------------- CONFUSIONS ----------------

    print("\n=== STRING CONFUSION (rows=true, cols=pred) ===")
    str_conf = np.zeros((6, 6), dtype=int)
    for t, p in zip(y_string, pred_string):
        str_conf[t, p] += 1
    for i in range(6):
        print(f"true {i}:", str_conf[i])

    print("\n=== FRET CONFUSION (rows=true, cols=pred) ===")
    fret_conf = np.zeros((8, 8), dtype=int)
    for t, p in zip(y_fret, pred_fret):
        fret_conf[t, p] += 1
    for i in range(8):
        print(f"true {i}:", fret_conf[i])

    # ---------------- COLLAPSE CHECK ----------------

    unique_preds = np.unique(pred_pos)
    print("\nUnique predicted positions:", len(unique_preds), "/ 48")

    if len(unique_preds) < 10:
        print("⚠️  MODEL COLLAPSE DETECTED")
    else:
        print("✅ Prediction distribution healthy")

if __name__ == "__main__":
    main()
