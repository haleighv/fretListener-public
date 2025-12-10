#!/usr/bin/env python3

import csv
from pathlib import Path
import numpy as np
import torch
import librosa

from attack_model import AttackPositionNet
from pitch_yin import extract_pitch_yin

# =========================================================
# PATHS (REAL)
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = ROOT / "data" / "pilot_1" / "processed_dataset"
ATTACK_MODEL_PATH = ROOT / "src" / "pilot2_all_strings" / "pilot2_attack_position_48class.pt"
OUT_CSV = ROOT / "data" / "pilot_1" / "fusion_dataset.csv"

# =========================================================
# AUDIO + FEATURE PARAMS (MATCH TRAINING)
# =========================================================

SAMPLE_RATE = 48000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
FMIN = 55.0
FMAX = 2000.0

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# =========================================================
# LOAD ATTACK MODEL
# =========================================================

ckpt = torch.load(ATTACK_MODEL_PATH, map_location=DEVICE)
ATTACK_FRAMES = int(ckpt["max_attack_frames"])

attack_model = AttackPositionNet(num_pos=48).to(DEVICE)
attack_model.load_state_dict(ckpt["model_state_dict"])
attack_model.eval()

print(f"[info] Using ATTACK_FRAMES = {ATTACK_FRAMES}")

# =========================================================
# ATTACK LOGMEL EXTRACTION (LOCAL IMPLEMENTATION)
# =========================================================

def waveform_to_logmel_attack(y, attack_frames):
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )

    logmel = librosa.power_to_db(mel, ref=np.max)

    T = logmel.shape[1]
    if T < attack_frames:
        logmel = np.pad(logmel, ((0, 0), (0, attack_frames - T)))
    elif T > attack_frames:
        logmel = logmel[:, :attack_frames]

    return logmel.astype(np.float32)

# =========================================================
# POSITION LOOKUP (0–5 STRINGS, 0–7 FRETS)
# =========================================================

POSITION_LOOKUP = [(s, f) for s in range(6) for f in range(8)]

# =========================================================
# BUILD DATASET
# =========================================================

rows = []

for pos_idx, (s, f) in enumerate(POSITION_LOOKUP):
    folder = DATASET_ROOT / f"s{s+1}_f{f}"

    if not folder.exists():
        print(f"[skip] missing {folder}")
        continue

    for wav in sorted(folder.glob("*.wav")):
        y, _ = librosa.load(wav, sr=SAMPLE_RATE)

        # --- ATTACK FEATURES ---
        logmel = waveform_to_logmel_attack(y, ATTACK_FRAMES)
        x = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            embedding = attack_model(x, return_embedding=True)[0].cpu().numpy()

        # --- PITCH FROM SUSTAIN ---
        _, midi = extract_pitch_yin(y)

        rows.append([
            embedding.tolist(),
            int(midi),
            int(s),
            int(f),
            int(pos_idx)
        ])

        print(f"[ok] {wav.name}")

# =========================================================
# SAVE CSV
# =========================================================

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["embedding", "pitch_midi", "string", "fret", "pos"])
    writer.writerows(rows)

print(f"\n✅ Wrote fusion dataset → {OUT_CSV}")
