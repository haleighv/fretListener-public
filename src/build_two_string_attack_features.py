#!/usr/bin/env python

import numpy as np
import pandas as pd
import librosa
from pathlib import Path

# ============================
# CONFIG
# ============================

# Root where daw_dataset_builder.py wrote the clips
PROCESSED_ROOT = Path("processed_dataset")

# Only use strings 1 and 2 for this sanity check
ALLOWED_STRINGS = {1, 2}

# Output
OUT_ROOT = Path("pilot1_two_strings")
FEAT_DIR = OUT_ROOT / "features"
META_CSV = OUT_ROOT / "metadata.csv"

# Audio / feature settings
TARGET_SR = 16_000

# Attack window (extended from 80 ms -> 120 ms)
ATTACK_MS = 120.0
ATTACK_SAMPLES = int(TARGET_SR * (ATTACK_MS / 1000.0))

N_MELS = 64
N_FFT = 512
HOP_LENGTH = 128
POWER = 2.0
EPS = 1e-10


# ============================
# MAIN
# ============================

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    # Expect subdirs like "s1_f0", "s2_f7", etc.
    for pos_dir in sorted(PROCESSED_ROOT.iterdir()):
        if not pos_dir.is_dir():
            continue

        base = pos_dir.name  # e.g. "s1_f0"
        parts = base.split("_")
        if len(parts) != 2:
            continue
        s_part, f_part = parts
        if not (s_part.startswith("s") and f_part.startswith("f")):
            continue

        string_num = int(s_part[1:])
        fret_num = int(f_part[1:])

        # Skip anything not s1 or s2
        if string_num not in ALLOWED_STRINGS:
            continue

        print(f"Processing position {base} (string={string_num}, fret={fret_num})")

        # Turn string 1 → 0, string 2 → 1 for labels
        string_idx = string_num - 1

        for wav_path in sorted(pos_dir.glob("*.wav")):
            y, sr = librosa.load(wav_path, sr=TARGET_SR)

            # Fixed-length attack window
            if len(y) < ATTACK_SAMPLES:
                pad = ATTACK_SAMPLES - len(y)
                y = np.pad(y, (0, pad), mode="constant")
            else:
                y = y[:ATTACK_SAMPLES]

            mel = librosa.feature.melspectrogram(
                y=y,
                sr=TARGET_SR,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS,
                power=POWER,
            )
            logmel = np.log(mel + EPS).astype(np.float32)

            feat_name = wav_path.stem + ".npy"
            feat_path = FEAT_DIR / feat_name
            np.save(feat_path, logmel)

            rows.append({
                "feat_file": feat_name,
                "string_idx": string_idx,  # 0 or 1
                "fret": fret_num,          # 0..7
                "string_num": string_num,
                "pos_dir": base,
            })

    df = pd.DataFrame(rows)
    df.to_csv(META_CSV, index=False)
    print(f"Wrote {len(df)} rows to {META_CSV}")
    print(f"Features saved in {FEAT_DIR}")


if __name__ == "__main__":
    main()
