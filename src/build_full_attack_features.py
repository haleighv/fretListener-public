#!/usr/bin/env python

import numpy as np
import pandas as pd
import librosa
from pathlib import Path

# ============================
# CONFIG
# ============================

# Root produced by daw_dataset_builder.py
PROCESSED_ROOT = Path("processed_dataset")

# Use all 6 strings
ALLOWED_STRINGS = {1, 2, 3, 4, 5, 6}

# Output
OUT_ROOT = Path("pilot2_all_strings")
FEAT_DIR = OUT_ROOT / "features"
META_CSV = OUT_ROOT / "metadata_full.csv"

# Audio / feature settings
TARGET_SR = 16_000

# Attack window (same as 2-string sanity check)
ATTACK_MS = 120.0
ATTACK_SAMPLES = int(TARGET_SR * (ATTACK_MS / 1000.0))

N_MELS = 64
N_FFT = 512
HOP_LENGTH = 128
POWER = 2.0
EPS = 1e-10

# Standard tuning, string 1 = high E
# Values are MIDI numbers for open strings.
TUNING_MIDI = {
    1: 64,  # E4
    2: 59,  # B3
    3: 55,  # G3
    4: 50,  # D3
    5: 45,  # A2
    6: 40,  # E2
}

# For this dataset (strings 1–6, frets 0–7), MIDI range is 40..71
MIDI_MIN = 40


# ============================
# HELPERS
# ============================

NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F",
                    "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(m):
    """Convert MIDI integer to note name with octave, e.g. 60 -> 'C4'."""
    pitch_class = m % 12
    octave = (m // 12) - 1
    return f"{NOTE_NAMES_SHARP[pitch_class]}{octave}"


# ============================
# MAIN
# ============================

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    # Expect subdirs like "s1_f0", "s2_f7", ..., "s6_f7"
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

        if string_num not in ALLOWED_STRINGS:
            continue

        # Derive MIDI + note name
        if string_num not in TUNING_MIDI:
            raise ValueError(f"Unknown string_num: {string_num}")

        midi = TUNING_MIDI[string_num] + fret_num
        note_name = midi_to_name(midi)

        # String index for model: 0..5
        string_idx = string_num - 1

        # Note index for model: 0..(N-1); since this dataset covers 40..71,
        # use midi - MIDI_MIN so range is contiguous.
        note_idx = midi - MIDI_MIN

        print(
            f"Processing position {base} "
            f"(string={string_num}, fret={fret_num}, midi={midi} {note_name})"
        )

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

            rows.append(
                {
                    "feat_file": feat_name,
                    "string_idx": string_idx,  # 0..5
                    "string_num": string_num,  # 1..6
                    "fret": fret_num,          # 0..7
                    "midi": midi,              # 40..71
                    "note_idx": note_idx,      # 0..31
                    "note_name": note_name,    # e.g. 'G#3'
                    "pos_dir": base,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(META_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {META_CSV}")
    print(f"Features saved in {FEAT_DIR}")

    # Quick sanity
    print("\nUnique strings:", sorted(df['string_idx'].unique()))
    print("Unique frets:", sorted(df['fret'].unique()))
    print("MIDI range:", df['midi'].min(), "to", df['midi'].max())
    print("Unique note_idx:", sorted(df['note_idx'].unique()))


if __name__ == "__main__":
    main()
