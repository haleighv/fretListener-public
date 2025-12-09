# src/make_features.py

import csv
from pathlib import Path

import librosa  # pip install librosa
import numpy as np
import torch

from labels import NOTE_TO_INDEX

ROOT = Path(__file__).resolve().parents[1]
META_DIR = ROOT / "data" / "pilot_0" / "guitar_dataset" / "meta"
FEATURE_META = META_DIR / "pilot_0_features_metadata.csv"

FEATURES_DIR = ROOT / "data" / "pilot_0" / "guitar_dataset" / "features"

# Audio / feature hyperparameters
TARGET_SR = 16_000
TARGET_DURATION_SEC = 0.5
N_FFT = 512
HOP_LENGTH = 128
N_MELS = 64
EPS = 1e-10


def load_feature_metadata():
    if not FEATURE_META.exists():
        raise FileNotFoundError(f"Feature metadata not found: {FEATURE_META}")
    with FEATURE_META.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Type-cast
    for r in rows:
        r["string"] = int(r["string"])
        r["fret"] = int(r["fret"])
        r["octave"] = int(r["octave"])
        r["take"] = int(r["take"])
        r["midi"] = int(r["midi"])
        r["is_ambiguous"] = r["is_ambiguous"].lower() == "true"
    return rows


def standardize_audio(abs_path: Path):
    """
    Load audio, convert to mono, resample to TARGET_SR, peak-normalize,
    trim silence, and pad/crop to fixed length.
    """
    y, sr = librosa.load(str(abs_path), sr=None, mono=True)  # original sr
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    # Trim leading/trailing silence (40 dB threshold)
    y, _ = librosa.effects.trim(y, top_db=40)

    # Peak normalize
    peak = np.max(np.abs(y)) if y.size > 0 else 0.0
    if peak > 0:
        y = y / peak

    # Pad or crop to TARGET_DURATION_SEC
    target_len = int(TARGET_SR * TARGET_DURATION_SEC)
    if y.size < target_len:
        pad_width = target_len - y.size
        y = np.pad(y, (0, pad_width), mode="constant")
    elif y.size > target_len:
        y = y[:target_len]

    return y.astype(np.float32), sr


def audio_to_logmel(y: np.ndarray, sr: int):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        center=True,
        power=2.0,
    )
    logmel = np.log(mel + EPS)
    return logmel.astype(np.float32)  # shape (n_mels, T)



def process_all():
    rows = load_feature_metadata()
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    for idx, r in enumerate(rows):
        rel_path = Path(r["filepath"])
        abs_path = ROOT / rel_path
        if not abs_path.exists():
            raise FileNotFoundError(f"Missing audio file: {abs_path}")

        y, sr = standardize_audio(abs_path)
        logmel = audio_to_logmel(y, sr)

        # Tensor shape: (1, n_mels, T)
        feat = torch.from_numpy(logmel).unsqueeze(0)

        note_name = r["note"]
        if note_name not in NOTE_TO_INDEX:
            raise ValueError(f"Unknown note name: {note_name}")
        note_id = NOTE_TO_INDEX[note_name]

        out = {
            "features": feat,  # Tensor[1, N_MELS, T]
            "string": r["string"],
            "fret": r["fret"],
            "note": note_name,
            "note_id": note_id,
            "octave": r["octave"],
            "midi": r["midi"],
            "is_ambiguous": r["is_ambiguous"],
            "take": r["take"],
            "session": r["session"],
            "guitar_model": r["guitar_model"],
            "pickup_type": r["pickup_type"],
            "pickup_selected": r["pickup_selected"],
            "timestamp": r["timestamp"],
        }

        out_name = Path(r["filename"]).with_suffix(".pt").name
        out_path = FEATURES_DIR / out_name
        torch.save(out, out_path)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(rows)}")

    print(f"Done. Saved {len(rows)} feature tensors to {FEATURES_DIR}")


if __name__ == "__main__":
    process_all()
