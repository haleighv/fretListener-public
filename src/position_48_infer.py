#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import librosa
from pathlib import Path

# ============================
# CONFIG (must match training)
# ============================

SR = 44100
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 64
FMIN = 65.0
FMAX = 2000.0

NUM_STRINGS = 6
NUM_FRETS = 8
NUM_POS = NUM_STRINGS * NUM_FRETS

MAX_ATTACK_FRAMES = 20  # use same as in train_attack_position_48class.py

# Standard tuning: string_idx 0..5 = E4, B3, G3, D3, A2, E2
OPEN_STRING_MIDI = [64, 59, 55, 50, 45, 40]


# ============================
# MODEL (same as training)
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
        h = self.backbone(x)          # (B, 128, 1, 1)
        h = h.view(x.size(0), -1)     # (B, 128)
        h = self.dropout(h)
        h = torch.relu(self.fc(h))
        logits_pos = self.head_pos(h)
        return logits_pos


# ============================
# AUDIO → LOGMEL
# ============================

def waveform_to_logmel_full(y: np.ndarray, sr: int = SR) -> np.ndarray:
    if y.ndim > 1:
        y = librosa.to_mono(y)

    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        sr = SR

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        center=True,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    return logmel.astype(np.float32)


def logmel_to_attack_patch(logmel: np.ndarray, max_frames: int = MAX_ATTACK_FRAMES) -> np.ndarray:
    # logmel: (n_mels, n_frames)
    x = logmel
    if x.shape[1] > max_frames:
        x = x[:, :max_frames]
    x = (x - x.mean()) / (x.std() + 1e-6)
    return x.astype(np.float32)


# ============================
# MIDI / NOTE HELPERS
# ============================

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]


def string_fret_to_midi(string_idx: int, fret: int) -> int:
    """
    string_idx: 0..5 (1 = high E → idx 0; 6 = low E → idx 5)
    fret: 0..N
    """
    return OPEN_STRING_MIDI[string_idx] + fret


def midi_to_note_name(midi: int):
    name = NOTE_NAMES[midi % 12]
    octave = (midi // 12) - 1
    return name, octave


def decode_pos_idx(pos_idx: int):
    string_idx = pos_idx // NUM_FRETS
    fret = pos_idx % NUM_FRETS
    return string_idx, fret


# ============================
# LOADING / INFERENCE
# ============================

def load_position_model(ckpt_path: str | Path, device: torch.device | None = None):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    ckpt_path = Path(ckpt_path)
    state = torch.load(ckpt_path, map_location=device)

    model = Position48CNN(dropout=0.2)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    return model, device


def predict_position_from_waveform(
    model: Position48CNN,
    device: torch.device,
    y: np.ndarray,
    sr: int = SR,
):
    """
    Returns:
      dict with string_idx, fret, midi, note_name, note_octave,
      and posterior over the 48 positions.
    """
    logmel_full = waveform_to_logmel_full(y, sr=sr)
    x_attack = logmel_to_attack_patch(logmel_full, MAX_ATTACK_FRAMES)

    x_tensor = torch.from_numpy(x_attack).unsqueeze(0).unsqueeze(0)  # (1,1,M,T)
    x_tensor = x_tensor.to(device)

    with torch.no_grad():
        logits_pos = model(x_tensor)
        probs = torch.softmax(logits_pos, dim=1)[0]  # (48,)

    pos_idx = int(torch.argmax(probs).item())
    pos_prob = float(probs[pos_idx].item())

    string_idx, fret = decode_pos_idx(pos_idx)
    midi = string_fret_to_midi(string_idx, fret)
    note_name, note_octave = midi_to_note_name(midi)

    return {
        "pos_idx": pos_idx,
        "pos_prob": pos_prob,
        "string_idx": string_idx,
        "fret": fret,
        "midi": midi,
        "note_name": note_name,
        "note_octave": note_octave,
        "probs": probs.cpu().numpy(),
    }


# ============================
# CLI EXAMPLE
# ============================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="48-class position inference from WAV")
    parser.add_argument("ckpt", type=str, help="Path to 48-class checkpoint .pt")
    parser.add_argument("wav", type=str, help="Path to mono WAV file")
    args = parser.parse_args()

    model, device = load_position_model(args.ckpt)
    y, sr = librosa.load(args.wav, sr=None, mono=True)

    result = predict_position_from_waveform(model, device, y, sr)

    print("Prediction:")
    print(f"  string_idx={result['string_idx']} (1-based string={result['string_idx']+1})")
    print(f"  fret={result['fret']}")
    print(f"  midi={result['midi']}")
    print(f"  note={result['note_name']}{result['note_octave']}")
    print(f"  pos_idx={result['pos_idx']} prob={result['pos_prob']:.3f}")


if __name__ == "__main__":
    main()
