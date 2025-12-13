#!/usr/bin/env python

import argparse
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torchaudio

import sys

# Ensure <repo_root>/src is importable when running from tools/
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from note_model import NoteCNN


"""
USAGE (single file):
python tools/infer_note_from_wav.py \
  --checkpoint artifacts/models/note_20251211_164726/note_cnn_best.pt \
  --wav data/strat_neck_40/organized/s5_f7/s5_f7_t23_sessB.wav

USAGE (batch):
python tools/infer_note_from_wav.py \
  --checkpoint artifacts/models/note_20251211_164726/note_cnn_best.pt \
  --dataset-root data/strat_neck_40 \
  --margin 0.10
"""


# -----------------------------
# Device
# -----------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------
# MIDI helpers
# -----------------------------

def midi_to_name(midi: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F",
             "F#", "G", "G#", "A", "A#", "B"]
    note = names[midi % 12]
    octave = (midi // 12) - 1
    return f"{note}{octave}"


def midi_to_freq(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


# -----------------------------
# Checkpoint
# -----------------------------

def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device)

    mel_params = ckpt["mel_params"]

    model = NoteCNN(
        num_classes=ckpt["num_classes"],
        n_mels=mel_params["n_mels"],
        dropout_p=0.3,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=ckpt["sample_rate"],
        n_fft=mel_params["n_fft"],
        hop_length=mel_params["hop_length"],
        n_mels=mel_params["n_mels"],
        f_min=mel_params["f_min"],
        f_max=mel_params["f_max"],
    )
    db = torchaudio.transforms.AmplitudeToDB()

    return {
        "model": model,
        "mel": mel,
        "db": db,
        "idx_to_midi": ckpt["idx_to_midi"],
        "sample_rate": ckpt["sample_rate"],
        "attack_samples": ckpt["attack_samples"],
        "num_frames": mel_params["num_frames"],
        "device": device,
    }


# -----------------------------
# Preprocessing
# -----------------------------

def load_and_preprocess(
    wav_path: Path,
    state: Dict[str, Any],
) -> torch.Tensor:
    wav, sr = torchaudio.load(wav_path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != state["sample_rate"]:
        wav = torchaudio.functional.resample(
            wav, sr, state["sample_rate"]
        )

    T = state["attack_samples"]
    if wav.size(1) < T:
        wav = torch.nn.functional.pad(wav, (0, T - wav.size(1)))
    else:
        wav = wav[:, :T]

    spec = state["mel"](wav)
    spec = state["db"](spec)

    mean = spec.mean(dim=2, keepdim=True)
    std = spec.std(dim=2, keepdim=True) + 1e-6
    spec = (spec - mean) / std

    _, _, t = spec.shape
    nf = state["num_frames"]
    if t < nf:
        spec = torch.nn.functional.pad(spec, (0, nf - t))
    else:
        spec = spec[:, :, :nf]

    return spec.unsqueeze(0)  # [1,1,n_mels,n_frames]


# -----------------------------
# Pitch tie-break
# -----------------------------

def estimate_f0_autocorr(x: np.ndarray, sr: int) -> Optional[float]:
    if x.size < int(0.02 * sr):
        return None

    x = x.astype(np.float32)
    x -= np.mean(x)
    x /= (np.max(np.abs(x)) + 1e-6)

    r = np.correlate(x, x, mode="full")[x.size - 1:]

    fmin, fmax = 70.0, 1200.0
    lag_min = int(sr / fmax)
    lag_max = int(sr / fmin)

    if lag_max <= lag_min or lag_max >= len(r):
        return None

    seg = r[lag_min:lag_max]
    lag = lag_min + np.argmax(seg)
    if lag <= 0:
        return None

    return sr / lag


# -----------------------------
# Path parsing
# -----------------------------

_POS_RE = re.compile(r"(?:^|/)(s(?P<s>\d+)_f(?P<f>\d+))(?:_|/|$)")


def parse_string_fret(p: Path) -> Optional[Tuple[int, int]]:
    m = _POS_RE.search(str(p))
    if not m:
        return None
    return int(m.group("s")), int(m.group("f"))


def expected_midi(string_idx: int, fret: int, bases: List[int]) -> Optional[int]:
    if string_idx < 1 or string_idx > len(bases):
        return None
    return bases[string_idx - 1] + fret


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--wav", type=str)
    ap.add_argument("--dataset-root", type=str)
    ap.add_argument("--glob", default="organized/**/*.wav")
    ap.add_argument("--string-base-midi",
                    default="64,59,55,50,45,40")
    ap.add_argument("--margin", type=float, default=0.10)
    args = ap.parse_args()

    device = get_device()
    state = load_checkpoint(Path(args.checkpoint), device)

    # ---------- Single ----------
    if args.wav:
        spec = load_and_preprocess(Path(args.wav), state)
        spec = spec.to(device)

        with torch.no_grad():
            probs = torch.softmax(state["model"](spec), dim=1)[0]

        vals, idxs = torch.topk(probs, 5)
        for i, (v, idx) in enumerate(zip(vals, idxs), 1):
            midi = state["idx_to_midi"][int(idx)]
            print(f"{i}) {midi_to_name(midi)}  prob={float(v):.3f}")
        return

    # ---------- Batch ----------
    bases = [int(x) for x in args.string_base_midi.split(",")]
    root = Path(args.dataset_root)
    files = sorted(root.glob(args.glob))

    total = correct = ambiguous = tiebreak_run = tiebreak_changed = 0

    for p in files:
        pos = parse_string_fret(p)
        if not pos:
            continue

        exp = expected_midi(pos[0], pos[1], bases)
        if exp is None:
            continue

        spec = load_and_preprocess(p, state).to(device)
        with torch.no_grad():
            probs = torch.softmax(state["model"](spec), dim=1)[0]

        vals, idxs = torch.topk(probs, 2)
        p1, p2 = float(vals[0]), float(vals[1])
        m1 = state["idx_to_midi"][int(idxs[0])]
        m2 = state["idx_to_midi"][int(idxs[1])]

        total += 1
        if m1 == exp:
            correct += 1

        if (p1 - p2) < args.margin:
            ambiguous += 1
            if abs(m1 - m2) == 1:
                tiebreak_run += 1
                wav, _ = torchaudio.load(p)
                f0 = estimate_f0_autocorr(wav[0].numpy(), state["sample_rate"])
                if f0:
                    chosen = m1 if abs(f0 - midi_to_freq(m1)) < abs(f0 - midi_to_freq(m2)) else m2
                    if chosen != m1:
                        tiebreak_changed += 1

    print()
    print("Batch summary")
    print("-------------")
    print(f"Evaluated:           {total}")
    print(f"Correct (top1):      {correct}")
    print(f"Accuracy (top1):     {correct/total:.4f}")
    print(f"Ambiguous (<margin): {ambiguous}")
    print(f"Tie-break run:       {tiebreak_run}")
    print(f"Tie-break changed:   {tiebreak_changed}")


if __name__ == "__main__":
    main()
