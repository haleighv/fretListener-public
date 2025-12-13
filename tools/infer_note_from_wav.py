#!/usr/bin/env python

import argparse
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torchaudio

import sys
from pathlib import Path

# Ensure <repo_root>/src is importable when running from tools/
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from note_model import NoteCNN



"""
USAGE:
cd ~/code/fretListener

python tools/infer_note_from_wav.py \
  --checkpoint artifacts/note_20251211_164726/note_cnn_best.pt \
  --wav data/strat_neck_40/organized/s5_f7/s5_f7_t23_sessB.wav

cd ~/code/fretListener

python tools/infer_note_from_wav.py \
  --checkpoint artifacts/models/note_20251211_164726/note_cnn_best.pt \
  --dataset-root data/strat_neck_40

"""


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def midi_to_name(midi: int) -> str:
    note_names = ["C", "C#", "D", "D#", "E", "F",
                  "F#", "G", "G#", "A", "A#", "B"]
    name = note_names[midi % 12]
    octave = (midi // 12) - 1
    return f"{name}{octave}"


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device)

    num_classes = ckpt["num_classes"]
    mel_params = ckpt["mel_params"]
    n_mels = mel_params["n_mels"]

    # Match training configuration
    model = NoteCNN(num_classes=num_classes, n_mels=n_mels, dropout_p=0.3)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    midi_to_idx = ckpt["midi_to_idx"]
    idx_to_midi = ckpt["idx_to_midi"]
    sample_rate = ckpt["sample_rate"]
    attack_samples = ckpt["attack_samples"]

    # Keep transforms on CPU (avoid MPS STFT issues)
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=mel_params["n_fft"],
        hop_length=mel_params["hop_length"],
        n_mels=mel_params["n_mels"],
        f_min=mel_params["f_min"],
        f_max=mel_params["f_max"],
    )
    db_transform = torchaudio.transforms.AmplitudeToDB()

    return {
        "model": model,
        "midi_to_idx": midi_to_idx,
        "idx_to_midi": idx_to_midi,
        "mel_spec": mel_spec,
        "db_transform": db_transform,
        "sample_rate": sample_rate,
        "attack_samples": attack_samples,
        "num_frames": mel_params["num_frames"],
        "device": device,
    }


def load_and_preprocess_wav(
    wav_path: Path,
    sample_rate: int,
    attack_samples: int,
    mel_spec: torchaudio.transforms.MelSpectrogram,
    db_transform: torchaudio.transforms.AmplitudeToDB,
    num_frames: int,
) -> torch.Tensor:
    if not wav_path.is_file():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    wav, sr = torchaudio.load(wav_path)  # [C, T]
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        wav = torchaudio.functional.resample(
            wav, orig_freq=sr, new_freq=sample_rate
        )

    if wav.size(1) < attack_samples:
        pad = attack_samples - wav.size(1)
        wav = torch.nn.functional.pad(wav, (0, pad))
    elif wav.size(1) > attack_samples:
        wav = wav[:, :attack_samples]

    # Transforms stay on CPU
    spec = mel_spec(wav)          # [1, n_mels, T]
    spec_db = db_transform(spec)  # [1, n_mels, T]

    # Per-sample normalization over time
    mean = spec_db.mean(dim=2, keepdim=True)
    std = spec_db.std(dim=2, keepdim=True) + 1e-6
    spec_db = (spec_db - mean) / std

    _, n_mels, T = spec_db.shape
    if T < num_frames:
        pad_t = num_frames - T
        spec_db = torch.nn.functional.pad(spec_db, (0, pad_t))
    elif T > num_frames:
        spec_db = spec_db[:, :, :num_frames]

    # Add batch dimension: [1, 1, n_mels, num_frames]
    spec_db = spec_db.unsqueeze(0)
    return spec_db


def predict_note(
    model: NoteCNN,
    spec_batch: torch.Tensor,
    idx_to_midi: Dict[int, int],
    device: torch.device,
):
    spec_batch = spec_batch.to(device, dtype=torch.float32)
    with torch.no_grad():
        logits = model(spec_batch)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    pred_idx_int = int(pred_idx.item())
    confidence = float(conf.item())
    midi = idx_to_midi[pred_idx_int]
    note_name = midi_to_name(midi)
    return midi, note_name, confidence


_POS_RE = re.compile(r"(?:^|/)(s(?P<s>\d+)_f(?P<f>\d+))(?:_|/|$)")


def parse_string_fret_from_path(p: Path) -> Optional[Tuple[int, int]]:
    """
    Extract (string, fret) from either:
      - filename containing "s{X}_f{Y}" (e.g., s5_f7_t23_sessB.wav)
      - or parent directory named "s{X}_f{Y}" (e.g., .../organized/s5_f7/...)
    """
    s = str(p.as_posix())

    m = _POS_RE.search(s)
    if m:
        try:
            string_idx = int(m.group("s"))
            fret = int(m.group("f"))
            return string_idx, fret
        except Exception:
            return None

    return None


def expected_midi_from_string_fret(
    string_idx_1based: int,
    fret: int,
    string_base_midi: List[int],
) -> Optional[int]:
    if string_idx_1based < 1 or string_idx_1based > len(string_base_midi):
        return None
    return int(string_base_midi[string_idx_1based - 1] + fret)


def run_single(
    state: Dict[str, Any],
    wav_path: Path,
) -> Tuple[int, str, float]:
    spec_batch = load_and_preprocess_wav(
        wav_path=wav_path,
        sample_rate=state["sample_rate"],
        attack_samples=state["attack_samples"],
        mel_spec=state["mel_spec"],
        db_transform=state["db_transform"],
        num_frames=state["num_frames"],
    )
    return predict_note(
        model=state["model"],
        spec_batch=spec_batch,
        idx_to_midi=state["idx_to_midi"],
        device=state["device"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Offline inference for NoteCNN on a single WAV file, or batch test on a dataset."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained note_cnn checkpoint (.pt).",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--wav",
        type=str,
        help="Path to input WAV file.",
    )
    group.add_argument(
        "--dataset-root",
        type=str,
        help="Dataset root to batch test (expects WAV paths containing s{string}_f{fret}).",
    )

    parser.add_argument(
        "--glob",
        type=str,
        default="organized/**/*.wav",
        help="Glob under --dataset-root to find WAVs. Default: organized/**/*.wav",
    )

    parser.add_argument(
        "--string-base-midi",
        type=str,
        default="64,59,55,50,45,40",
        help=(
            "Comma-separated base MIDI per string s1..sN. "
            "Default assumes standard tuning with s1=high E4(64): 64,59,55,50,45,40"
        ),
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap for batch mode (0 = no cap).",
    )

    args = parser.parse_args()

    device = get_device()
    print("Using device:", device)

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    state = load_checkpoint(ckpt_path, device)

    # Single-file mode
    if args.wav:
        wav_path = Path(args.wav).expanduser().resolve()
        midi, note_name, confidence = run_single(state, wav_path)

        print(f"WAV:        {wav_path}")
        print(f"Pred MIDI:  {midi}")
        print(f"Pred note:  {note_name}")
        print(f"Confidence: {confidence:.3f}")
        return

    # Batch mode
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    try:
        base_list = [int(x.strip()) for x in args.string_base_midi.split(",") if x.strip() != ""]
    except Exception:
        raise SystemExit(f"Invalid --string-base-midi: {args.string_base_midi}")

    wav_paths = sorted(dataset_root.glob(args.glob))
    wav_paths = [p for p in wav_paths if p.is_file()]
    if args.max_files and args.max_files > 0:
        wav_paths = wav_paths[: args.max_files]

    if not wav_paths:
        raise SystemExit(f"No WAVs found under {dataset_root} with glob: {args.glob}")

    total = 0
    correct = 0
    skipped = 0

    print(f"Batch testing dataset: {dataset_root}")
    print(f"Glob: {args.glob}")
    print(f"Files: {len(wav_paths)}")
    print(f"String base MIDI (s1..s{len(base_list)}): {base_list}")
    print()

    for p in wav_paths:
        pos = parse_string_fret_from_path(p)
        if not pos:
            skipped += 1
            continue

        s_idx, fret = pos
        exp_midi = expected_midi_from_string_fret(s_idx, fret, base_list)
        if exp_midi is None:
            skipped += 1
            continue

        pred_midi, pred_name, conf = run_single(state, p)
        exp_name = midi_to_name(exp_midi)

        ok = (pred_midi == exp_midi)
        total += 1
        if ok:
            correct += 1

        status = "OK" if ok else "MISS"
        print(
            f"{status}  exp={exp_name}({exp_midi})  "
            f"pred={pred_name}({pred_midi})  conf={conf:.3f}  "
            f"path={p.relative_to(dataset_root)}"
        )

    if total == 0:
        raise SystemExit("No evaluable files found (no parsable s{X}_f{Y} patterns).")

    acc = correct / total
    print()
    print("Batch summary")
    print("-------------")
    print(f"Evaluated: {total}")
    print(f"Correct:   {correct}")
    print(f"Skipped:   {skipped}")
    print(f"Accuracy:  {acc:.4f}")


if __name__ == "__main__":
    main()
