#!/usr/bin/env python

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torchaudio

from note_model import NoteCNN


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


def main():
    parser = argparse.ArgumentParser(
        description="Offline inference for NoteCNN on a single WAV file."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained note_cnn checkpoint (.pt).",
    )
    parser.add_argument(
        "--wav",
        type=str,
        required=True,
        help="Path to input WAV file.",
    )
    args = parser.parse_args()

    device = get_device()
    print("Using device:", device)

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    wav_path = Path(args.wav).expanduser().resolve()

    state = load_checkpoint(ckpt_path, device)
    spec_batch = load_and_preprocess_wav(
        wav_path=wav_path,
        sample_rate=state["sample_rate"],
        attack_samples=state["attack_samples"],
        mel_spec=state["mel_spec"],
        db_transform=state["db_transform"],
        num_frames=state["num_frames"],
    )

    midi, note_name, confidence = predict_note(
        model=state["model"],
        spec_batch=spec_batch,
        idx_to_midi=state["idx_to_midi"],
        device=device,
    )

    print(f"WAV:        {wav_path}")
    print(f"Pred MIDI:  {midi}")
    print(f"Pred note:  {note_name}")
    print(f"Confidence: {confidence:.3f}")


if __name__ == "__main__":
    main()
