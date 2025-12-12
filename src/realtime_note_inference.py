#!/usr/bin/env python

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import sounddevice as sd
import torch
import torchaudio

from note_model import NoteCNN

"""
Example usage:

python src/realtime_note_inference.py \
  --checkpoint artifacts/note_20251211_164726/note_cnn_best.pt \
  --threshold 0.02 \
  --silence-ratio 0.5 \
  --min-silence-ms 80
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

    # Model on compute device (MPS/CUDA/CPU)
    model = NoteCNN(num_classes=num_classes, n_mels=n_mels, dropout_p=0.3)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    midi_to_idx = ckpt["midi_to_idx"]
    idx_to_midi = ckpt["idx_to_midi"]
    sample_rate = ckpt["sample_rate"]
    attack_samples = ckpt["attack_samples"]

    # Keep torchaudio transforms on CPU to avoid MPS STFT issues
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


def build_logmel_from_buffer(
    buffer_1d: np.ndarray,
    sample_rate: int,
    attack_samples: int,
    mel_spec: torchaudio.transforms.MelSpectrogram,
    db_transform: torchaudio.transforms.AmplitudeToDB,
    num_frames: int,
) -> torch.Tensor:
    """
    buffer_1d: numpy array [attack_samples], float64
    Returns: torch tensor [1, 1, n_mels, num_frames] on CPU.
    """
    assert buffer_1d.ndim == 1
    if buffer_1d.shape[0] != attack_samples:
        raise ValueError(
            f"Expected buffer of length {attack_samples}, got {buffer_1d.shape[0]}"
        )

    # Waveform on CPU
    wav = torch.from_numpy(buffer_1d.astype(np.float32)).unsqueeze(0)  # [1, T]

    # Torchaudio transforms on CPU
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

    # [1, 1, n_mels, num_frames] on CPU
    spec_db = spec_db.unsqueeze(0)
    return spec_db


def predict_from_buffer(
    state: Dict[str, Any],
    buffer_1d: np.ndarray,
):
    spec_batch_cpu = build_logmel_from_buffer(
        buffer_1d=buffer_1d,
        sample_rate=state["sample_rate"],
        attack_samples=state["attack_samples"],
        mel_spec=state["mel_spec"],
        db_transform=state["db_transform"],
        num_frames=state["num_frames"],
    )

    model = state["model"]
    idx_to_midi = state["idx_to_midi"]
    device = state["device"]

    # Move log-mel batch to compute device
    spec_batch = spec_batch_cpu.to(device, dtype=torch.float32)

    with torch.no_grad():
        logits = model(spec_batch)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    pred_idx_int = int(pred_idx.item())
    confidence = float(conf.item())
    midi = idx_to_midi[pred_idx_int]
    note_name = midi_to_name(midi)
    return midi, note_name, confidence


def choose_input_device() -> int:
    print("Available audio devices:")
    devices = sd.query_devices()
    for idx, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            print(f"{idx:2d}: {d['name']}  (inputs: {d['max_input_channels']})")

    while True:
        value = input("Select input device index: ").strip()
        try:
            dev_idx = int(value)
        except ValueError:
            print("Please enter an integer device index.")
            continue
        if 0 <= dev_idx < len(devices) and devices[dev_idx]["max_input_channels"] > 0:
            return dev_idx
        print("Invalid device index or device has no input channels.")


def choose_channel(max_channels: int) -> int:
    print(f"Input channels available: 1..{max_channels}")
    while True:
        value = input("Select channel (1-based index): ").strip()
        try:
            ch = int(value)
        except ValueError:
            print("Please enter an integer channel number.")
            continue
        if 1 <= ch <= max_channels:
            return ch - 1  # zero-based index
        print("Channel out of range.")


def main():
    parser = argparse.ArgumentParser(
        description="Realtime note inference using NoteCNN with trigger and re-arm."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained note_cnn checkpoint (.pt).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Amplitude RMS threshold for trigger detection. Default: 0.02",
    )
    parser.add_argument(
        "--silence-ratio",
        type=float,
        default=0.5,
        help="Silence threshold as fraction of trigger threshold. Default: 0.5",
    )
    parser.add_argument(
        "--min-silence-ms",
        type=float,
        default=80.0,
        help="Minimum silence duration to re-arm trigger (ms). Default: 80 ms.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="Audio block size in frames. Default: 1024.",
    )
    args = parser.parse_args()

    device = get_device()
    print("Using compute device:", device)

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    state = load_checkpoint(ckpt_path, device)

    sample_rate = state["sample_rate"]
    attack_samples = state["attack_samples"]

    print(f"Model sample rate: {sample_rate} Hz")
    print(f"Attack window:     {attack_samples} samples "
          f"({attack_samples / sample_rate:.3f} s)")

    # Prompt for input device and channel
    dev_idx = choose_input_device()
    dev_info = sd.query_devices(dev_idx)
    print(f"Selected device: {dev_idx} - {dev_info['name']}")

    ch_idx = choose_channel(dev_info["max_input_channels"])
    print(f"Selected channel: {ch_idx + 1} (0-based index {ch_idx})")

    block_size = args.block_size
    trig_thresh = args.threshold
    silence_thresh = args.silence_ratio * trig_thresh
    min_silence_samples = int((args.min_silence_ms / 1000.0) * sample_rate)

    print()
    print("Trigger armed. Play one note at a time.")
    print("Workflow:")
    print("  1) Wait for signal above threshold -> trigger")
    print("  2) Capture attack window and run inference")
    print("  3) Wait for signal to fall quiet for min silence -> re-arm")
    print("Press Ctrl+C to stop.")
    print()

    # Ring buffer for last attack_samples
    buffer = np.zeros(attack_samples, dtype=np.float32)

    state_mode = "armed"  # "armed" or "waiting_silence"
    silence_counter = 0

    def audio_callback(indata, frames, time, status):
        nonlocal buffer, state_mode, silence_counter

        if status:
            print(f"Audio callback status: {status}", flush=True)

        # Extract selected channel, flatten to 1D float32
        mono = indata[:, ch_idx].astype(np.float32)

        # Update ring buffer
        if len(mono) >= attack_samples:
            buffer = mono[-attack_samples:]
        else:
            buffer = np.roll(buffer, -len(mono))
            buffer[-len(mono):] = mono

        # Compute RMS over current block for trigger logic
        rms = float(np.sqrt(np.mean(mono**2))) if mono.size > 0 else 0.0

        if state_mode == "armed":
            if rms >= trig_thresh:
                # Trigger detection
                try:
                    midi, note_name, confidence = predict_from_buffer(
                        state=state,
                        buffer_1d=buffer.copy(),
                    )
                    print(
                        f"Triggered: RMS={rms:.4f}  "
                        f"Pred: {note_name} (MIDI {midi}, conf {confidence:.3f})"
                    )
                except Exception as e:
                    print(f"Error during inference: {e}")

                state_mode = "waiting_silence"
                silence_counter = 0

        elif state_mode == "waiting_silence":
            if rms <= silence_thresh:
                silence_counter += frames
            else:
                silence_counter = 0

            if silence_counter >= min_silence_samples:
                print("Silence detected, re-arming trigger.")
                state_mode = "armed"
                silence_counter = 0

    try:
        with sd.InputStream(
            device=dev_idx,
            channels=dev_info["max_input_channels"],
            samplerate=sample_rate,
            dtype="float32",
            blocksize=block_size,
            callback=audio_callback,
        ):
            print("Realtime inference stream active.")
            print("Ctrl+C to exit.")
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopping realtime inference.")
    except Exception as e:
        print(f"Audio error: {e}")


if __name__ == "__main__":
    main()
