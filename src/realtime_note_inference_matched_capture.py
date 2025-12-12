#!/usr/bin/env python3

import argparse
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import sounddevice as sd
import torch
import torchaudio
from scipy.io import wavfile

from note_model import NoteCNN


"""
Realtime note inference with capture path matched to guitarDatasetCollector.py:

- peak-based trigger on selected channel (not RMS)
- circular pre-buffer (default 50ms)
- post-trigger capture (default 0.50s)
- optional sweep over windows inside captured segment to find best confidence
- mel+db preprocessing on CPU (matches training transforms; avoids MPS FFT limitations)
- optional WAV saving: raw captured audio + best inference window

Example usage:

python src/realtime_note_inference_matched_capture.py \
  --checkpoint artifacts/note_20251211_164726/note_cnn_best.pt \
  --trigger-threshold 0.01 \
  --pre-ms 50 \
  --post-ms 500 \
  --confidence 0.85 \
  --sweep-step-ms 5 \
  --save-dir artifacts/captures \
  --block-size 512
"""


# -----------------------------
# Compute device
# -----------------------------

def get_model_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------
# MIDI helpers
# -----------------------------

def midi_to_name(midi: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note = names[midi % 12]
    octave = (midi // 12) - 1
    return f"{note}{octave}"


# -----------------------------
# Checkpoint + transforms
# -----------------------------

def load_checkpoint(checkpoint_path: Path, model_device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    num_classes = int(ckpt["num_classes"])
    mel_params = ckpt["mel_params"]
    n_mels = int(mel_params["n_mels"])

    model = NoteCNN(num_classes=num_classes, n_mels=n_mels, dropout_p=0.0)
    model.load_state_dict(ckpt["model_state"])
    model.to(model_device)
    model.eval()

    # IMPORTANT: mel+db on CPU to avoid MPS FFT operator issues
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=int(ckpt["sample_rate"]),
        n_fft=int(mel_params["n_fft"]),
        hop_length=int(mel_params["hop_length"]),
        n_mels=int(mel_params["n_mels"]),
        f_min=float(mel_params["f_min"]),
        f_max=float(mel_params["f_max"]),
    )
    db = torchaudio.transforms.AmplitudeToDB()

    return {
        "model": model,
        "model_device": model_device,
        "mel": mel,
        "db": db,
        "idx_to_midi": ckpt["idx_to_midi"],
        "sample_rate": int(ckpt["sample_rate"]),
        "attack_samples": int(ckpt["attack_samples"]),  # training window length
        "num_frames": int(mel_params["num_frames"]),
    }


def waveform_to_logmel_cpu(
    wav_1d: np.ndarray,
    mel: torchaudio.transforms.MelSpectrogram,
    db: torchaudio.transforms.AmplitudeToDB,
    num_frames: int,
) -> torch.Tensor:
    """
    Training-identical preprocessing: log-mel + per-sample mean/std over time + pad/crop to num_frames.
    Runs entirely on CPU.
    Returns: [1, 1, n_mels, num_frames]
    """
    wav = wav_1d.astype(np.float32)

    # No forced normalization here except DC removal; peak normalization is handled by caller when desired.
    wav = wav - float(np.mean(wav))

    wav_t = torch.from_numpy(wav).unsqueeze(0)  # [1, T]

    spec = mel(wav_t)         # [1, n_mels, T]
    spec_db = db(spec)        # [1, n_mels, T]

    mean = spec_db.mean(dim=2, keepdim=True)
    std = spec_db.std(dim=2, keepdim=True) + 1e-6
    spec_db = (spec_db - mean) / std

    _, _, t = spec_db.shape
    if t < num_frames:
        spec_db = torch.nn.functional.pad(spec_db, (0, num_frames - t))
    elif t > num_frames:
        spec_db = spec_db[:, :, :num_frames]

    return spec_db.unsqueeze(0)  # [1, 1, n_mels, num_frames]


def predict_window(
    state: Dict[str, Any],
    window_1d: np.ndarray,
    peak_normalize: bool = True,
) -> Tuple[int, str, float]:
    """
    Predict MIDI + name + confidence for a single window.
    """
    w = window_1d.astype(np.float32)

    if peak_normalize:
        peak = float(np.max(np.abs(w)))
        if peak > 0:
            w = (w / peak) * 0.95

    spec = waveform_to_logmel_cpu(
        wav_1d=w,
        mel=state["mel"],
        db=state["db"],
        num_frames=state["num_frames"],
    )

    # Model may be on mps/cuda/cpu; move features only (mel stays CPU)
    spec = spec.to(state["model_device"], dtype=torch.float32)

    with torch.no_grad():
        logits = state["model"](spec)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)

    conf_f = float(conf.item())
    idx_i = int(idx.item())
    midi = int(state["idx_to_midi"][idx_i])
    return midi, midi_to_name(midi), conf_f


def sweep_best_window(
    captured: np.ndarray,
    state: Dict[str, Any],
    sweep_step_samples: int,
    conf_threshold: float,
) -> Tuple[int, str, float, int]:
    """
    Slide an attack-sized window across the captured audio and return:
      (best_midi, best_name, best_conf, best_start_sample)

    If any window exceeds conf_threshold, returns immediately with that window.
    """
    attack = state["attack_samples"]
    if captured.shape[0] < attack:
        # pad if short
        padded = np.zeros(attack, dtype=np.float32)
        padded[:captured.shape[0]] = captured.astype(np.float32)
        captured = padded

    best_midi, best_name, best_conf, best_start = -1, "", 0.0, 0

    last_start = max(0, captured.shape[0] - attack)
    for start in range(0, last_start + 1, max(1, sweep_step_samples)):
        window = captured[start:start + attack]
        midi, name, conf = predict_window(state, window, peak_normalize=True)

        if conf > best_conf:
            best_midi, best_name, best_conf, best_start = midi, name, conf, start

        if conf >= conf_threshold:
            return midi, name, conf, start

    return best_midi, best_name, best_conf, best_start


# -----------------------------
# Audio device selection
# -----------------------------

def choose_input_device() -> int:
    devices = sd.query_devices()
    print("Available audio devices (inputs only):")
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            print(f"{idx:2d}: {d['name']}  (inputs: {d['max_input_channels']})")

    while True:
        v = input("Select input device index: ").strip()
        try:
            dev = int(v)
        except ValueError:
            print("Please enter an integer device index.")
            continue

        if 0 <= dev < len(devices) and devices[dev].get("max_input_channels", 0) > 0:
            return dev

        print("Invalid device index or device has no input channels.")


def choose_channel(max_channels: int) -> int:
    print(f"Input channels available: 1..{max_channels}")
    while True:
        v = input("Select channel (1-based index): ").strip()
        try:
            ch = int(v)
        except ValueError:
            print("Please enter an integer channel number.")
            continue

        if 1 <= ch <= max_channels:
            return ch - 1

        print("Channel out of range.")


# -----------------------------
# WAV saving
# -----------------------------

def save_wav_int16(path: Path, sr: int, audio_float32: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(audio_float32.astype(np.float32), -1.0, 1.0)
    x_i16 = np.int16(x * 32767)
    wavfile.write(str(path), sr, x_i16)


# -----------------------------
# Main
# -----------------------------

def main():
    p = argparse.ArgumentParser(description="Realtime inference with matched capture path.")
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--trigger-threshold", type=float, default=0.01,
                   help="Peak amplitude threshold to trigger capture (collector-style).")
    p.add_argument("--pre-ms", type=float, default=50.0,
                   help="Pre-trigger audio to keep (ms). Default 50ms.")
    p.add_argument("--post-ms", type=float, default=500.0,
                   help="Post-trigger capture duration (ms). Default 500ms.")
    p.add_argument("--max-wait-sec", type=float, default=10.0,
                   help="Max time to wait for a trigger before printing timeout and re-arming.")
    p.add_argument("--block-size", type=int, default=512,
                   help="InputStream blocksize. Default 512 (responsive detection).")

    # Re-arm control (peak-based)
    p.add_argument("--silence-threshold", type=float, default=0.006,
                   help="Peak amplitude below this counts as silence for re-arming.")
    p.add_argument("--min-silence-ms", type=float, default=80.0,
                   help="How long signal must stay below silence-threshold before re-arming.")

    # Windowing strategy
    p.add_argument("--confidence", type=float, default=0.85,
                   help="Stop sweep early if a window exceeds this confidence.")
    p.add_argument("--sweep-step-ms", type=float, default=5.0,
                   help="Window sweep step size (ms). Default 5ms.")

    # Debug capture saving
    p.add_argument("--save-dir", type=str, default="",
                   help="If set, saves captured audio and chosen window WAVs to this directory.")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    model_device = get_model_device()
    print("Using compute device:", model_device)

    state = load_checkpoint(ckpt_path, model_device)

    sr = state["sample_rate"]
    attack = state["attack_samples"]

    pre_samples = int(round((args.pre_ms / 1000.0) * sr))
    post_samples = int(round((args.post_ms / 1000.0) * sr))
    capture_total = pre_samples + post_samples

    sweep_step_samples = max(1, int(round((args.sweep_step_ms / 1000.0) * sr)))
    min_silence_samples = max(1, int(round((args.min_silence_ms / 1000.0) * sr)))
    max_wait_samples = int(round(args.max_wait_sec * sr))

    print(f"Model sample rate: {sr} Hz")
    print(f"Training attack window: {attack} samples ({attack/sr:.3f}s)")
    print(f"Capture (matched): pre={pre_samples} ({args.pre_ms:.1f}ms) "
          f"+ post={post_samples} ({args.post_ms:.1f}ms) "
          f"= {capture_total} samples ({capture_total/sr:.3f}s)")
    print(f"Sweep step: {sweep_step_samples} samples ({args.sweep_step_ms:.1f}ms)")
    print(f"Trigger threshold (peak): {args.trigger_threshold}")
    print(f"Silence threshold (peak): {args.silence_threshold} "
          f"for {min_silence_samples} samples ({args.min_silence_ms:.1f}ms)")
    if args.save_dir:
        print(f"WAV saving enabled: {args.save_dir}")

    # Choose device/channel
    dev = choose_input_device()
    dev_info = sd.query_devices(dev, "input")
    print(f"Selected device: {dev} - {dev_info['name']}")
    ch = choose_channel(int(dev_info["max_input_channels"]))
    print(f"Selected channel: {ch + 1} (0-based index {ch})")

    # State machine
    mode = "armed"  # armed -> capturing -> waiting_silence
    timeout_ctr = 0
    silence_ctr = 0

    # Pre-buffer (collector-style circular) holds all channels, but only selected channel is used for inference
    max_channels = int(dev_info["max_input_channels"])
    pre_buffer = np.zeros((pre_samples, max_channels), dtype=np.float32) if pre_samples > 0 else None
    pre_idx = 0

    # Capture buffers
    triggered = False
    recording_parts = []
    recorded_post = 0
    captured_audio = None  # 1D float32, selected channel only

    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else None

    def reset_capture_state():
        nonlocal triggered, recording_parts, recorded_post, captured_audio, timeout_ctr, pre_idx
        triggered = False
        recording_parts = []
        recorded_post = 0
        captured_audio = None
        timeout_ctr = 0
        pre_idx = 0

    reset_capture_state()

    def callback(indata, frames, time_info, status):
        nonlocal mode, timeout_ctr, silence_ctr
        nonlocal pre_buffer, pre_idx
        nonlocal triggered, recording_parts, recorded_post, captured_audio

        if status:
            # avoid spam; print only if needed
            if "overflow" in str(status).lower():
                print("Audio overflow detected", flush=True)

        # selected channel extraction (collector-style)
        if ch < indata.shape[1]:
            chan = indata[:, ch].astype(np.float32, copy=True)
        else:
            chan = indata[:, 0].astype(np.float32, copy=True)

        peak = float(np.max(np.abs(chan))) if chan.size else 0.0

        if mode == "armed":
            timeout_ctr += frames

            # update pre-trigger circular buffer with ALL channels (collector-style)
            if pre_buffer is not None and pre_samples > 0:
                # write frames (handle wrap)
                n = min(frames, pre_samples)
                for i in range(n):
                    pre_buffer[pre_idx] = indata[i].astype(np.float32)
                    pre_idx = (pre_idx + 1) % pre_samples

            if peak >= args.trigger_threshold:
                # trigger: assemble capture segment
                triggered = True

                # reconstruct chronological pre-audio from circular buffer
                if pre_samples > 0 and pre_buffer is not None:
                    if pre_idx > 0:
                        pre_chron = np.concatenate([
                            pre_buffer[pre_idx:, ch if ch < max_channels else 0],
                            pre_buffer[:pre_idx, ch if ch < max_channels else 0],
                        ])
                    else:
                        pre_chron = pre_buffer[:, ch if ch < max_channels else 0]
                    pre_audio = pre_chron[-pre_samples:].astype(np.float32)
                else:
                    pre_audio = np.zeros((0,), dtype=np.float32)

                recording_parts = [pre_audio, chan]
                recorded_post = frames  # count post-trigger samples (chan already included)

                mode = "capturing"
                timeout_ctr = 0
                print("Triggered: capturing...", flush=True)

            elif timeout_ctr >= max_wait_samples:
                print("Timeout: no trigger detected. Re-armed.", flush=True)
                timeout_ctr = 0

        elif mode == "capturing":
            # collect until post_samples reached
            recording_parts.append(chan)
            recorded_post += frames

            if recorded_post >= post_samples:
                # finalize capture
                cap = np.concatenate(recording_parts).astype(np.float32)

                # trim/pad to exact capture_total
                if cap.shape[0] > capture_total:
                    cap = cap[:capture_total]
                elif cap.shape[0] < capture_total:
                    cap = np.pad(cap, (0, capture_total - cap.shape[0]), mode="constant")

                captured_audio = cap

                # Run sweep over windows to pick best matching attack window
                midi, name, conf, start = sweep_best_window(
                    captured=captured_audio,
                    state=state,
                    sweep_step_samples=sweep_step_samples,
                    conf_threshold=args.confidence,
                )

                # Save debug WAVs if requested
                if save_dir is not None:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    raw_path = save_dir / f"cap_{ts}_raw.wav"
                    best_path = save_dir / f"cap_{ts}_best_{name}_conf{conf:.2f}.wav"
                    save_wav_int16(raw_path, sr, captured_audio)

                    # best window (attack-sized)
                    best_win = captured_audio[start:start + attack]
                    if best_win.shape[0] < attack:
                        best_win = np.pad(best_win, (0, attack - best_win.shape[0]), mode="constant")
                    save_wav_int16(best_path, sr, best_win)

                print(
                    f"Detected: {name} (MIDI {midi}) conf={conf:.3f} "
                    f"best_offset={start/sr:.3f}s peak={peak:.3f}",
                    flush=True,
                )

                # move to waiting for silence
                mode = "waiting_silence"
                silence_ctr = 0

        elif mode == "waiting_silence":
            if peak <= args.silence_threshold:
                silence_ctr += frames
            else:
                silence_ctr = 0

            if silence_ctr >= min_silence_samples:
                print("Silence detected, re-arming.", flush=True)
                mode = "armed"
                silence_ctr = 0
                reset_capture_state()

    print()
    print("Trigger armed. Play one note at a time.")
    print("Workflow:")
    print("  1) Wait for peak above trigger threshold -> trigger")
    print("  2) Capture pre+post segment (collector-style)")
    print("  3) Sweep windows, pick best confidence window")
    print("  4) Wait for silence -> re-arm")
    print("Ctrl+C to stop.")
    print()

    try:
        with sd.InputStream(
            device=dev,
            channels=max_channels,
            samplerate=sr,
            dtype="float32",
            blocksize=args.block_size,
            callback=callback,
        ):
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"Audio error: {e}")


if __name__ == "__main__":
    main()
