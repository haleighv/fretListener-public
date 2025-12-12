#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List
from datetime import datetime

import numpy as np
import sounddevice as sd
import torch
import torchaudio

from note_model import NoteCNN

'''
Example usage:
python src/realtime_note_inference_sweep.py \
  --checkpoint artifacts/note_20251211_164726/note_cnn_best.pt \
  --threshold 0.02 \
  --silence-ratio 0.5 \
  --min-silence-ms 80 \
  --capture-sec 0.9 \
  --sweep-step-ms 5 \
  --confidence 0.85 \
  --block-size 1024

With debug saving:
python src/realtime_note_inference_sweep.py \
  --checkpoint artifacts/note_20251211_164726/note_cnn_best.pt \
  --threshold 0.02 \
  --silence-ratio 0.5 \
  --min-silence-ms 80 \
  --capture-sec 0.9 \
  --sweep-step-ms 5 \
  --confidence 0.85 \
  --block-size 1024 \
  --save-dir artifacts/realtime_debug \
  --topk 5
'''

# -----------------------------
# Device selection
# -----------------------------

def get_compute_device() -> torch.device:
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


# -----------------------------
# Checkpoint loading
# -----------------------------

def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")

    model = NoteCNN(
        num_classes=ckpt["num_classes"],
        n_mels=ckpt["mel_params"]["n_mels"],
        dropout_p=0.0,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # IMPORTANT:
    # Mel + dB transforms are kept on CPU to avoid MPS FFT limitations.
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=ckpt["sample_rate"],
        n_fft=ckpt["mel_params"]["n_fft"],
        hop_length=ckpt["mel_params"]["hop_length"],
        n_mels=ckpt["mel_params"]["n_mels"],
        f_min=ckpt["mel_params"]["f_min"],
        f_max=ckpt["mel_params"]["f_max"],
    )  # CPU

    db = torchaudio.transforms.AmplitudeToDB()  # CPU

    return {
        "model": model,
        "mel": mel,
        "db": db,
        "idx_to_midi": ckpt["idx_to_midi"],
        "midi_to_idx": ckpt.get("midi_to_idx", None),
        "sample_rate": ckpt["sample_rate"],
        "attack_samples": ckpt["attack_samples"],
        "num_frames": ckpt["mel_params"]["num_frames"],
        "mel_params": ckpt["mel_params"],
        "device": device,
    }


# -----------------------------
# Preprocessing (CPU, training-identical)
# -----------------------------

def waveform_to_logmel(
    wav: np.ndarray,
    mel,
    db,
    num_frames: int,
) -> torch.Tensor:
    wav = wav.astype(np.float32)
    wav = wav - np.mean(wav)  # DC removal
    wav_t = torch.from_numpy(wav).unsqueeze(0)  # [1, T]

    spec = mel(wav_t)         # [1, n_mels, T]
    spec = db(spec)           # [1, n_mels, T]

    mean = spec.mean(dim=2, keepdim=True)
    std = spec.std(dim=2, keepdim=True) + 1e-6
    spec = (spec - mean) / std

    _, _, t = spec.shape
    if t < num_frames:
        spec = torch.nn.functional.pad(spec, (0, num_frames - t))
    elif t > num_frames:
        spec = spec[:, :, :num_frames]

    return spec.unsqueeze(0)  # [1, 1, n_mels, frames]


def compute_topk(
    model: torch.nn.Module,
    spec_batch: torch.Tensor,
    idx_to_midi: Dict[int, int],
    k: int,
) -> List[Tuple[int, str, float]]:
    # returns [(midi, name, prob), ...] sorted desc by prob
    with torch.no_grad():
        logits = model(spec_batch)
        probs = torch.softmax(logits, dim=1).squeeze(0)  # [C]
        k = max(1, min(int(k), int(probs.numel())))
        vals, idxs = torch.topk(probs, k=k)

    out: List[Tuple[int, str, float]] = []
    for p, i in zip(vals.tolist(), idxs.tolist()):
        midi = idx_to_midi[int(i)]
        out.append((int(midi), midi_to_name(int(midi)), float(p)))
    return out


# -----------------------------
# Debug saving
# -----------------------------

def save_wav_mono(path: Path, wav: np.ndarray, sample_rate: int) -> None:
    """
    Save mono float32 wav via torchaudio.save.
    Expects wav shape [T] (np.ndarray).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    x = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)  # [1, T]
    torchaudio.save(str(path), x, sample_rate)


def make_event_dir(save_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = save_dir / f"event_{ts}"
    out.mkdir(parents=True, exist_ok=False)
    return out


# -----------------------------
# Sweep inference
# -----------------------------

def sweep_inference(
    buffer: np.ndarray,
    state: Dict[str, Any],
    sweep_step_samples: int,
    conf_threshold: float,
) -> Tuple[int, str, float, int, np.ndarray]:
    """
    Returns:
      midi, name, conf, offset_samples, chosen_window
    If no window meets conf_threshold, returns the best window found.
    """
    best_midi = -1
    best_name = ""
    best_conf = 0.0
    best_start = -1
    best_window: np.ndarray = np.zeros((state["attack_samples"],), dtype=np.float32)

    attack = state["attack_samples"]
    model = state["model"]
    mel = state["mel"]
    db = state["db"]
    device = state["device"]
    idx_to_midi = state["idx_to_midi"]

    if sweep_step_samples <= 0:
        sweep_step_samples = 1

    for start in range(0, len(buffer) - attack + 1, sweep_step_samples):
        window = buffer[start:start + attack].astype(np.float32)

        # Peak normalize (keeps shape, reduces gain mismatch)
        peak = float(np.max(np.abs(window)))
        if peak > 0:
            window = (window / peak) * 0.95

        spec = waveform_to_logmel(
            window,
            mel=mel,
            db=db,
            num_frames=state["num_frames"],
        ).to(device, dtype=torch.float32)

        with torch.no_grad():
            logits = model(spec)
            probs = torch.softmax(logits, dim=1)
            conf_t, idx_t = torch.max(probs, dim=1)

        conf = float(conf_t.item())
        midi = int(idx_to_midi[int(idx_t.item())])
        name = midi_to_name(midi)

        if conf > best_conf:
            best_conf = conf
            best_midi = midi
            best_name = name
            best_start = start
            best_window = window.copy()

        if conf >= conf_threshold:
            return midi, name, conf, start, window.copy()

    return best_midi, best_name, best_conf, best_start, best_window


# -----------------------------
# Audio device helpers
# -----------------------------

def choose_input_device() -> int:
    devices = sd.query_devices()
    print("Available input devices:")
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            print(f"{i:2d}: {d['name']} ({d['max_input_channels']} ch)")
    while True:
        v = input("Select device index: ").strip()
        try:
            idx = int(v)
        except ValueError:
            continue
        if 0 <= idx < len(devices) and devices[idx]["max_input_channels"] > 0:
            return idx


def choose_channel(max_ch: int) -> int:
    while True:
        v = input(f"Select channel (1-{max_ch}): ").strip()
        try:
            ch = int(v)
        except ValueError:
            continue
        if 1 <= ch <= max_ch:
            return ch - 1


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Realtime note inference with sweep + optional debug saving.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--silence-ratio", type=float, default=0.5)
    parser.add_argument("--min-silence-ms", type=float, default=80)
    parser.add_argument("--capture-sec", type=float, default=0.9)
    parser.add_argument("--sweep-step-ms", type=float, default=5)
    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--save-dir", type=str, default=None, help="If set, write per-trigger debug artifacts here.")
    parser.add_argument("--topk", type=int, default=5, help="Top-K predictions to print/save for chosen window.")
    args = parser.parse_args()

    device = get_compute_device()
    print("Using compute device:", device)

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    state = load_checkpoint(ckpt_path, device)

    sr = int(state["sample_rate"])
    capture_samples = int(args.capture_sec * sr)
    sweep_step = int((args.sweep_step_ms / 1000.0) * sr)

    dev = choose_input_device()
    info = sd.query_devices(dev)
    ch = choose_channel(info["max_input_channels"])

    silence_thresh = float(args.threshold * args.silence_ratio)
    min_silence = int((args.min_silence_ms / 1000.0) * sr)

    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else None

    print(f"Model sample rate:  {sr} Hz")
    print(f"Attack samples:     {int(state['attack_samples'])} ({state['attack_samples']/sr:.3f}s)")
    print(f"Capture window:     {capture_samples} ({capture_samples/sr:.3f}s)")
    print(f"Sweep step:         {sweep_step} samples ({args.sweep_step_ms:.1f}ms)")
    print(f"Trigger threshold:  {args.threshold}")
    print(f"Silence threshold:  {silence_thresh} (ratio {args.silence_ratio})")
    print(f"Min silence:        {min_silence} samples ({args.min_silence_ms:.1f}ms)")
    if save_dir:
        print(f"Debug save dir:     {save_dir}")
    print("Realtime inference active. Ctrl+C to exit.")

    ring = np.zeros(capture_samples, dtype=np.float32)
    mode = "armed"
    silence_ctr = 0

    def callback(indata, frames, time, status):
        nonlocal ring, mode, silence_ctr

        if status:
            print(f"Audio status: {status}", flush=True)

        mono = indata[:, ch].astype(np.float32)

        # update ring
        n = len(mono)
        if n >= capture_samples:
            ring = mono[-capture_samples:].copy()
        else:
            ring = np.roll(ring, -n)
            ring[-n:] = mono

        rms = float(np.sqrt(np.mean(mono ** 2))) if mono.size else 0.0

        if mode == "armed" and rms >= args.threshold:
            buf = ring.copy()

            midi, name, conf, offset, chosen_window = sweep_inference(
                buf,
                state,
                sweep_step,
                args.confidence,
            )

            offset_sec = offset / sr if offset >= 0 else -1.0
            print(
                f"Detected: {name} (MIDI {midi}) "
                f"conf={conf:.3f} offset={offset_sec:.3f}s "
                f"trigger_rms={rms:.4f}"
            )

            # Compute top-k on the chosen window (for debug)
            try:
                spec = waveform_to_logmel(
                    chosen_window,
                    mel=state["mel"],
                    db=state["db"],
                    num_frames=state["num_frames"],
                ).to(state["device"], dtype=torch.float32)
                topk = compute_topk(state["model"], spec, state["idx_to_midi"], k=args.topk)
                topk_str = " | ".join([f"{nm}:{p:.3f}" for _, nm, p in topk])
                print(f"Top-{len(topk)}: {topk_str}")
            except Exception as e:
                topk = []
                print(f"Top-k debug failed: {e}")

            # Save artifacts if requested
            if save_dir:
                try:
                    event_dir = make_event_dir(save_dir)
                    save_wav_mono(event_dir / "capture.wav", buf, sr)
                    save_wav_mono(event_dir / "window.wav", chosen_window, sr)

                    # meta
                    meta_lines = [
                        f"checkpoint={ckpt_path}",
                        f"device={state['device']}",
                        f"sample_rate={sr}",
                        f"attack_samples={state['attack_samples']}",
                        f"capture_samples={capture_samples}",
                        f"sweep_step_samples={sweep_step}",
                        f"confidence_target={args.confidence}",
                        f"trigger_threshold={args.threshold}",
                        f"trigger_rms={rms}",
                        f"silence_ratio={args.silence_ratio}",
                        f"silence_threshold={silence_thresh}",
                        f"min_silence_ms={args.min_silence_ms}",
                        f"min_silence_samples={min_silence}",
                        f"pred_midi={midi}",
                        f"pred_name={name}",
                        f"pred_conf={conf}",
                        f"offset_samples={offset}",
                        f"offset_sec={offset_sec}",
                    ]
                    (event_dir / "meta.txt").write_text("\n".join(meta_lines) + "\n")

                    # topk
                    if topk:
                        topk_lines = [f"{m}\t{nm}\t{p:.6f}" for (m, nm, p) in topk]
                        (event_dir / "topk.txt").write_text("\n".join(topk_lines) + "\n")
                    else:
                        (event_dir / "topk.txt").write_text("(topk unavailable)\n")

                    print(f"Saved debug artifacts: {event_dir}")
                except Exception as e:
                    print(f"Debug save failed: {e}")

            mode = "waiting"
            silence_ctr = 0

        elif mode == "waiting":
            if rms <= silence_thresh:
                silence_ctr += frames
            else:
                silence_ctr = 0

            if silence_ctr >= min_silence:
                mode = "armed"
                silence_ctr = 0
                print("Re-armed")

    with sd.InputStream(
        device=dev,
        channels=info["max_input_channels"],
        samplerate=sr,
        blocksize=args.block_size,
        dtype="float32",
        callback=callback,
    ):
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("Stopped.")


if __name__ == "__main__":
    main()
