#!/usr/bin/env python3

import argparse
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

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
- sweep over attack windows inside captured segment to find best confidence window
- mel+db preprocessing on CPU (matches training transforms; avoids MPS FFT limitations)
- WAV saving: raw captured audio + chosen inference window
- NEW:
  - print top-5 candidates for the chosen window
  - enforce --margin (top1_conf - top2_conf must be >= margin)
  - if margin fails AND top-2 are adjacent semitone, run pitch tie-break
  - print "Top-2 approach" results per trigger
"""
"""
Usage:
cd ~/code/fretListener

python tools/realtime_note_inference.py \
  --checkpoint artifacts/note_20251211_164726/note_cnn_best.pt \
  --trigger-threshold 0.01 \
  --pre-ms 50 \
  --post-ms 500 \
  --silence-threshold 0.006 \
  --min-silence-ms 80 \
  --confidence 0.85 \
  --sweep-step-ms 5 \
  --block-size 512 \
  --save-dir artifacts/captures

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

    # idx_to_midi may be dict with int keys or strings depending on save/load
    idx_to_midi = ckpt["idx_to_midi"]
    if isinstance(idx_to_midi, dict):
        idx_to_midi_norm = {int(k): int(v) for k, v in idx_to_midi.items()}
    else:
        idx_to_midi_norm = idx_to_midi

    return {
        "model": model,
        "model_device": model_device,
        "mel": mel,
        "db": db,
        "idx_to_midi": idx_to_midi_norm,
        "sample_rate": int(ckpt["sample_rate"]),
        "attack_samples": int(ckpt["attack_samples"]),
        "num_frames": int(mel_params["num_frames"]),
    }


# -----------------------------
# Preprocessing (CPU, training-identical)
# -----------------------------

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
    wav = wav - float(np.mean(wav))  # DC removal

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


# -----------------------------
# Model inference helpers
# -----------------------------

def _peak_normalize(window: np.ndarray) -> np.ndarray:
    w = window.astype(np.float32)
    peak = float(np.max(np.abs(w))) if w.size else 0.0
    if peak > 0:
        w = (w / peak) * 0.95
    return w


def predict_topk_window(
    state: Dict[str, Any],
    window_1d: np.ndarray,
    topk: int,
    peak_normalize: bool = True,
) -> Tuple[List[Tuple[int, str, float]], torch.Tensor]:
    """
    Returns:
      candidates: list of (midi, name, prob) sorted desc by prob, length=min(topk, num_classes)
      probs: full probability tensor [1, num_classes] (on CPU)
    """
    w = _peak_normalize(window_1d) if peak_normalize else window_1d.astype(np.float32)

    spec = waveform_to_logmel_cpu(
        wav_1d=w,
        mel=state["mel"],
        db=state["db"],
        num_frames=state["num_frames"],
    )

    spec = spec.to(state["model_device"], dtype=torch.float32)
    with torch.no_grad():
        logits = state["model"](spec)
        probs = torch.softmax(logits, dim=1)

    probs_cpu = probs.detach().to("cpu")
    k = min(int(topk), probs_cpu.shape[1])
    vals, idxs = torch.topk(probs_cpu, k=k, dim=1)

    out: List[Tuple[int, str, float]] = []
    for p, idx in zip(vals[0].tolist(), idxs[0].tolist()):
        midi = int(state["idx_to_midi"][int(idx)])
        out.append((midi, midi_to_name(midi), float(p)))

    return out, probs_cpu


# -----------------------------
# Pitch tie-break (adjacent semitone)
# -----------------------------

def estimate_f0_autocorr_hz(
    x: np.ndarray,
    sr: int,
    fmin: float = 70.0,
    fmax: float = 1200.0,
) -> Optional[float]:
    """
    Lightweight autocorrelation f0 estimate.
    Returns Hz or None.
    """
    if x.size < int(sr * 0.02):  # require ~20ms
        return None

    y = x.astype(np.float32)
    y = y - float(np.mean(y))
    y = _peak_normalize(y)

    # window to reduce edge effects
    win = np.hanning(y.size).astype(np.float32)
    y = y * win

    # autocorr (biased ok for tie-break)
    r = np.correlate(y, y, mode="full")[y.size - 1:]

    # ignore lag 0; find peak in plausible lag range
    lag_min = int(sr / fmax)
    lag_max = int(sr / fmin)
    if lag_max <= lag_min + 2 or lag_max >= r.size:
        return None

    seg = r[lag_min:lag_max]
    if seg.size == 0:
        return None

    i = int(np.argmax(seg))
    best_lag = lag_min + i
    if best_lag <= 0:
        return None

    # reject if correlation peak too weak
    peak = float(seg[i])
    if peak <= 0:
        return None

    return float(sr / best_lag)


def midi_to_freq_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def pitch_tiebreak_adjacent(
    window: np.ndarray,
    sr: int,
    midi_a: int,
    midi_b: int,
) -> Optional[int]:
    """
    If f0 estimate is valid, pick whichever MIDI is closer in Hz.
    Returns chosen midi, or None if no valid pitch estimate.
    """
    f0 = estimate_f0_autocorr_hz(window, sr=sr)
    if f0 is None:
        return None

    fa = midi_to_freq_hz(midi_a)
    fb = midi_to_freq_hz(midi_b)

    da = abs(f0 - fa)
    db = abs(f0 - fb)

    return midi_a if da <= db else midi_b


# -----------------------------
# Sweep inference
# -----------------------------

def sweep_best_window(
    captured: np.ndarray,
    state: Dict[str, Any],
    sweep_step_samples: int,
    topk: int,
) -> Tuple[int, np.ndarray, List[Tuple[int, str, float]], int]:
    """
    Sweep and select the window with highest top-1 confidence.

    Returns:
      best_start (int),
      best_window (np.ndarray attack-sized),
      best_topk (list[(midi,name,prob)]),
      best_peak (int)  [not used; kept signature stable if desired later]
    """
    attack = state["attack_samples"]
    sr = state["sample_rate"]

    if captured.shape[0] < attack:
        padded = np.zeros(attack, dtype=np.float32)
        padded[:captured.shape[0]] = captured.astype(np.float32)
        captured = padded

    best_conf = -1.0
    best_start = 0
    best_window = captured[:attack].copy()
    best_topk = []

    last_start = max(0, captured.shape[0] - attack)
    step = max(1, int(sweep_step_samples))

    for start in range(0, last_start + 1, step):
        window = captured[start:start + attack]

        topk_list, _ = predict_topk_window(
            state=state,
            window_1d=window,
            topk=topk,
            peak_normalize=True,
        )

        if not topk_list:
            continue

        conf = topk_list[0][2]
        if conf > best_conf:
            best_conf = conf
            best_start = start
            best_window = window.copy()
            best_topk = topk_list

    return best_start, best_window, best_topk, 0


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
# Decision logic: margin + adjacent pitch tie-break
# -----------------------------

def decide_with_margin_and_pitch(
    best_window: np.ndarray,
    best_topk: List[Tuple[int, str, float]],
    sr: int,
    margin: float,
) -> Tuple[int, str, float, str]:
    """
    Enforces margin; if margin fails AND top-2 are adjacent semitone, runs pitch tie-break.
    Returns: (midi, name, conf, decision_reason)
    """
    if not best_topk:
        return -1, "UNKNOWN", 0.0, "no_candidates"

    m1, n1, p1 = best_topk[0]
    if len(best_topk) < 2:
        return m1, n1, p1, "only_one_candidate"

    m2, n2, p2 = best_topk[1]
    gap = p1 - p2

    # Pass margin => accept top1
    if gap >= margin:
        return m1, n1, p1, f"margin_ok(gap={gap:.3f}>=margin={margin:.3f})"

    # Fail margin
    if abs(m1 - m2) == 1:
        chosen = pitch_tiebreak_adjacent(best_window, sr=sr, midi_a=m1, midi_b=m2)
        if chosen is not None:
            if chosen == m1:
                return m1, n1, p1, f"adjacent_pitch_tiebreak->top1(gap={gap:.3f}<margin={margin:.3f})"
            return m2, n2, p2, f"adjacent_pitch_tiebreak->top2(gap={gap:.3f}<margin={margin:.3f})"

        # No pitch estimate => keep top1 but mark ambiguous
        return m1, n1, p1, f"adjacent_no_pitch_estimate_keep_top1(gap={gap:.3f}<margin={margin:.3f})"

    # Not adjacent => keep top1 but mark ambiguous (enforced margin cannot resolve)
    return m1, n1, p1, f"margin_fail_non_adjacent_keep_top1(gap={gap:.3f}<margin={margin:.3f})"


# -----------------------------
# Main
# -----------------------------

def main():
    p = argparse.ArgumentParser(description="Realtime inference with matched capture path + margin + pitch tie-break.")
    p.add_argument("--checkpoint", required=True, type=str)

    # Trigger / capture
    p.add_argument("--trigger-threshold", type=float, default=0.01,
                   help="Peak amplitude threshold to trigger capture (collector-style).")
    p.add_argument("--pre-ms", type=float, default=50.0,
                   help="Pre-trigger audio to keep (ms). Default 50ms.")
    p.add_argument("--post-ms", type=float, default=500.0,
                   help="Post-trigger capture duration (ms). Default 500ms.")
    p.add_argument("--max-wait-sec", type=float, default=10.0,
                   help="Max time to wait for a trigger before printing timeout and re-arming.")
    p.add_argument("--block-size", type=int, default=512,
                   help="InputStream blocksize. Default 512.")

    # Re-arm control (peak-based)
    p.add_argument("--silence-threshold", type=float, default=0.006,
                   help="Peak amplitude below this counts as silence for re-arming.")
    p.add_argument("--min-silence-ms", type=float, default=80.0,
                   help="How long signal must stay below silence-threshold before re-arming.")

    # Sweep / decision
    p.add_argument("--sweep-step-ms", type=float, default=5.0,
                   help="Window sweep step size (ms). Default 5ms.")
    p.add_argument("--topk", type=int, default=5,
                   help="How many candidates to print per trigger. Default 5.")
    p.add_argument("--margin", type=float, default=0.10,
                   help="Required top1-top2 prob gap; if below and adjacent semitone, run pitch tie-break.")

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
    print(f"Capture: pre={pre_samples} ({args.pre_ms:.1f}ms) + post={post_samples} ({args.post_ms:.1f}ms) "
          f"= {capture_total} samples ({capture_total/sr:.3f}s)")
    print(f"Sweep step: {sweep_step_samples} samples ({args.sweep_step_ms:.1f}ms)")
    print(f"Trigger threshold (peak): {args.trigger_threshold}")
    print(f"Silence threshold (peak): {args.silence_threshold} for {min_silence_samples} samples ({args.min_silence_ms:.1f}ms)")
    print(f"Top-k print: {args.topk}")
    print(f"Margin: {args.margin:.3f}")
    if args.save_dir:
        print(f"WAV saving enabled: {args.save_dir}")

    # Choose device/channel
    dev = choose_input_device()
    dev_info = sd.query_devices(dev, "input")
    print(f"Selected device: {dev} - {dev_info['name']}")
    max_channels = int(dev_info["max_input_channels"])
    ch = choose_channel(max_channels)
    print(f"Selected channel: {ch + 1} (0-based index {ch})")

    # State machine
    mode = "armed"  # armed -> capturing -> waiting_silence
    timeout_ctr = 0
    silence_ctr = 0

    # Pre-buffer (collector-style circular) stores ALL channels
    pre_buffer = np.zeros((pre_samples, max_channels), dtype=np.float32) if pre_samples > 0 else None
    pre_idx = 0

    # Capture buffers
    recording_parts: List[np.ndarray] = []
    recorded_post = 0

    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else None

    def reset_capture_state():
        nonlocal recording_parts, recorded_post, timeout_ctr, pre_idx
        recording_parts = []
        recorded_post = 0
        timeout_ctr = 0
        pre_idx = 0

    reset_capture_state()

    def _prebuf_write(block: np.ndarray):
        """
        Writes the last N frames of the block into the circular pre-buffer.
        Fixes a common regression: when frames > pre_samples, only the *latest* samples should remain.
        """
        nonlocal pre_idx, pre_buffer
        if pre_buffer is None or pre_samples <= 0:
            return

        frames = block.shape[0]
        if frames <= 0:
            return

        # keep only last pre_samples frames
        if frames > pre_samples:
            block = block[-pre_samples:]
            frames = block.shape[0]

        # write with wrap
        end = pre_idx + frames
        if end <= pre_samples:
            pre_buffer[pre_idx:end, :] = block.astype(np.float32, copy=False)
        else:
            first = pre_samples - pre_idx
            pre_buffer[pre_idx:, :] = block[:first].astype(np.float32, copy=False)
            remaining = frames - first
            pre_buffer[:remaining, :] = block[first:].astype(np.float32, copy=False)

        pre_idx = (pre_idx + frames) % pre_samples

    def _prebuf_read_channel() -> np.ndarray:
        """
        Returns chronological pre-buffer for selected channel.
        """
        if pre_buffer is None or pre_samples <= 0:
            return np.zeros((0,), dtype=np.float32)

        if pre_idx == 0:
            return pre_buffer[:, ch].astype(np.float32, copy=True)

        return np.concatenate([
            pre_buffer[pre_idx:, ch],
            pre_buffer[:pre_idx, ch],
        ]).astype(np.float32, copy=True)

    def callback(indata, frames, time_info, status):
        nonlocal mode, timeout_ctr, silence_ctr
        nonlocal recording_parts, recorded_post

        if status:
            if "overflow" in str(status).lower():
                print("Audio overflow detected", flush=True)

        # selected channel extraction (collector-style)
        chan = indata[:, ch].astype(np.float32, copy=False)
        peak = float(np.max(np.abs(chan))) if chan.size else 0.0

        if mode == "armed":
            timeout_ctr += frames

            # update pre-buffer with ALL channels
            if pre_buffer is not None:
                _prebuf_write(indata)

            if peak >= args.trigger_threshold:
                pre_audio = _prebuf_read_channel()
                # capture includes current block immediately after trigger as well
                recording_parts = [pre_audio, chan.astype(np.float32, copy=True)]
                recorded_post = int(chan.shape[0])

                mode = "capturing"
                timeout_ctr = 0
                print("Triggered: capturing...", flush=True)

            elif timeout_ctr >= max_wait_samples:
                print("Timeout: no trigger detected. Re-armed.", flush=True)
                timeout_ctr = 0

        elif mode == "capturing":
            recording_parts.append(chan.astype(np.float32, copy=True))
            recorded_post += int(chan.shape[0])

            if recorded_post >= post_samples:
                cap = np.concatenate(recording_parts).astype(np.float32, copy=False)

                # trim/pad to exact capture_total
                if cap.shape[0] > capture_total:
                    cap = cap[:capture_total]
                elif cap.shape[0] < capture_total:
                    cap = np.pad(cap, (0, capture_total - cap.shape[0]), mode="constant")

                # pick best window by sweep (top1 confidence)
                best_start, best_window, best_topk, _ = sweep_best_window(
                    captured=cap,
                    state=state,
                    sweep_step_samples=sweep_step_samples,
                    topk=args.topk,
                )

                # Top-2 approach output
                # Approach A: raw top1 on best window
                if best_topk:
                    a_midi, a_name, a_conf = best_topk[0]
                else:
                    a_midi, a_name, a_conf = -1, "UNKNOWN", 0.0

                # Approach B: margin enforced + adjacent pitch tie-break
                b_midi, b_name, b_conf, reason = decide_with_margin_and_pitch(
                    best_window=best_window,
                    best_topk=best_topk,
                    sr=sr,
                    margin=args.margin,
                )

                # Print top-5
                print("Top candidates (best window):", flush=True)
                for i, (m, n, p_) in enumerate(best_topk[:args.topk], start=1):
                    print(f"  {i:>2d}) {n:<4s} MIDI {m:<3d}  prob={p_:.3f}", flush=True)

                # Print approaches + final
                print(
                    f"Approach A (top1): {a_name} (MIDI {a_midi}) prob={a_conf:.3f}  "
                    f"best_offset={best_start/sr:.3f}s",
                    flush=True,
                )
                print(
                    f"Approach B (margin+tiebreak): {b_name} (MIDI {b_midi}) prob={b_conf:.3f}  "
                    f"reason={reason}",
                    flush=True,
                )

                # Save WAVs (raw + chosen window) if requested
                if save_dir is not None:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    raw_path = save_dir / f"cap_{ts}_raw.wav"
                    best_path = save_dir / f"cap_{ts}_best_{b_name}_prob{b_conf:.2f}.wav"
                    save_wav_int16(raw_path, sr, cap)

                    # chosen window must be attack-sized
                    win = best_window
                    if win.shape[0] < attack:
                        win = np.pad(win, (0, attack - win.shape[0]), mode="constant")
                    else:
                        win = win[:attack]
                    save_wav_int16(best_path, sr, win)

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
    print("  3) Sweep windows; pick best-confidence window")
    print("  4) Print top-k; apply margin; adjacent pitch tie-break if needed")
    print("  5) Wait for silence -> re-arm")
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
