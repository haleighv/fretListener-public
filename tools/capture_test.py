#!/usr/bin/env python3
"""
tools/capture_chop_test.py

Isolated capture + chop test (no GUI).
- Captures from a fixed input device + channel
- Writes raw capture WAV
- Runs the exact same onset detection + chopping path as the main tool
- Writes clips into a class-style directory

Dependencies:
  pip install sounddevice soundfile numpy librosa

Run:
  python tools/capture_chop_test.py
Stop:
  Ctrl+C
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa


# ============================
# CAPTURE / ONSET CONFIG
# ============================

TARGET_SR = 16_000

PRE_ROLL_MS = 30
CLIP_DURATION_MS = 600

PRE_ROLL_SAMPLES = int(TARGET_SR * (PRE_ROLL_MS / 1000.0))
CLIP_SAMPLES = int(TARGET_SR * (CLIP_DURATION_MS / 1000.0))

ONSET_RMS_WINDOW = 2048
ONSET_THRESHOLD = 0.010
MIN_GAP_SEC = 0.35
MIN_GAP_SAMPLES = int(TARGET_SR * MIN_GAP_SEC)

ANALYSIS_HOP = 512

# Auto-stop after silence
SILENCE_THRESHOLD = 0.005  # RMS below this = silence
SILENCE_DURATION_SEC = 1.0  # Stop after this much silence
SILENCE_SAMPLES = int(TARGET_SR * SILENCE_DURATION_SEC)

# Pre-roll noise gate - fade in first N ms to eliminate pop
PREROLL_FADE_MS = 5
PREROLL_FADE_SAMPLES = int(TARGET_SR * (PREROLL_FADE_MS / 1000.0))


# ============================
# ISOLATED TEST SETTINGS
# ============================

BLOCKSIZE = ANALYSIS_HOP

DEVICE_INDEX = 3
INPUT_CHANNEL_INDEX = 4  # 0-based index (input #5 == channel 6). Change to 4 if you mean 1-based.

ENABLE_MONITOR = False
OUTPUT_DEVICE_INDEX: Optional[int] = None
OUTPUT_CHANNELS = 2
OUTPUT_LATENCY = "low"

OUT_ROOT = Path("capture_chop_test_out")
RAW_WAV = OUT_ROOT / "raw_capture.wav"

# Pretend target for directory/filenames (does not affect processing)
TEST_STRING = 6
TEST_FRET = 0
TEST_SESSION = "A"


def compute_rms(x: np.ndarray) -> float:
    """Compute RMS of audio signal."""
    return float(np.sqrt(np.mean(x ** 2))) if x.size > 0 else 0.0


# ============================
# ONSET DETECTION & CHOPPING
# ============================

def detect_onsets(audio: np.ndarray) -> List[int]:
    """
    Detect note onsets using RMS with rising-edge detection.

    Returns list of sample indices where onsets occur.
    """
    total_samples = len(audio)
    rms_vals = []
    onset_indices = []

    for i in range(0, total_samples - ONSET_RMS_WINDOW, ANALYSIS_HOP):
        frame = audio[i:i + ONSET_RMS_WINDOW]
        rms = compute_rms(frame)
        rms_vals.append(rms)

        if rms > ONSET_THRESHOLD:
            if len(rms_vals) < 2:
                onset_indices.append(i)
            else:
                prev_rms = rms_vals[-2]
                if rms > prev_rms * 1.5:
                    onset_indices.append(i)

    # Hard inter-onset suppression
    clean_onsets = []
    last = -10**12
    for idx in onset_indices:
        if idx - last >= MIN_GAP_SAMPLES:
            clean_onsets.append(idx)
            last = idx

    return clean_onsets


def chop_recording(
    audio: np.ndarray,
    output_dir: Path,
    string_num: int,
    fret: int,
    session: str = "A"
) -> int:
    """
    Chop a recording into individual clips based on onset detection.
    Applies high-pass filter and fade-in/out to eliminate pops/artifacts.

    Returns number of clips saved.
    """
    # Remove DC offset from entire recording first
    audio = audio - np.mean(audio)

    # Apply simple high-pass filter to remove rumble (cutoff ~20Hz)
    alpha = 0.995
    filtered = np.zeros_like(audio)
    filtered[0] = audio[0]
    for i in range(1, len(audio)):
        filtered[i] = alpha * (filtered[i-1] + audio[i] - audio[i-1])
    audio = filtered

    onsets = detect_onsets(audio)
    total_samples = len(audio)

    # Create smooth fade windows (raised cosine)
    fade_ms = 3
    fade_samples = int(TARGET_SR * fade_ms / 1000)
    if fade_samples > 0:
        fade_in = 0.5 * (1 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))
        fade_out = 0.5 * (1 + np.cos(np.pi * np.arange(fade_samples) / fade_samples))
        fade_in = fade_in.astype(np.float32)
        fade_out = fade_out.astype(np.float32)
    else:
        fade_in = None
        fade_out = None

    clips_saved = 0
    for take_idx, onset_idx in enumerate(onsets, start=1):
        start = max(0, onset_idx - PRE_ROLL_SAMPLES)
        end = start + CLIP_SAMPLES

        if end > total_samples:
            continue

        clip = audio[start:end].copy()

        # Apply fade-in at start
        if fade_in is not None and len(clip) >= len(fade_in):
            clip[:len(fade_in)] *= fade_in

        # Apply fade-out at end
        if fade_out is not None and len(clip) >= len(fade_out):
            clip[-len(fade_out):] *= fade_out

        # Naming: s{STRING}_f{FRET}_t{TAKE}_sess{SESSION}.wav
        out_name = f"s{string_num}_f{fret}_t{take_idx:02d}_sess{session}.wav"
        out_path = output_dir / out_name

        sf.write(str(out_path), clip, TARGET_SR)
        clips_saved += 1

    return clips_saved


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    device_info = sd.query_devices(DEVICE_INDEX)
    max_in = int(device_info.get("max_input_channels", 0))
    if max_in <= 0:
        raise RuntimeError(f"Device {DEVICE_INDEX} has no input channels")

    if INPUT_CHANNEL_INDEX < 0 or INPUT_CHANNEL_INDEX >= max_in:
        raise RuntimeError(
            f"INPUT_CHANNEL_INDEX={INPUT_CHANNEL_INDEX} out of range for device {DEVICE_INDEX} (max_input_channels={max_in})"
        )

    print(f"Input device #{DEVICE_INDEX}: {device_info.get('name')}")
    print(f"Input channel index: {INPUT_CHANNEL_INDEX} (0-based)")
    print(f"Sample rate: {TARGET_SR}  blocksize: {BLOCKSIZE}")
    print(f"Monitoring: {'ON' if ENABLE_MONITOR else 'OFF'}")
    print(f"Output root: {OUT_ROOT.resolve()}")
    print("Recording... Ctrl+C to stop")

    buffers: List[np.ndarray] = []
    monitor_audio: Optional[np.ndarray] = None

    status_counts = {}
    total_frames = 0

    silence_samples = 0
    onset_count = 0
    last_rms = 0.0
    samples_since_onset = MIN_GAP_SAMPLES

    def audio_callback(indata, frames, time_info, status):
        nonlocal monitor_audio, total_frames
        nonlocal silence_samples, onset_count, last_rms, samples_since_onset

        if status:
            key = str(status)
            status_counts[key] = status_counts.get(key, 0) + 1

        if indata.ndim == 2:
            mono = indata[:, INPUT_CHANNEL_INDEX].astype(np.float32, copy=True)
        else:
            mono = indata.astype(np.float32, copy=True)

        buffers.append(mono)
        total_frames += int(frames)

        if ENABLE_MONITOR:
            monitor_audio = mono

        # Match the main tool’s “enough takes then silence => stop” behavior
        rms = compute_rms(mono)
        samples_since_onset += len(mono)

        if rms > ONSET_THRESHOLD:
            silence_samples = 0
            if samples_since_onset >= MIN_GAP_SAMPLES:
                if last_rms <= 0 or rms > last_rms * 1.5:
                    onset_count += 1
                    samples_since_onset = 0
        elif rms < SILENCE_THRESHOLD:
            if onset_count >= 10:  # mimic default takes_per_note behavior for stopping
                silence_samples += len(mono)

        last_rms = rms

        if silence_samples >= SILENCE_SAMPLES:
            raise sd.CallbackStop()

    def output_callback(outdata, frames, time_info, status):
        nonlocal monitor_audio
        if not ENABLE_MONITOR or monitor_audio is None:
            outdata.fill(0)
            return
        mono = monitor_audio
        n = min(len(mono), frames)
        outdata[:n, 0] = mono[:n]
        outdata[:n, 1] = mono[:n]
        if n < frames:
            outdata[n:, :] = 0

    in_stream = sd.InputStream(
        device=DEVICE_INDEX,
        channels=max_in,
        samplerate=TARGET_SR,
        blocksize=BLOCKSIZE,
        dtype="float32",
        callback=audio_callback,
    )

    out_stream = None
    if ENABLE_MONITOR:
        out_stream = sd.OutputStream(
            device=OUTPUT_DEVICE_INDEX,
            samplerate=TARGET_SR,
            blocksize=BLOCKSIZE,
            channels=OUTPUT_CHANNELS,
            dtype="float32",
            callback=output_callback,
            latency=OUTPUT_LATENCY,
        )

    try:
        if out_stream is not None:
            out_stream.start()
        in_stream.start()
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            in_stream.stop()
            in_stream.close()
        except Exception:
            pass
        if out_stream is not None:
            try:
                out_stream.stop()
                out_stream.close()
            except Exception:
                pass

    if not buffers:
        print("No audio captured.")
        return

    full_audio = np.concatenate(buffers)
    duration = len(full_audio) / float(TARGET_SR)

    sf.write(str(RAW_WAV), full_audio, TARGET_SR)

    print("")
    print("Capture complete.")
    print(f"Raw duration: {duration:0.2f}s  samples: {len(full_audio)}")
    if status_counts:
        print("Status events:")
        for k, v in sorted(status_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {v:6d}  {k}")
    else:
        print("Status events: none")

    class_dir = OUT_ROOT / "organized" / f"s{TEST_STRING}_f{TEST_FRET}"
    class_dir.mkdir(parents=True, exist_ok=True)

    clips_saved = chop_recording(
        full_audio,
        class_dir,
        TEST_STRING,
        TEST_FRET,
        TEST_SESSION
    )

    print("")
    print(f"Saved raw: {RAW_WAV}")
    print(f"Saved clips: {clips_saved} -> {class_dir}")


if __name__ == "__main__":
    main()
