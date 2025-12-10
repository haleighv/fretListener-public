#!/usr/bin/env python3

import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import librosa

# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "src/pilot2_all_strings"
    / "pilot2_attack_position_48class.pt"
)

SAMPLE_RATE = 48000

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
FMIN = 55.0
FMAX = 2000.0

ATTACK_FRAMES = 32  # overridden by checkpoint

RMS_WINDOW_SEC = 0.05
RMS_THRESHOLD_ON = 0.012     # trigger
RMS_THRESHOLD_OFF = 0.005   # silence re-arm


# =========================================================
# MODEL (MATCHES TRAINING)
# =========================================================

class AttackPositionNet(nn.Module):
    def __init__(self, n_mels=64, num_pos=48):
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

        self.fc = nn.Linear(128, 128)
        self.head_pos = nn.Linear(128, num_pos)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc(x))
        x = self.head_pos(x)
        return x


def load_attack_position_model(device):
    global ATTACK_FRAMES

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state = checkpoint["model_state_dict"]
    ATTACK_FRAMES = int(checkpoint.get("max_attack_frames", ATTACK_FRAMES))

    print(f"[model] Using ATTACK_FRAMES={ATTACK_FRAMES}")

    model = AttackPositionNet(n_mels=N_MELS, num_pos=48).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


# =========================================================
# POSITION DECODING
# =========================================================

POSITION_LOOKUP = [(s, f) for s in range(6) for f in range(8)]

STANDARD_TUNING_MIDI = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]

def midi_to_name(midi):
    note = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return f"{note}{octave}"

def decode_position(pos_idx):
    s, f = POSITION_LOOKUP[pos_idx]
    midi = STANDARD_TUNING_MIDI[s] + f
    return s + 1, f, midi, midi_to_name(midi)


# =========================================================
# FEATURE EXTRACTION (MATCHES TRAINING)
# =========================================================

def waveform_to_logmel(y):
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )

    logmel = librosa.power_to_db(mel, ref=np.max)

    T = logmel.shape[1]
    if T < ATTACK_FRAMES:
        logmel = np.pad(logmel, ((0, 0), (0, ATTACK_FRAMES - T)))
    elif T > ATTACK_FRAMES:
        logmel = logmel[:, :ATTACK_FRAMES]

    return logmel.astype(np.float32)


def predict(model, device, y):
    logmel = waveform_to_logmel(y)
    x = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs))
        p = float(probs[idx])

    return (*decode_position(idx), p)


# =========================================================
# DEVICE + CHANNEL SELECTION
# =========================================================

def select_device_and_channel():
    print("\nAvailable input devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"{i}: {dev['name']}")

    device_id = int(input("\nSelect device #: "))
    dev = sd.query_devices(device_id, "input")

    channel_idx = 0
    if dev["max_input_channels"] > 1:
        channel_idx = int(input("Select input channel (1-based): ")) - 1

    return device_id, channel_idx


# =========================================================
# REALTIME STATE MACHINE LOOP
# =========================================================

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_attack_position_model(device)

    dev_id, ch_idx = select_device_and_channel()
    sd.default.device = dev_id

    rms_window = int(RMS_WINDOW_SEC * SAMPLE_RATE)
    attack_samples = ATTACK_FRAMES * HOP_LENGTH

    ring_len = rms_window + attack_samples
    ring = np.zeros(ring_len, dtype=np.float32)
    wp = 0

    state = "ARMED"   # ARMED → CAPTURED → WAIT_SILENCE

    print("\n[realtime] Listening for ATTACK → SILENCE cycles…")

    def callback(indata, frames, time_info, status):
        nonlocal ring, wp, state

        y = indata[:, ch_idx] if indata.ndim > 1 else indata[:, 0]

        if len(y) > ring_len:
            y = y[-ring_len:]

        n = len(y)
        end = wp + n

        if end <= ring_len:
            ring[wp:end] = y
        else:
            first = ring_len - wp
            ring[wp:] = y[:first]
            ring[:n - first] = y[first:]

        wp = (wp + n) % ring_len

        rms = np.sqrt(np.mean(ring[-rms_window:] ** 2))

        # ---------------- STATE MACHINE ---------------- #

        if state == "ARMED":
            if rms > RMS_THRESHOLD_ON:
                attack = ring[-attack_samples:]
                string, fret, midi, note, p = predict(model, device, attack)

                print(
                    f"ONSET → string {string}, fret {fret}, "
                    f"note {note}, MIDI {midi} (p={p:.2f})"
                )

                state = "WAIT_SILENCE"

        elif state == "WAIT_SILENCE":
            if rms < RMS_THRESHOLD_OFF:
                state = "ARMED"
                print("[realtime] Re-armed\n")

    try:
        with sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            callback=callback,
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping.")


if __name__ == "__main__":
    main()
