import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import librosa
import torch
import torch.nn.functional as F

from multihead_cnn import GuitarMultiHeadCNN
from labels import INDEX_TO_NOTE


# ============================
# MODEL / FEATURE CONFIG
# ============================

TARGET_SR = 16_000
TARGET_DURATION_SEC = 0.5
WINDOW_SAMPLES = int(TARGET_SR * TARGET_DURATION_SEC)

N_FFT = 512
HOP_LENGTH = 128
N_MELS = 64
EPS = 1e-10

CAPTURE_SR = 16_000
BLOCKSIZE = 1024

BUFFER_SECONDS = 3.0
BUFFER_SAMPLES = int(TARGET_SR * BUFFER_SECONDS)


# ============================
# ONSET DETECTION
# ============================

ONSET_WINDOW_SAMPLES = 2048
ONSET_THRESHOLD = 0.01
RESET_THRESHOLD = 0.005
RESET_SECONDS = 0.3
POLL_INTERVAL_SEC = 0.02


# ============================
# GUITAR PHYSICS
# ============================

STRING_OPEN_MIDI = {
    6: 40,  # E2
    5: 45,  # A2
    4: 50,  # D3
    3: 55,  # G3
    2: 59,  # B3
    1: 64,  # E4
}


# ============================
# MODEL LOADING
# ============================

def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(device):
    script_dir = Path(__file__).resolve().parent
    ckpt_path = script_dir / "pilot0_multihead_cnn.pt"

    model = GuitarMultiHeadCNN(
        n_mels=N_MELS,
        num_notes=12,
        num_strings=6,
        num_frets=6,
        hidden_dim=128,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ============================
# FEATURE PIPELINE
# ============================

def standardize_window(y, sr):
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    y, _ = librosa.effects.trim(y, top_db=40)

    peak = np.max(np.abs(y)) if y.size > 0 else 0.0
    if peak > 0:
        y = y / peak

    if y.size < WINDOW_SAMPLES:
        y = np.pad(y, (0, WINDOW_SAMPLES - y.size))
    elif y.size > WINDOW_SAMPLES:
        y = y[:WINDOW_SAMPLES]

    return y.astype(np.float32)


def window_to_logmel(y):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        center=True,
        power=2.0,
    )
    return np.log(mel + EPS).astype(np.float32)


# ============================
# DECODING
# ============================

def midi_to_note_octave(midi):
    note_idx = midi % 12
    octave = midi // 12 - 1
    return f"{INDEX_TO_NOTE[note_idx]}{octave}"


def decode_prediction(logits_note, logits_string, logits_fret):
    probs_note = F.softmax(logits_note, dim=1)[0]
    probs_string = F.softmax(logits_string, dim=1)[0]
    probs_fret = F.softmax(logits_fret, dim=1)[0]

    note_id = int(torch.argmax(probs_note))
    string_id = int(torch.argmax(probs_string)) + 1
    fret_id = int(torch.argmax(probs_fret))

    open_midi = STRING_OPEN_MIDI[string_id]
    derived_note = midi_to_note_octave(open_midi + fret_id)

    return {
        "string": string_id,
        "fret": fret_id,
        "note": INDEX_TO_NOTE[note_id],
        "derived_note": derived_note,
        "conf_note": float(probs_note[note_id]),
        "conf_string": float(probs_string[string_id - 1]),
        "conf_fret": float(probs_fret[fret_id]),
    }


# ============================
# UTIL
# ============================

def compute_rms(x):
    return float(np.sqrt(np.mean(x ** 2))) if x.size > 0 else 0.0


# ============================
# MAIN
# ============================

def main():
    print("\nAvailable audio devices:")
    print("-" * 40)

    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"{i}: {device['name']} (inputs: {device['max_input_channels']})")

    # ---- DEVICE SELECTION (MATCHES YOUR TOOL) ----
    selected_device = None
    while True:
        try:
            device_id = input("\nSelect input device number (or Enter for default): ").strip()
            if device_id == '':
                selected_device = sd.default.device[0] if sd.default.device else 0
                break
            device_id = int(device_id)
            if 0 <= device_id < len(devices) and devices[device_id]['max_input_channels'] > 0:
                sd.default.device = device_id
                selected_device = device_id
                break
            else:
                print("Invalid device number")
        except ValueError:
            print("Please enter a valid number")

    device_info = sd.query_devices(selected_device, 'input')
    max_channels = device_info['max_input_channels']

    input_channel = 1
    if max_channels > 1:
        print(f"\n{device_info['name']} has {max_channels} input channels.")
        while True:
            try:
                channel = input(f"Select input channel (1-{max_channels}): ").strip()
                channel = int(channel)
                if 1 <= channel <= max_channels:
                    input_channel = channel
                    break
                else:
                    print(f"Please enter a number between 1 and {max_channels}")
            except ValueError:
                print("Please enter a valid number")

    input_channel -= 1

    # ---- MODEL ----
    device = select_device()
    model = load_model(device)

    print("\n[realtime] Listening for plucks…")

    buffer = deque(maxlen=BUFFER_SAMPLES)
    armed = True
    last_silence_time = time.time()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)

        if input_channel < indata.shape[1]:
            mono = indata[:, input_channel].copy()
        else:
            mono = indata[:, 0].copy()

        for s in mono:
            buffer.append(float(s))

    try:
        with sd.InputStream(
            device=selected_device,
            channels=max_channels,
            samplerate=CAPTURE_SR,
            blocksize=BLOCKSIZE,
            callback=audio_callback,
        ):
            while True:
                time.sleep(POLL_INTERVAL_SEC)

                if len(buffer) < max(WINDOW_SAMPLES, ONSET_WINDOW_SAMPLES):
                    continue

                buf = np.array(buffer, dtype=np.float32)
                rms = compute_rms(buf[-ONSET_WINDOW_SAMPLES:])
                now = time.time()

                if armed and rms >= ONSET_THRESHOLD:
                    window = buf[-WINDOW_SAMPLES:]
                    y_std = standardize_window(window, CAPTURE_SR)
                    logmel = window_to_logmel(y_std)

                    feat = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = model(feat)
                        pred = decode_prediction(
                            outputs["note"],
                            outputs["string"],
                            outputs["fret"],
                        )

                    print(
                        f"ONSET → string {pred['string']} (p={pred['conf_string']:.2f}), "
                        f"fret {pred['fret']} (p={pred['conf_fret']:.2f}), "
                        f"note {pred['note']} (p={pred['conf_note']:.2f}), "
                        f"derived {pred['derived_note']}"
                    )

                    armed = False
                    last_silence_time = now

                elif not armed:
                    if rms < RESET_THRESHOLD:
                        if (now - last_silence_time) >= RESET_SECONDS:
                            armed = True
                            print("[realtime] Re-armed")

                    else:
                        last_silence_time = now

    except KeyboardInterrupt:
        print("\n[realtime] Stopped.")


if __name__ == "__main__":
    main()
