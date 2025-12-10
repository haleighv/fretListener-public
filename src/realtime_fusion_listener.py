import argparse
import numpy as np
import torch
import torchaudio
import sounddevice as sd
from collections import deque

from attack_model import AttackPositionNet
from fusion_net import FusionNet
from pitch_yin import yin, hz_to_midi, midi_to_note_name


# -----------------------------
# Device Handling (MPS-first)
# -----------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------
# RMS Utility
# -----------------------------

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


# -----------------------------
# Position Decode
# -----------------------------

def decode_pos(pos: int):
    string = pos // 8 + 1
    fret = pos % 8
    return string, fret


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-checkpoint", required=True)
    parser.add_argument("--fusion-checkpoint", required=True)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--attack-duration", type=float, default=0.19)
    parser.add_argument("--block-duration", type=float, default=0.02)

    # ---- USER-SELECTED INTERFACE + INPUT ----
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--input-channel", type=int, required=True)

    # ---- RMS GATE ----
    parser.add_argument("--rms-on", type=float, default=0.02)
    parser.add_argument("--rms-off", type=float, default=0.01)

    args = parser.parse_args()

    device = get_device()
    print("Using torch device:", device)

    # -----------------------------
    # Load Models
    # -----------------------------

    attack = AttackPositionNet().to(device)
    attack.load_state_dict(
        torch.load(args.attack_checkpoint, map_location=device)["model_state"]
    )
    attack.eval()

    fusion = FusionNet().to(device)
    fusion.load_state_dict(
        torch.load(args.fusion_checkpoint, map_location=device)["model_state"]
    )
    fusion.eval()

    # -----------------------------
    # Audio + Feature Transforms
    # -----------------------------

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
        f_min=55,
        f_max=2000,
    )
    db = torchaudio.transforms.AmplitudeToDB()

    attack_samples = int(args.sample_rate * args.attack_duration)
    block_size = int(args.sample_rate * args.block_duration)
    blocks_needed = int(np.ceil(attack_samples / block_size))

    # -----------------------------
    # State Machine
    # -----------------------------

    STATE_ARMED = 0
    STATE_CAPTURE = 1
    STATE_WAIT_SILENCE = 2

    state = STATE_ARMED
    ring = deque(maxlen=blocks_needed)

    # -----------------------------
    # Audio Callback
    # -----------------------------

    def audio_cb(indata, frames, time, status):
        if status:
            print("Audio status:", status)

        ch = args.input_channel
        if ch >= indata.shape[1]:
            return

        mono = indata[:, ch].copy()
        ring.append(mono)

    # -----------------------------
    # Device Listing
    # -----------------------------

    print("Available Audio Devices:")
    print(sd.query_devices())
    print("Using device index:", args.device_index)
    print("Using input channel:", args.input_channel)

    # -----------------------------
    # Stream
    # -----------------------------

    with sd.InputStream(
        device=args.device_index,
        channels=sd.query_devices(args.device_index)["max_input_channels"],
        samplerate=args.sample_rate,
        blocksize=block_size,
        callback=audio_cb,
    ):
        print("Listening (one-shot attack mode)...")

        while True:
            if len(ring) < ring.maxlen:
                continue

            audio = np.concatenate(list(ring), axis=0).astype(np.float32)

            level = rms(audio)

            # =============================
            # ARMED → CAPTURE
            # =============================
            if state == STATE_ARMED:
                if level > args.rms_on:
                    state = STATE_CAPTURE
                    continue

            # =============================
            # CAPTURE → INFER (ONE-SHOT)
            # =============================
            if state == STATE_CAPTURE:
                # ---- Trim attack window ----
                attack_audio = audio[:attack_samples]

                # ---- Log-Mel ----
                spec = db(mel(torch.tensor(attack_audio).unsqueeze(0)))
                spec = spec[:, :, :32]
                spec = spec.unsqueeze(0).to(device, dtype=torch.float32)

                # ---- Attack Embedding ----
                with torch.no_grad():
                    _, emb = attack(spec, return_embedding=True)

                emb = emb.to(dtype=torch.float32)

                # ---- Pitch (YIN) ----
                pitch_hz = yin(attack_audio, args.sample_rate)
                pitch_midi = hz_to_midi(pitch_hz)

                if pitch_midi is None:
                    pitch_val = -1.0
                else:
                    pitch_val = float(pitch_midi)

                pitch_tensor = torch.tensor(
                    [[pitch_val]],
                    dtype=torch.float32,
                    device=device,
                )

                # ---- Fusion ----
                fusion_in = torch.cat([emb, pitch_tensor], dim=1)

                with torch.no_grad():
                    logits = fusion(fusion_in)
                    pred = logits.argmax(1).item()
                    prob = torch.softmax(logits, dim=1)[0, pred].item()

                string, fret = decode_pos(pred)
                note = midi_to_note_name(pitch_midi)

                print(
                    f"String {string} | Fret {fret} | "
                    f"{note if note else 'Unknown'} | "
                    f"Conf {prob*100:.1f}%"
                )

                state = STATE_WAIT_SILENCE
                ring.clear()
                continue

            # =============================
            # WAIT SILENCE → REARM
            # =============================
            if state == STATE_WAIT_SILENCE:
                if level < args.rms_off:
                    state = STATE_ARMED
                    ring.clear()
                    continue


if __name__ == "__main__":
    main()
