import time
import numpy as np
import sounddevice as sd
import torch
import librosa

from attack_model import AttackPositionNet
from fusion_net import FusionNet
from pitch_yin import extract_pitch_yin

SAMPLE_RATE = 48000
ATTACK_FRAMES = 32
HOP_LENGTH = 256
RMS_ON = 0.012
RMS_OFF = 0.005

ATTACK_MODEL_PATH = "../models/attack_model.pt"
FUSION_MODEL_PATH = "../models/fusion_model.pt"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

attack_model = AttackPositionNet().to(device)
attack_model.load_state_dict(torch.load(ATTACK_MODEL_PATH, map_location=device))
attack_model.eval()

fusion_model = FusionNet().to(device)
fusion_model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=device))
fusion_model.eval()

def waveform_to_logmel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=1024, hop_length=HOP_LENGTH, n_mels=64)
    logmel = librosa.power_to_db(mel, ref=np.max)
    return logmel[:, :ATTACK_FRAMES].astype(np.float32)

def select_device():
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            print(f"{i}: {d['name']}")
    dev = int(input("Select input device: "))
    ch = int(input("Select input channel (1-based): ")) - 1
    return dev, ch

dev, ch = select_device()
sd.default.device = dev

ring = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
wp = 0
state = "ARMED"

def callback(indata, frames, t, status):
    global wp, state

    y = indata[:, ch]

    n = len(y)
    ring[wp:wp+n] = y
    wp = (wp + n) % ring.size

    rms = np.sqrt(np.mean(y ** 2))

    if state == "ARMED" and rms > RMS_ON:
        attack = ring[-int(ATTACK_FRAMES * HOP_LENGTH):]
        logmel = waveform_to_logmel(attack)
        x = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = attack_model(x, return_embedding=True)
            sustain = ring[-int(0.25 * SAMPLE_RATE):]
            _, midi = extract_pitch_yin(sustain)
            logits = fusion_model(emb, torch.tensor([midi]).to(device))
            pos = int(torch.argmax(logits))

        string = pos // 8 + 1
        fret = pos % 8

        print(f"STRING {string}  FRET {fret}  MIDI {midi}")
        state = "WAIT"

    elif state == "WAIT" and rms < RMS_OFF:
        state = "ARMED"

with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=callback):
    while True:
        time.sleep(0.1)
