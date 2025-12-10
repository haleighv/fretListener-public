import numpy as np
import librosa

SAMPLE_RATE = 48000
FMIN = 55.0
FMAX = 2000.0

def extract_pitch_yin(y):
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    f0 = librosa.yin(
        y,
        fmin=FMIN,
        fmax=FMAX,
        sr=SAMPLE_RATE,
        frame_length=2048,
        hop_length=256,
    )

    f0 = f0[np.isfinite(f0)]
    if len(f0) == 0:
        return None, None

    freq = np.median(f0)
    midi = int(np.round(librosa.hz_to_midi(freq)))

    return float(freq), int(midi)
