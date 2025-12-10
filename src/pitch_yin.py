import numpy as np

def yin(x, sr, fmin=55.0, fmax=2000.0, frame_size=2048, threshold=0.1):
    x = x.astype(np.float32)
    if x.ndim != 1:
        x = np.mean(x, axis=-1)

    if len(x) < frame_size:
        x = np.pad(x, (0, frame_size - len(x)))
    else:
        x = x[:frame_size]

    x -= np.mean(x)

    min_tau = int(sr / fmax)
    max_tau = int(sr / fmin)

    d = np.zeros(max_tau + 1)
    for tau in range(1, max_tau + 1):
        diff = x[:-tau] - x[tau:]
        d[tau] = np.sum(diff * diff)

    cmnd = np.zeros_like(d)
    running_sum = 0.0
    for tau in range(1, max_tau + 1):
        running_sum += d[tau]
        cmnd[tau] = d[tau] * tau / (running_sum + 1e-8)

    for tau in range(min_tau, max_tau):
        if cmnd[tau] < threshold:
            return sr / tau
    return None

def hz_to_midi(f):
    if f is None:
        return None
    return 69 + 12 * np.log2(f / 440.0)

def midi_to_note_name(midi):
    if midi is None:
        return None
    midi = int(round(midi))
    notes = ["C", "C#", "D", "D#", "E", "F",
             "F#", "G", "G#", "A", "A#", "B"]
    name = notes[midi % 12]
    octave = (midi // 12) - 1
    return f"{name}{octave}"
