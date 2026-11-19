#!/usr/bin/env python3
"""Spectrogram figure: the same pitch (D3) played in its three positions.

Computes log-mel spectrograms with the note model's training settings
(48 kHz, n_fft 1024, hop 128, 64 HTK mels, 55-3000 Hz, first 250 ms of each clip),
averaged over every take of each position, plus each position's average spectral profile.

Usage:  python docs/make_spectrograms.py --dataset ../fretListener/data/strat_mia_full_bridge_20
Writes docs/figures/spectrograms_{light,dark}.png. Needs the recordings (not in this repo).
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SR, N_FFT, HOP, N_MELS, FMIN, FMAX = 48000, 1024, 128, 64, 55.0, 3000.0
ATTACK = int(0.25 * SR)
OUT = Path(__file__).resolve().parent / "figures"

POSITIONS = [  # (label, file prefix): all D3, MIDI 50
    ("Low E string, fret 10", "s6_f10_"),
    ("A string, fret 5", "s5_f5_"),
    ("Open D string", "s4_f0_"),
]

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", grid="#e4e3df",
                  series=["#2a78d6", "#eb6834", "#1baf7a"], ramp=["#fcfcfb", "#9ec5f0", "#2a78d6", "#0d3a73"]),
    "dark": dict(surface="#1a1a19", text="#f0efec", text2="#c3c2b7", grid="#34332f",
                 series=["#3987e5", "#d95926", "#199e70"], ramp=["#1a1a19", "#16365e", "#3987e5", "#cfe3fa"]),
}


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def mel_filterbank():
    """HTK-scale triangular filters, no normalization (torchaudio MelSpectrogram defaults)."""
    freqs = np.linspace(0, SR / 2, N_FFT // 2 + 1)
    pts = mel_to_hz(np.linspace(hz_to_mel(FMIN), hz_to_mel(FMAX), N_MELS + 2))
    diff = np.diff(pts)
    slopes = pts[None, :] - freqs[:, None]
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    return np.maximum(0, np.minimum(down, up)), pts[1:-1]  # [n_freqs, n_mels], centers


FB, CENTERS = mel_filterbank()
LIVE = FB.sum(axis=0) > 0  # with n_fft 1024 the lowest filter(s) catch no FFT bins; drop them
WIN = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)  # periodic Hann


def logmel(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim == 2:
        x = x.mean(axis=1)
    assert sr == SR, f"{path}: expected {SR} Hz, got {sr}"
    x = x[:ATTACK]
    x = np.pad(x, (0, max(0, ATTACK - len(x))))
    x = np.pad(x, N_FFT // 2, mode="reflect")  # center=True
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP] * WIN
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    return 10 * np.log10(np.maximum(power @ FB[:, LIVE], 1e-10)).T  # [live mels, frames]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset root containing audio/")
    args = ap.parse_args()
    audio = Path(args.dataset).expanduser() / "audio"

    specs, ns = [], []
    for _, prefix in POSITIONS:
        files = sorted(audio.glob(prefix + "*.wav"))
        assert files, f"no clips matching {prefix}*.wav in {audio}"
        stack = np.stack([logmel(f) for f in files])
        specs.append(stack.mean(axis=0))
        ns.append(len(files))
    vmax = max(s.max() for s in specs)
    vmin = vmax - 80
    t_ms = np.arange(specs[0].shape[1]) * HOP / SR * 1000
    ticks = [100, 200, 400, 800, 1600, 3000]
    centers = CENTERS[LIVE]
    nm = len(centers)
    yt = np.interp(ticks, centers, np.arange(nm))

    OUT.mkdir(parents=True, exist_ok=True)
    for mode, t in THEMES.items():
        cmap = LinearSegmentedColormap.from_list("seq", t["ramp"])
        fig = plt.figure(figsize=(12, 4.6), facecolor=t["surface"])
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15], wspace=0.28,
                              left=0.06, right=0.98, top=0.74, bottom=0.14)
        for i, ((label, _), spec, n) in enumerate(zip(POSITIONS, specs, ns)):
            ax = fig.add_subplot(gs[i])
            ax.imshow(spec, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                      extent=[t_ms[0], t_ms[-1], -0.5, nm - 0.5])
            ax.set_title(label, color=t["text"], fontsize=10.5, loc="left")
            ax.text(0.02, 0.97, f"{n} takes", transform=ax.transAxes, va="top", fontsize=8.5, color=t["text2"])
            ax.set_yticks(yt, [f"{v:,}" for v in ticks] if i == 0 else [])
            ax.set_xlabel("Time (ms)", color=t["text2"], fontsize=9)
            ax.tick_params(colors=t["text2"], labelsize=8.5, length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_facecolor(t["surface"])
            if i == 0:
                ax.set_ylabel("Frequency (Hz, mel scale)", color=t["text2"], fontsize=9)
        ax = fig.add_subplot(gs[3])
        ax.set_facecolor(t["surface"])
        for (label, _), spec, c in zip(POSITIONS, specs, t["series"]):
            prof = spec.mean(axis=1)
            ax.plot(prof - prof.max(), np.arange(nm), color=c, linewidth=2, label=label)
        ax.set_yticks(yt, [])
        ax.set_ylim(-0.5, nm - 0.5)
        ax.set_xlim(-60, 3)
        ax.set_xlabel("Average level (dB, relative to peak)", color=t["text2"], fontsize=9)
        ax.set_title("Spectral profile", color=t["text"], fontsize=10.5, loc="left")
        ax.tick_params(colors=t["text2"], labelsize=8.5, length=0)
        ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
        ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(t["grid"])
        leg = ax.legend(frameon=False, fontsize=8.5, loc="lower left")
        for tx in leg.get_texts():
            tx.set_color(t["text2"])
        fig.text(0.06, 0.95, "Same note, three places: D3 (147 Hz) on three strings",
                 fontsize=13, weight="bold", color=t["text"], va="top")
        fig.text(0.06, 0.885, "Log-mel spectrograms of the first 250 ms (the model's input window), averaged over every take. The open string is brightest above 1 kHz",
                 fontsize=10, color=t["text2"], va="top")
        fig.savefig(OUT / f"spectrograms_{mode}.png", dpi=200, facecolor=t["surface"])
        plt.close(fig)
    print("wrote", [f"spectrograms_{m}.png" for m in THEMES])


if __name__ == "__main__":
    main()
