#!/usr/bin/env python3
"""
demo_playback.py - Run the fretListener GUI on recorded WAV clips instead of live audio.

Each clip goes through the same inference path as live capture (note model sweep,
then string disambiguation). Predictions are printed next to the ground truth parsed
from the filename (s{string}_f{fret}_...). The window then shows the first clip that
was predicted correctly and needed disambiguation, and can save a screenshot of it.

Usage (run from the repo that holds your checkpoints and data):
    python tools/demo_playback.py \
        --note-checkpoint artifacts/models/note_pilot_v3/note_cnn_best.pt \
        --disambig-dir artifacts/models/disambig_triplet_batch_20251223_222859 \
        --wav data/my_guitar/audio/s6_f10_t08_sessA.wav data/my_guitar/audio/s5_f5_t03_sessA.wav \
        --screenshot docs/screenshots/fretlistener.png
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import tkinter as tk  # noqa: E402

import fretListener as fl  # noqa: E402
from common.audio import load_wav_mono  # noqa: E402



def parse_truth(path: Path):
    m = re.match(r"s(\d)_f(\d+)_", path.name)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def main():
    sys.stdout.reconfigure(line_buffering=True)  # show results live even when piped to tee
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note-checkpoint", required=True)
    ap.add_argument("--disambig-dir", required=True)
    ap.add_argument("--wav", nargs="+", required=True)
    ap.add_argument("--screenshot", help="Save a PNG of the window: click the fretListener window when the camera cursor appears (macOS)")
    ap.add_argument("--hold", type=float, default=0, help="Seconds to keep the window open after (0 = until closed)")
    args = ap.parse_args()

    root = tk.Tk()
    app = fl.FretListenerGUI(root)

    # Show the settings this demo actually uses (never written back to the settings cache)
    app.mode_var.set(fl.MODE_NOTE_DISAMBIG)
    app._on_mode_change()
    app.note_checkpoint_var.set(args.note_checkpoint)
    app.disambig_dir_var.set(args.disambig_dir)
    app.params["save_dir"] = ""
    root.update()

    print("Loading models...")
    device = app._get_model_device()
    app.state = app._load_checkpoints(Path(args.note_checkpoint), None, device, Path(args.disambig_dir))
    sr = app.state["sample_rate"]
    sweep = max(1, int(round(app.params.get("sweep_step_ms", 5.0) / 1000.0 * sr)))
    app.capture_state = {"sweep_step_samples": sweep, "sr": sr, "attack": app.state["attack_samples"]}

    showcase = None
    correct = 0
    print(f"\n{'clip':34s} {'truth':>8s} {'pred':>8s}  note")
    for w in args.wav:
        p = Path(w)
        audio, _ = load_wav_mono(p, target_sr=sr)
        _, window, topk = app._sweep_best_window_note(audio, sweep, topk=5)
        midi, name, prob = app._decide_note(topk, app.params.get("margin", 0.1))
        s_idx, fret, info = app._disambiguate_note(midi, window)
        pred = (6 - s_idx, fret)  # string_idx 0 = low E; string_num 1 = high E
        truth = parse_truth(p)
        ok = pred == truth
        correct += ok
        print(f"{p.name:34s} {f'S{truth[0]}F{truth[1]}':>8s} {f'S{pred[0]}F{pred[1]}':>8s}  "
              f"{name} p={prob:.2f}{info}  {'OK' if ok else 'MISS'}")
        if ok and showcase is None and "Unambiguous" not in info:
            showcase = (p, audio)
    print(f"\n{correct}/{len(args.wav)} positions correct")

    if showcase is None:
        print("No correctly predicted ambiguous clip to show; displaying the first clip.")
        p = Path(args.wav[0])
        showcase = (p, load_wav_mono(p, target_sr=sr)[0])
    print(f"Showing: {showcase[0].name}")
    app._process_captured_audio(showcase[1])

    def capture():
        # Interactive window pick: only the clicked window is saved, so nothing else on screen leaks in.
        out = Path(args.screenshot).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        print("Click the fretListener window to capture it (Esc to cancel)...", flush=True)
        rc = subprocess.call(["screencapture", "-x", "-o", "-W", str(out)])
        print(f"Screenshot {'saved to ' + str(out) if rc == 0 and out.exists() else 'cancelled'}")

    if args.screenshot:
        root.after(1500, capture)  # let the message loop draw the result first
    if args.hold:
        root.after(int(args.hold * 1000), root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
