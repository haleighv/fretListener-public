#!/usr/bin/env python3
"""
tools/data_collection.py

MIDI-guided data collection for fretListener.

Workflow:
1. User selects a note (string + fret)
2. Records continuously while user plays multiple takes
3. On stop, detects onsets and chops into individual clips
4. Saves clips with proper naming convention

Dependencies:
  pip install sounddevice soundfile numpy librosa

Run:
  python tools/data_collection.py
"""

from __future__ import annotations

import csv
import re
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# DATASET / META HELPERS
# ============================

STANDARD_TUNING_MIDI = {
    6: 40,  # E2 (low E)
    5: 45,  # A2
    4: 50,  # D3
    3: 55,  # G3
    2: 59,  # B3
    1: 64,  # E4 (high E)
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

TAKE_REGEX = re.compile(r"_t(\d+)", re.IGNORECASE)


def compute_rms(x: np.ndarray) -> float:
    """Compute RMS of audio signal."""
    return float(np.sqrt(np.mean(x ** 2))) if x.size > 0 else 0.0


def midi_from_string_fret(string_num: int, fret: int) -> int:
    """Get MIDI note number from string and fret."""
    base = STANDARD_TUNING_MIDI.get(string_num)
    if base is None:
        raise ValueError(f"Invalid string number: {string_num}")
    return int(base + fret)


def midi_to_note_name(midi: int) -> str:
    """Convert MIDI note number to note name."""
    note = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return f"{note}{octave}"


def parse_class_name(name: str) -> Tuple[int, int, int]:
    """
    Parse class name like 's1_f0' or 's6_f7'.
    Returns: (string, fret, pos)
    """
    if not name or "_f" not in name:
        raise ValueError(f"Invalid class name: {name}")
    
    parts = name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid class name format: {name}")
    
    s_part = parts[0]
    f_part = parts[1]
    
    if not s_part.startswith("s") or not f_part.startswith("f"):
        raise ValueError(f"Invalid class name format: {name}")
    
    string = int(s_part[1:])
    fret = int(f_part[1:])
    pos = (string - 1) * 25 + fret
    
    return string, fret, pos


def extract_take_from_filename(filename: str) -> int:
    """Extract take number from filename like 's1_f0_t1.wav'."""
    m = TAKE_REGEX.search(filename)
    if not m:
        return -1
    return int(m.group(1))


def safe_slug(s: str) -> str:
    """Create safe filename slug."""
    s = (s or "").strip().lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_\-\+]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed"


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


# ============================
# GUI IMPORTS
# ============================

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:
    print("Error: tkinter not available. Please install python3-tk")
    exit(1)


# ============================
# GUI: FRETBOARD
# ============================

class GuitarFretboard(tk.Canvas):
    """Visual guitar fretboard widget showing current note."""

    # Display order: index 0 = top of screen (high E), index 5 = bottom (low E)
    STRING_NAMES_TOP_TO_BOTTOM = ["E4", "B3", "G3", "D3", "A2", "E2"]

    def __init__(self, parent, frets: int = 12, **kwargs):
        super().__init__(parent, **kwargs)
        self.frets = int(frets)
        self.active_notes: set[Tuple[int, int]] = set()

        self.string_spacing = 28
        self.fret_spacing = 50
        self.dot_radius = 10

        width = self.fret_spacing * (self.frets + 1) + 80
        height = self.string_spacing * 7 + 40

        self.config(width=width, height=height, bg="#2b2b2b", highlightthickness=0)
        self.draw_fretboard()

    def draw_fretboard(self):
        """Draw the fretboard."""
        self.delete("all")

        for fret in range(self.frets + 1):
            x = 45 + fret * self.fret_spacing
            y_start = 25
            y_end = 25 + self.string_spacing * 5

            if fret == 0:
                self.create_line(x, y_start, x, y_end, width=5, fill="#d4a574")
            else:
                self.create_line(x, y_start, x, y_end, width=2, fill="#8b7355")

        # Fret markers (between frets, not on them)
        for fret in [3, 5, 7, 9, 15, 17, 19, 21]:
            if fret <= self.frets:
                x = 45 + fret * self.fret_spacing - self.fret_spacing // 2
                marker_y = 25 + self.string_spacing * 2.5
                self.create_oval(x - 4, marker_y - 4, x + 4, marker_y + 4,
                               fill="#555", outline="")
        
        for fret in [12, 24]:
            if fret <= self.frets:
                x = 45 + fret * self.fret_spacing - self.fret_spacing // 2
                marker_y1 = 25 + self.string_spacing * 1.5
                marker_y2 = 25 + self.string_spacing * 3.5
                self.create_oval(x - 4, marker_y1 - 4, x + 4, marker_y1 + 4,
                               fill="#555", outline="")
                self.create_oval(x - 4, marker_y2 - 4, x + 4, marker_y2 + 4,
                               fill="#555", outline="")

        # Draw strings - top to bottom: E4(thin) -> E2(thick)
        for s in range(6):
            y = 25 + s * self.string_spacing
            x_start = 45
            x_end = 45 + self.frets * self.fret_spacing

            # s=0 is top (E4, thinnest), s=5 is bottom (E2, thickest)
            thickness = 1 + s * 0.4
            self.create_line(x_start, y, x_end, y, width=thickness, fill="#c0c0c0")

            self.create_text(18, y, text=self.STRING_NAMES_TOP_TO_BOTTOM[s],
                           font=("Arial", 9), fill="#888")

    def clear_notes(self):
        """Clear all active notes."""
        self.delete("note")
        self.active_notes.clear()

    def show_note(self, string_num: int, fret: int):
        """
        Show a note at given position.
        string_num: 6=low E, 5=A, 4=D, 3=G, 2=B, 1=high E
        Display: row 0=E4(top), row 5=E2(bottom)
        """
        self.clear_notes()
        # string 6 (low E) -> display row 5 (bottom)
        # string 1 (high E) -> display row 0 (top)
        display_idx = string_num - 1
        self._draw_note(display_idx, fret)

    def _draw_note(self, display_idx: int, fret: int):
        """Draw a note marker at display position."""
        if fret < 0 or fret > self.frets:
            return
        if display_idx < 0 or display_idx >= 6:
            return

        y = 25 + display_idx * self.string_spacing
        
        if fret == 0:
            x = 45
        else:
            x = 45 + fret * self.fret_spacing - self.fret_spacing // 2

        self.create_oval(
            x - self.dot_radius, y - self.dot_radius,
            x + self.dot_radius, y + self.dot_radius,
            fill="#ff6b6b", outline="#ff4444", width=2, tags="note"
        )


# ============================
# CAPTURE TARGET
# ============================

@dataclass
class CaptureTarget:
    """Represents a note to capture (string + fret)."""
    
    string_num: int  # 6..1 (6=low E, 1=high E)
    fret: int        # 0..N

    @property
    def class_dirname(self) -> str:
        return f"s{self.string_num}_f{self.fret}"

    @property
    def midi(self) -> int:
        return midi_from_string_fret(self.string_num, self.fret)

    @property
    def note(self) -> str:
        return midi_to_note_name(self.midi)


# ============================
# BUILDER: METADATA
# ============================

def build_metadata(
    output_root: Path,
    organized_dir: Path,
    samples_per_take: int,
    frets_per_string: int
) -> None:
    """Build metadata CSV from organized directory."""
    audio_dir = output_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    meta_csv = meta_dir / "metadata.csv"
    
    rows: List[Dict[str, any]] = []
    file_id = 0

    class_dirs = sorted(
        [p for p in organized_dir.iterdir() if p.is_dir() and p.name.startswith("s")],
        key=lambda p: p.name
    )
    
    if not class_dirs:
        print(f"Warning: No class directories found in {output_root}")
        return

    max_fret = 0
    for cls_dir in class_dirs:
        try:
            _, fret, _ = parse_class_name(cls_dir.name)
            max_fret = max(max_fret, fret)
        except ValueError:
            continue

    frets_per_string_plus1 = int(max_fret) + 1

    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        
        try:
            string, fret, _ = parse_class_name(cls_name)
        except ValueError as e:
            print(f"Warning: Skipping {cls_name}: {e}")
            continue

        wavs = sorted(cls_dir.glob("*.wav"))
        if not wavs:
            continue

        for wav_file in wavs:
            try:
                dst_path = audio_dir / wav_file.name
                if not dst_path.exists():
                    shutil.copy2(str(wav_file), str(dst_path))

                data, sr = sf.read(str(dst_path), dtype="float32", always_2d=False)
                num_samples = int(data.shape[0]) if hasattr(data, "shape") else len(data)

                take = extract_take_from_filename(dst_path.name)
                pos = (string - 1) * frets_per_string_plus1 + fret
                midi = midi_from_string_fret(string, fret)
                note = midi_to_note_name(midi)

                rows.append({
                    "id": file_id,
                    "dataset_name": output_root.name,
                    "relpath": str(dst_path.relative_to(output_root)),
                    "filename": dst_path.name,
                    "string": string,
                    "fret": fret,
                    "pos": pos,
                    "midi": midi,
                    "note": note,
                    "take": take,
                    "sample_rate": int(sr),
                    "num_samples": num_samples,
                    "source_relpath": str(dst_path.relative_to(output_root)),
                })
                file_id += 1

            except Exception as e:
                print(f"Warning: Failed to process {wav_file}: {e}")
                continue

    if not rows:
        print("Warning: No valid audio files found to build metadata")
        return

    fieldnames = [
        "id", "dataset_name", "relpath", "filename",
        "string", "fret", "pos", "midi", "note", "take",
        "sample_rate", "num_samples", "source_relpath"
    ]

    with open(meta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Metadata written to {meta_csv}")
    print(f"Total entries: {len(rows)}")


# ============================
# GUI APP - SINGLE WINDOW
# ============================

class DataCollectionGUI:
    """Main GUI application for data collection - batch recording approach."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("FretListener Data Collection")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.configure(bg="#1e1e1e")

        # State
        self.is_recording = False
        self._stopping = False  # functional: gate callbacks during shutdown
        self.stream: Optional[sd.InputStream] = None
        self.output_stream: Optional[sd.OutputStream] = None
        self.recording_buffer: List[np.ndarray] = []

        # Audio devices
        self.device_list: List[Tuple[int, str]] = []
        self.channel_list: List[int] = []

        # Capture targets
        self.current_targets: List[CaptureTarget] = []
        self.target_index = 0
        self.takes_per_note = 10
        self.frets_per_string = 12
        self.session_id = "A"

        # Live onset counter
        self.onset_count = 0
        self.last_rms = 0.0
        self.samples_since_onset = MIN_GAP_SAMPLES
        
        # Silence detection for auto-stop
        self.silence_samples = 0
        self.has_had_audio = False  # Track if we've seen any audio yet

        # Storage
        self.output_root: Optional[Path] = None
        self.organized_dir: Optional[Path] = None

        # Monitoring cache (functional: avoid Tk calls in audio threads)
        self.monitor_enabled = True
        self.monitor_audio = None

        # Current target
        self.string_var = tk.IntVar(value=6)
        self.fret_var = tk.IntVar(value=0)
        self.string_var.trace_add("write", lambda *a: self._update_target_display())
        self.fret_var.trace_add("write", lambda *a: self._update_target_display())

        self._setup_ui()
        self._refresh_devices()

        # functional: keep monitor_enabled in sync on the main thread
        try:
            self.monitor_var.trace_add("write", lambda *a: self._sync_monitor_enabled())
            self._sync_monitor_enabled()
        except Exception:
            pass

    def _sync_monitor_enabled(self):
        self.monitor_enabled = bool(self.monitor_var.get())

    def _setup_ui(self):
        """Set up single-window UI."""
        style = ttk.Style()
        style.theme_use('clam')
        
        bg_dark = "#1e1e1e"
        bg_mid = "#2d2d2d"
        fg = "#e0e0e0"
        fg_dim = "#888888"
        accent = "#4fc3f7"
        
        style.configure(".", background=bg_dark, foreground=fg)
        style.configure("TFrame", background=bg_dark)
        style.configure("TLabel", background=bg_dark, foreground=fg)
        style.configure("TLabelframe", background=bg_dark, foreground=fg)
        style.configure("TLabelframe.Label", background=bg_dark, foreground=accent)
        
        # Fix combobox colors
        style.map('TCombobox', 
                  fieldbackground=[('readonly', bg_mid)],
                  foreground=[('readonly', fg)],
                  selectbackground=[('readonly', bg_mid)],
                  selectforeground=[('readonly', fg)])
        style.configure("TCombobox", fieldbackground=bg_mid, foreground=fg, background=bg_mid)
        
        style.configure("TEntry", fieldbackground=bg_mid, foreground=fg)
        style.configure("TSpinbox", fieldbackground=bg_mid, foreground=fg)
        
        # Progress bar with accent color
        style.configure("Accent.Horizontal.TProgressbar",
                       troughcolor=bg_mid,
                       background=accent,
                       lightcolor=accent,
                       darkcolor=accent)
        
        # Button styles
        style.configure("Start.TButton", 
                       font=("Arial", 14, "bold"),
                       padding=(20, 15))
        style.configure("Stop.TButton",
                       font=("Arial", 14, "bold"),
                       padding=(20, 15))
        style.configure("Action.TButton",
                       font=("Arial", 11),
                       padding=(15, 10))

        # Main container
        main = ttk.Frame(self.root, padding="15")
        main.pack(fill="both", expand=True)

        # === LEFT COLUMN: Config + Controls ===
        left_col = ttk.Frame(main)
        left_col.pack(side="left", fill="y", padx=(0, 15))

        # Audio Device
        device_frame = ttk.LabelFrame(left_col, text="Audio Device", padding="10")
        device_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(device_frame, text="Device:").pack(anchor="w")
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(device_frame, textvariable=self.device_var, 
                                         state="readonly", width=30)
        self.device_combo.pack(fill="x", pady=(2, 8))
        self.device_combo.bind("<<ComboboxSelected>>", lambda e: self._on_device_selected())

        ttk.Label(device_frame, text="Channel:").pack(anchor="w")
        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(device_frame, textvariable=self.channel_var,
                                          state="readonly", width=30)
        self.channel_combo.pack(fill="x", pady=(2, 8))

        ttk.Button(device_frame, text="Refresh", 
                  command=self._refresh_devices,
                  style="Action.TButton").pack(anchor="e", pady=(5, 0))

        # Dataset Config
        dataset_frame = ttk.LabelFrame(left_col, text="Dataset", padding="10")
        dataset_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(dataset_frame, text="Guitar:").pack(anchor="w")
        self.guitar_var = tk.StringVar(value="strat")
        ttk.Entry(dataset_frame, textvariable=self.guitar_var, width=30).pack(fill="x", pady=(2, 8))

        ttk.Label(dataset_frame, text="Pickup Position:").pack(anchor="w")
        self.pickup_var = tk.StringVar(value="bridge")
        pickup_combo = ttk.Combobox(dataset_frame, textvariable=self.pickup_var,
                                    values=["bridge", "middle", "neck", "bridge+middle", "middle+neck"],
                                    state="readonly", width=28)
        pickup_combo.pack(fill="x", pady=(2, 8))
        pickup_combo.current(0)

        config_grid = ttk.Frame(dataset_frame)
        config_grid.pack(fill="x", pady=(5, 0))

        ttk.Label(config_grid, text="Takes/note:").grid(row=0, column=0, sticky="w")
        self.takes_var = tk.IntVar(value=10)
        ttk.Spinbox(config_grid, from_=1, to=20, textvariable=self.takes_var, width=8).grid(
            row=0, column=1, sticky="w", padx=(5, 15))

        ttk.Label(config_grid, text="Frets:").grid(row=0, column=2, sticky="w")
        self.frets_var = tk.IntVar(value=12)
        ttk.Spinbox(config_grid, from_=1, to=24, textvariable=self.frets_var, width=8).grid(
            row=0, column=3, sticky="w", padx=5)

        ttk.Label(dataset_frame, text="Session ID:").pack(anchor="w", pady=(8, 0))
        self.session_var = tk.StringVar(value="A")
        ttk.Entry(dataset_frame, textvariable=self.session_var, width=30).pack(fill="x", pady=(2, 0))

        # Control Buttons
        btn_frame = ttk.Frame(left_col)
        btn_frame.pack(fill="x", pady=15)

        self.record_btn = ttk.Button(
            btn_frame, 
            text="START SESSION",
            command=self.start_recording,
            style="Start.TButton"
        )
        self.record_btn.pack(fill="x", pady=(0, 8))

        self.stop_btn = ttk.Button(
            btn_frame,
            text="STOP SESSION",
            command=self.stop_session,
            state="disabled",
            style="Stop.TButton"
        )
        self.stop_btn.pack(fill="x")

        # Progress
        progress_frame = ttk.LabelFrame(left_col, text="Progress", padding="10")
        progress_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(progress_frame, text="Current Note:").pack(anchor="w")
        self.pb_note = ttk.Progressbar(progress_frame, maximum=100, value=0, 
                                        length=200, style="Accent.Horizontal.TProgressbar")
        self.pb_note.pack(fill="x", pady=(2, 2))
        self.pb_note_label = ttk.Label(progress_frame, text="0 / 10 takes")
        self.pb_note_label.pack(anchor="w", pady=(0, 8))

        ttk.Label(progress_frame, text="Current String:").pack(anchor="w")
        self.pb_string = ttk.Progressbar(progress_frame, maximum=100, value=0, 
                                          length=200, style="Accent.Horizontal.TProgressbar")
        self.pb_string.pack(fill="x", pady=(2, 2))
        self.pb_string_label = ttk.Label(progress_frame, text="0 / 13 frets")
        self.pb_string_label.pack(anchor="w", pady=(0, 8))

        ttk.Label(progress_frame, text="Total Session:").pack(anchor="w")
        self.pb_total = ttk.Progressbar(progress_frame, maximum=100, value=0, 
                                         length=200, style="Accent.Horizontal.TProgressbar")
        self.pb_total.pack(fill="x", pady=(2, 2))
        self.pb_total_label = ttk.Label(progress_frame, text="0 / 78 notes")
        self.pb_total_label.pack(anchor="w")

        # Audio Monitor
        monitor_frame = ttk.LabelFrame(left_col, text="Audio Monitor", padding="10")
        monitor_frame.pack(fill="x", pady=(0, 10))
        
        self.monitor_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(monitor_frame, text="Enable monitoring (hear input)", 
                       variable=self.monitor_var).pack(anchor="w")

        # Dataset Actions
        actions_frame = ttk.LabelFrame(left_col, text="Actions", padding="10")
        actions_frame.pack(fill="x")

        ttk.Button(actions_frame, text="Build Metadata",
                  command=self.build_metadata_action,
                  style="Action.TButton").pack(fill="x")

        # === RIGHT COLUMN: Display ===
        right_col = ttk.Frame(main)
        right_col.pack(side="left", fill="both", expand=True)

        # Current Target Display
        target_display = ttk.Frame(right_col)
        target_display.pack(fill="x", pady=(0, 10))

        # Big note display
        self.note_display_var = tk.StringVar(value="E2")
        note_label = tk.Label(target_display, textvariable=self.note_display_var,
                             font=("Arial", 48, "bold"),
                             fg=accent, bg=bg_dark)
        note_label.pack(side="left", padx=(0, 20))

        # String/Fret info
        info_frame = ttk.Frame(target_display)
        info_frame.pack(side="left", fill="y")

        self.string_label = ttk.Label(info_frame, text="String: 6", font=("Arial", 16))
        self.string_label.pack(anchor="w")
        self.fret_label = ttk.Label(info_frame, text="Fret: 0", font=("Arial", 16))
        self.fret_label.pack(anchor="w")
        self.midi_label = ttk.Label(info_frame, text="MIDI: 40", font=("Arial", 12), foreground=fg_dim)
        self.midi_label.pack(anchor="w")

        # Status
        self.status_var = tk.StringVar(value="Ready - Select target and start recording")
        status_label = ttk.Label(right_col, textvariable=self.status_var,
                                font=("Arial", 11), foreground=fg_dim)
        status_label.pack(anchor="w", pady=(0, 10))

        # Fretboard
        fretboard_frame = ttk.LabelFrame(right_col, text="Fretboard", padding="5")
        fretboard_frame.pack(fill="x", pady=(0, 10))

        self.fretboard = GuitarFretboard(fretboard_frame, frets=24)
        self.fretboard.pack()

        # Log
        log_frame = ttk.LabelFrame(right_col, text="Log", padding="5")
        log_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(log_frame, height=8, wrap="word",
                               bg=bg_mid, fg=fg,
                               insertbackground=fg,
                               selectbackground=accent,
                               font=("Consolas", 10))
        self.log_text.pack(fill="both", expand=True, side="left")

        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(fill="y", side="right")
        self.log_text.config(yscrollcommand=scrollbar.set)

        # Initial display update
        self._update_target_display()

    def _refresh_devices(self):
        """Refresh available audio devices."""
        try:
            devices = sd.query_devices()
            self.device_list = [
                (i, d['name'])
                for i, d in enumerate(devices)
                if d.get('max_input_channels', 0) > 0
            ]

            if self.device_list:
                device_names = [name for _, name in self.device_list]
                self.device_combo['values'] = device_names
                self.device_combo.current(0)
                self._on_device_selected()
            else:
                messagebox.showwarning("No Devices", "No audio input devices found")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to query devices: {e}")

    def _on_device_selected(self):
        """Handle device selection."""
        try:
            current = self.device_combo.current()
            if current < 0 or current >= len(self.device_list):
                return

            device_idx, _ = self.device_list[current]
            info = sd.query_devices(device_idx)
            max_ch = int(info.get('max_input_channels', 0))

            self.channel_list = list(range(max_ch))
            channel_names = [f"Channel {i+1}" for i in range(max_ch)]
            self.channel_combo['values'] = channel_names

            if channel_names:
                self.channel_combo.current(0)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to get device info: {e}")

    def _log(self, msg: str):
        """Log a message."""
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")

    def _update_target_display(self):
        """Update target display on fretboard and labels."""
        try:
            string = self.string_var.get()
            fret = self.fret_var.get()
            midi = midi_from_string_fret(string, fret)
            note = midi_to_note_name(midi)
            
            self.note_display_var.set(note)
            self.string_label.config(text=f"String: {string}")
            self.fret_label.config(text=f"Fret: {fret}")
            self.midi_label.config(text=f"MIDI: {midi}")
            
            self.fretboard.show_note(string, fret)
        except:
            pass

    def _ensure_output_dirs(self):
        """Ensure output directories exist."""
        guitar = safe_slug(self.guitar_var.get()) or "guitar"
        pickup = safe_slug(self.pickup_var.get()) or "bridge"
        takes = self.takes_var.get()
        
        dataset_name = f"{guitar}_{pickup}_{takes}"
        self.output_root = Path("data") / dataset_name
        self.organized_dir = self.output_root / "organized"
        self.organized_dir.mkdir(parents=True, exist_ok=True)

    def next_note(self):
        """Move to next note in sequence."""
        string = self.string_var.get()
        fret = self.fret_var.get()
        max_fret = self.frets_var.get()
        
        fret += 1
        if fret > max_fret:
            fret = 0
            string -= 1
            if string < 1:
                string = 6
        
        self.string_var.set(string)
        self.fret_var.set(fret)
        self._update_all_progress()

    def start_recording(self):
        """Start recording audio for current target."""
        if self.is_recording:
            return

        try:
            device_idx_combo = self.device_combo.current()
            channel_idx = self.channel_combo.current()

            if device_idx_combo < 0 or channel_idx < 0:
                messagebox.showerror("Error", "Please select device and channel")
                return

            device_idx, _ = self.device_list[device_idx_combo]
            self._ensure_output_dirs()

            # functional: cache monitor state for callbacks (avoid Tk access)
            self._sync_monitor_enabled()

            # Reset state
            self.recording_buffer = []
            self.onset_count = 0
            self.last_rms = 0.0
            self.samples_since_onset = MIN_GAP_SAMPLES
            self.silence_samples = 0
            self.has_had_audio = False
            self.takes_per_note = self.takes_var.get()

            # functional: clear shutdown gate
            self._stopping = False

            # Update progress display
            self.pb_note['maximum'] = 100
            self.pb_note['value'] = 0
            self.pb_note_label.config(text=f"0 / {self.takes_per_note} takes")
            self._update_all_progress()

            # Flag for auto-advance (set from callback, processed in main thread)
            self.should_advance = False
            self.monitor_audio = None

            def audio_callback(indata, frames, time_info, status):
                # Early exit if not recording / stopping
                if (not self.is_recording) or self._stopping:
                    return
                    
                if status:
                    pass

                # Get mono audio
                if indata.ndim == 2:
                    mono = indata[:, channel_idx].astype(np.float32, copy=True)
                else:
                    mono = indata.astype(np.float32, copy=True)

                # Store in buffer
                self.recording_buffer.append(mono)
                
                # Store for monitoring output (functional: no Tk access)
                if self.monitor_enabled:
                    self.monitor_audio = mono.copy()

                # Live onset detection for progress feedback
                rms = compute_rms(mono)
                self.samples_since_onset += len(mono)

                # Track if we've had audio above threshold
                if rms > ONSET_THRESHOLD:
                    self.has_had_audio = True
                    self.silence_samples = 0
                    
                    if self.samples_since_onset >= MIN_GAP_SAMPLES:
                        if self.last_rms <= 0 or rms > self.last_rms * 1.5:
                            self.onset_count += 1
                            self.samples_since_onset = 0
                            self.root.after(0, self._update_onset_progress)
                elif rms < SILENCE_THRESHOLD:
                    # Only start counting silence after we have enough takes
                    if self.onset_count >= self.takes_per_note:
                        self.silence_samples += len(mono)
                        
                        # Set flag for main thread to handle
                        if self.silence_samples >= SILENCE_SAMPLES and not self.should_advance:
                            self.should_advance = True
                            self.root.after(0, self._check_advance)

                self.last_rms = rms

            def output_callback(outdata, frames, time_info, status):
                # functional: never call Tk from audio thread; use cached monitor_enabled
                if self._stopping or (not self.monitor_enabled) or (self.monitor_audio is None):
                    outdata.fill(0)
                    return

                mono = self.monitor_audio
                n = min(len(mono), frames)
                outdata[:n, 0] = mono[:n]
                outdata[:n, 1] = mono[:n]
                if n < frames:
                    outdata[n:, :] = 0

            self.stream = sd.InputStream(
                device=device_idx,
                channels=sd.query_devices(device_idx)['max_input_channels'],
                samplerate=TARGET_SR,
                blocksize=ANALYSIS_HOP,
                dtype='float32',
                callback=audio_callback
            )
            
            # Start output stream for monitoring (functional: cached flag)
            if self.monitor_enabled:
                try:
                    self.output_stream = sd.OutputStream(
                        samplerate=TARGET_SR,
                        blocksize=ANALYSIS_HOP,
                        channels=2,
                        dtype='float32',
                        callback=output_callback,
                        latency='low'
                    )
                    self.output_stream.start()
                except Exception as e:
                    self._log(f"Monitor unavailable: {e}")
                    self.output_stream = None
            
            self.stream.start()

            self.is_recording = True
            self.record_btn.config(state="disabled")
            self.stop_btn.config(state="normal")

            string = self.string_var.get()
            fret = self.fret_var.get()
            note = midi_to_note_name(midi_from_string_fret(string, fret))
            
            self.status_var.set(f"Recording {note} - Play {self.takes_per_note} times")
            self._log(f"Recording s{string}_f{fret} ({note})")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recording: {e}")
            self.is_recording = False

    def _update_onset_progress(self):
        """Update onset progress display (called from main thread)."""
        count = min(self.onset_count, self.takes_per_note)
        self.pb_note['value'] = (count / self.takes_per_note) * 100
        self.pb_note_label.config(text=f"{count} / {self.takes_per_note} takes")

    def _update_all_progress(self):
        """Update all progress bars."""
        string = self.string_var.get()
        fret = self.fret_var.get()
        max_fret = self.frets_var.get()
        
        # String progress (frets completed on current string)
        frets_per_string = max_fret + 1
        string_progress = (fret / frets_per_string) * 100
        self.pb_string['value'] = string_progress
        self.pb_string_label.config(text=f"{fret} / {frets_per_string} frets")
        
        # Total progress
        total_notes = 6 * frets_per_string
        strings_done = 6 - string  # strings 6,5,4,3,2,1 -> 0,1,2,3,4,5 done
        notes_done = strings_done * frets_per_string + fret
        total_progress = (notes_done / total_notes) * 100
        self.pb_total['value'] = total_progress
        self.pb_total_label.config(text=f"{notes_done} / {total_notes} notes")

    def _check_advance(self):
        """Check if we should advance (called from main thread via after())."""
        if not self.should_advance or not self.is_recording:
            return
        self.should_advance = False
        self._auto_stop_and_advance()

    def _auto_stop_and_advance(self):
        """Auto-stop recording, process, and advance to next note."""
        if not self.is_recording:
            return
        
        # Stop and process current recording
        self._stop_and_process()
        
        # Check if we've completed the session
        string = self.string_var.get()
        fret = self.fret_var.get()
        max_fret = self.frets_var.get()
        
        # Check if this was the last note (string 1, max fret)
        if string == 1 and fret == max_fret:
            self._log("Session complete!")
            self.status_var.set("Session complete!")
            return
        
        # Advance to next note
        self.next_note()
        
        # Auto-start recording for next note after short delay
        self.root.after(500, self.start_recording)

    def _stop_and_process(self):
        """Stop recording and process without advancing."""
        if not self.is_recording:
            return

        # Mark as not recording first to stop callback processing
        self.is_recording = False

        # functional: gate callbacks while stopping streams
        self._stopping = True

        try:
            # functional: stop/close (avoid abort-related thread/GIL issues)
            if self.output_stream is not None:
                try:
                    self.output_stream.stop()
                    self.output_stream.close()
                except:
                    pass
                self.output_stream = None

            if self.stream is not None:
                try:
                    self.stream.stop()
                    self.stream.close()
                except:
                    pass
                self.stream = None

            self.record_btn.config(state="normal")
            self.stop_btn.config(state="disabled")

            self.status_var.set("Processing...")
            self.root.update()  # Force UI update

            # Concatenate all recorded audio
            if not self.recording_buffer:
                self._log("No audio recorded")
                self.status_var.set("Ready")
                self._stopping = False
                return

            full_audio = np.concatenate(self.recording_buffer)
            duration = len(full_audio) / TARGET_SR
            self._log(f"Recorded {duration:.1f}s")

            # Get target info
            string = self.string_var.get()
            fret = self.fret_var.get()
            session = self.session_var.get().strip() or "A"

            # Create output directory for this note
            class_dir = self.organized_dir / f"s{string}_f{fret}"
            class_dir.mkdir(parents=True, exist_ok=True)

            # Chop recording into clips
            clips_saved = chop_recording(
                full_audio,
                class_dir,
                string,
                fret,
                session
            )

            self._log(f"Saved {clips_saved} clips to s{string}_f{fret}/")
            
            # Update progress to show final count
            self.pb_note['value'] = 100
            self.pb_note_label.config(text=f"{clips_saved} / {self.takes_per_note} takes")

            note = midi_to_note_name(midi_from_string_fret(string, fret))
            self.status_var.set(f"Done - {clips_saved} clips for {note}")

            # Clear buffer
            self.recording_buffer = []

        except Exception as e:
            self._log(f"Error: {e}")
            self.status_var.set("Error during processing")
        finally:
            # functional: release stop gate after streams are down
            self._stopping = False

    def stop_session(self):
        """Stop the entire session immediately."""
        if self.is_recording:
            self._stop_and_process()
            self._log("Session stopped by user")
            self.status_var.set("Session stopped")
            return

        self.is_recording = False

        # functional: gate callbacks while stopping streams
        self._stopping = True
        
        if self.output_stream is not None:
            try:
                self.output_stream.stop()
                self.output_stream.close()
            except:
                pass
            self.output_stream = None
        
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
            self.stream = None
        
        self.record_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.recording_buffer = []
        
        self._log("Session stopped by user")
        self.status_var.set("Session stopped")

        # functional: release stop gate
        self._stopping = False

    def stop_recording(self):
        """Stop recording and process the captured audio (manual stop)."""
        self._stop_and_process()

    def build_metadata_action(self):
        """Build metadata for captured dataset."""
        try:
            self._ensure_output_dirs()
            
            if not self.organized_dir or not self.organized_dir.exists():
                messagebox.showwarning("Warning", "No dataset found. Capture some notes first.")
                return

            build_metadata(
                self.output_root,
                self.organized_dir,
                self.takes_var.get(),
                self.frets_var.get()
            )

            self._log("Metadata built successfully")
            messagebox.showinfo("Success", "Metadata built successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to build metadata: {e}")

    def on_close(self):
        """Handle window close."""
        if self.is_recording:
            self.stop_session()
        self.root.destroy()


def main():
    """Main entry point."""
    root = tk.Tk()
    root.minsize(950, 580)
    app = DataCollectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
