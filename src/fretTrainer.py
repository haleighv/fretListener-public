#!/usr/bin/env python3
"""
fretTrainer.py - Interactive Guitar Training Tool

A GUI application for learning scales, arpeggios, and fretboard navigation
using real-time note detection from a trained neural network model.

Features:
- Scale practice (major, minor, pentatonic, modes, etc.)
- Arpeggio practice (major, minor, diminished, 7ths, etc.)
- Octave/position training (find the same note everywhere)
- Interval training
- Visual fretboard with target notes highlighted
- Real-time feedback on played notes
- Progress tracking within exercises

Dependencies:
  pip install sounddevice soundfile numpy torch torchaudio scipy
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import time
import random
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Set
from dataclasses import dataclass
from enum import Enum

import numpy as np
import sounddevice as sd
import torch
import torchaudio
from scipy.io import wavfile

import sys

# Ensure <repo_root>/src is importable when running from tools/
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from note_model import NoteCNN


# ============================
# VERSION
# ============================

VERSION = "1.0"


# ============================
# MUSIC THEORY DATA
# ============================

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Standard tuning MIDI notes (low to high)
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4

# Scale intervals (semitones from root)
SCALES = {
    "Major": [0, 2, 4, 5, 7, 9, 11],
    "Natural Minor": [0, 2, 3, 5, 7, 8, 10],
    "Harmonic Minor": [0, 2, 3, 5, 7, 8, 11],
    "Melodic Minor": [0, 2, 3, 5, 7, 9, 11],
    "Pentatonic Major": [0, 2, 4, 7, 9],
    "Pentatonic Minor": [0, 3, 5, 7, 10],
    "Blues": [0, 3, 5, 6, 7, 10],
    "Dorian": [0, 2, 3, 5, 7, 9, 10],
    "Phrygian": [0, 1, 3, 5, 7, 8, 10],
    "Lydian": [0, 2, 4, 6, 7, 9, 11],
    "Mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "Locrian": [0, 1, 3, 5, 6, 8, 10],
    "Whole Tone": [0, 2, 4, 6, 8, 10],
    "Diminished (HW)": [0, 1, 3, 4, 6, 7, 9, 10],
    "Diminished (WH)": [0, 2, 3, 5, 6, 8, 9, 11],
    "Chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}

# Arpeggio intervals (semitones from root)
ARPEGGIOS = {
    "Major": [0, 4, 7],
    "Minor": [0, 3, 7],
    "Diminished": [0, 3, 6],
    "Augmented": [0, 4, 8],
    "Major 7": [0, 4, 7, 11],
    "Minor 7": [0, 3, 7, 10],
    "Dominant 7": [0, 4, 7, 10],
    "Minor 7b5": [0, 3, 6, 10],
    "Diminished 7": [0, 3, 6, 9],
    "Major 9": [0, 4, 7, 11, 14],
    "Minor 9": [0, 3, 7, 10, 14],
    "Dominant 9": [0, 4, 7, 10, 14],
    "Sus2": [0, 2, 7],
    "Sus4": [0, 5, 7],
    "Add9": [0, 4, 7, 14],
    "6th": [0, 4, 7, 9],
    "Minor 6th": [0, 3, 7, 9],
}

# Intervals for interval training
INTERVALS = {
    "Minor 2nd": 1,
    "Major 2nd": 2,
    "Minor 3rd": 3,
    "Major 3rd": 4,
    "Perfect 4th": 5,
    "Tritone": 6,
    "Perfect 5th": 7,
    "Minor 6th": 8,
    "Major 6th": 9,
    "Minor 7th": 10,
    "Major 7th": 11,
    "Octave": 12,
}


# ============================
# AUDIO CONFIG
# ============================

ANALYSIS_HOP = 1024


# ============================
# STATUS LIGHT WIDGET
# ============================

class StatusLight(tk.Canvas):
    """A status indicator light widget."""
    
    COLORS = {
        "ready": "#4ade80",       # Green - waiting for input
        "correct": "#22c55e",     # Bright green - correct note
        "wrong": "#ef4444",       # Red - wrong note
        "listening": "#facc15",   # Yellow - processing
        "stopped": "#4b5563",     # Dark gray - not running
        "complete": "#3b82f6",    # Blue - exercise complete
    }
    
    def __init__(self, parent, size: int = 40, **kwargs):
        super().__init__(parent, width=size + 4, height=size + 4, 
                        highlightthickness=0, **kwargs)
        self.size = size
        self.state = "stopped"
        self.pulse_on = True
        self.pulse_job = None
        self._draw()
    
    def _draw(self):
        self.delete("all")
        padding = 2
        color = self.COLORS.get(self.state, self.COLORS["stopped"])
        
        if self.state == "ready" and not self.pulse_on:
            color = self._dim_color(color, 0.6)
        
        self.create_oval(padding, padding, 
                        self.size + padding, self.size + padding,
                        fill=color, outline="#1e1e1e", width=2)
    
    def _dim_color(self, hex_color: str, factor: float) -> str:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"
    
    def set_state(self, state: str):
        if self.pulse_job:
            self.after_cancel(self.pulse_job)
            self.pulse_job = None
        
        self.state = state
        self.pulse_on = True
        
        if state == "ready":
            self._start_pulse()
        else:
            self._draw()
    
    def _start_pulse(self):
        self.pulse_on = not self.pulse_on
        self._draw()
        self.pulse_job = self.after(600, self._start_pulse)


# ============================
# TRAINING FRETBOARD WIDGET
# ============================

class TrainingFretboard(tk.Canvas):
    """Enhanced fretboard widget for training with multiple note states."""
    
    STRING_NAMES_TOP_TO_BOTTOM = ["E4", "B3", "G3", "D3", "A2", "E2"]
    STANDARD_TUNING = [40, 45, 50, 55, 59, 64]
    
    def __init__(self, parent, frets: int = 15, **kwargs):
        super().__init__(parent, **kwargs)
        self.frets = int(frets)
        
        # Note states: 'target', 'played', 'correct', 'wrong', 'hint'
        self.note_states: Dict[Tuple[int, int], str] = {}  # (string_idx, fret) -> state
        
        self.string_spacing = 28
        self.fret_spacing = 45
        self.dot_radius = 10
        
        width = self.fret_spacing * (self.frets + 1) + 80
        height = self.string_spacing * 7 + 40
        
        self.config(width=width, height=height, bg="#2b2b2b", highlightthickness=0)
        self.draw_fretboard()
    
    def draw_fretboard(self):
        self.delete("all")
        
        # Frets
        for fret in range(self.frets + 1):
            x = 45 + fret * self.fret_spacing
            y_start = 25
            y_end = 25 + self.string_spacing * 5
            
            if fret == 0:
                self.create_line(x, y_start, x, y_end, width=5, fill="#d4a574")
            else:
                self.create_line(x, y_start, x, y_end, width=2, fill="#8b7355")
        
        # Fret markers
        for fret in [3, 5, 7, 9, 15, 17, 19, 21]:
            if fret <= self.frets:
                x = 45 + fret * self.fret_spacing - self.fret_spacing // 2
                marker_y = 25 + self.string_spacing * 2.5
                self.create_oval(x - 4, marker_y - 4, x + 4, marker_y + 4,
                               fill="#444", outline="")
        
        for fret in [12, 24]:
            if fret <= self.frets:
                x = 45 + fret * self.fret_spacing - self.fret_spacing // 2
                self.create_oval(x - 4, 25 + self.string_spacing * 1.5 - 4,
                               x + 4, 25 + self.string_spacing * 1.5 + 4,
                               fill="#444", outline="")
                self.create_oval(x - 4, 25 + self.string_spacing * 3.5 - 4,
                               x + 4, 25 + self.string_spacing * 3.5 + 4,
                               fill="#444", outline="")
        
        # Strings
        for s in range(6):
            y = 25 + s * self.string_spacing
            x_start = 45
            x_end = 45 + self.frets * self.fret_spacing
            thickness = 1 + s * 0.4
            self.create_line(x_start, y, x_end, y, width=thickness, fill="#c0c0c0")
            self.create_text(18, y, text=self.STRING_NAMES_TOP_TO_BOTTOM[s],
                           font=("Arial", 9), fill="#888")
        
        # Fret numbers
        for fret in [1, 3, 5, 7, 9, 12, 15]:
            if fret <= self.frets:
                x = 45 + fret * self.fret_spacing - self.fret_spacing // 2
                self.create_text(x, 25 + self.string_spacing * 5 + 15,
                               text=str(fret), font=("Arial", 9), fill="#666")
        
        # Redraw notes
        self._redraw_notes()
    
    def _redraw_notes(self):
        self.delete("note")
        for (string_idx, fret), state in self.note_states.items():
            self._draw_note(string_idx, fret, state)
    
    def _draw_note(self, string_idx: int, fret: int, state: str):
        if fret < 0 or fret > self.frets or string_idx < 0 or string_idx >= 6:
            return
        
        display_idx = 5 - string_idx
        y = 25 + display_idx * self.string_spacing
        
        if fret == 0:
            x = 45
        else:
            x = 45 + fret * self.fret_spacing - self.fret_spacing // 2
        
        # Colors based on state
        colors = {
            'target': ("#fbbf24", "#f59e0b"),      # Yellow/amber - target note
            'next': ("#60a5fa", "#3b82f6"),        # Blue - next note to play
            'played': ("#a78bfa", "#8b5cf6"),      # Purple - just played
            'correct': ("#4ade80", "#22c55e"),     # Green - correct
            'wrong': ("#f87171", "#ef4444"),       # Red - wrong
            'hint': ("#94a3b8", "#64748b"),        # Gray - hint/optional
            'scale': ("#6ee7b7", "#34d399"),       # Teal - scale note
        }
        
        fill, outline = colors.get(state, ("#ffffff", "#cccccc"))
        
        radius = self.dot_radius if state != 'hint' else self.dot_radius - 2
        
        self.create_oval(
            x - radius, y - radius,
            x + radius, y + radius,
            fill=fill, outline=outline, width=2, tags="note"
        )
        
        # Add note name for target/next notes
        if state in ('target', 'next'):
            midi = self.STANDARD_TUNING[string_idx] + fret
            note_name = NOTE_NAMES[midi % 12]
            self.create_text(x, y, text=note_name, font=("Arial", 8, "bold"),
                           fill="#000", tags="note")
    
    def clear_notes(self):
        self.note_states.clear()
        self.delete("note")
    
    def set_note_state(self, string_idx: int, fret: int, state: str):
        self.note_states[(string_idx, fret)] = state
        self._redraw_notes()
    
    def remove_note(self, string_idx: int, fret: int):
        if (string_idx, fret) in self.note_states:
            del self.note_states[(string_idx, fret)]
            self._redraw_notes()
    
    def show_scale(self, root_midi: int, intervals: List[int], state: str = 'scale'):
        """Show all positions for a scale on the fretboard."""
        self.clear_notes()
        scale_notes = set((root_midi + i) % 12 for i in intervals)
        
        for string_idx, open_note in enumerate(self.STANDARD_TUNING):
            for fret in range(self.frets + 1):
                midi = open_note + fret
                if midi % 12 in scale_notes:
                    note_state = 'target' if midi % 12 == root_midi % 12 else state
                    self.note_states[(string_idx, fret)] = note_state
        
        self._redraw_notes()
    
    def show_target_note(self, midi: int, state: str = 'target'):
        """Show all positions for a single note."""
        self.clear_notes()
        for string_idx, open_note in enumerate(self.STANDARD_TUNING):
            fret = midi - open_note
            if 0 <= fret <= self.frets:
                self.note_states[(string_idx, fret)] = state
        self._redraw_notes()
    
    def show_sequence(self, midi_notes: List[int], current_idx: int):
        """Show a sequence with current note highlighted."""
        self.clear_notes()
        
        for i, midi in enumerate(midi_notes):
            for string_idx, open_note in enumerate(self.STANDARD_TUNING):
                fret = midi - open_note
                if 0 <= fret <= self.frets:
                    if i == current_idx:
                        self.note_states[(string_idx, fret)] = 'next'
                    elif i < current_idx:
                        self.note_states[(string_idx, fret)] = 'correct'
                    else:
                        self.note_states[(string_idx, fret)] = 'hint'
        
        self._redraw_notes()
    
    def midi_to_positions(self, midi: int) -> List[Tuple[int, int]]:
        """Get all fretboard positions for a MIDI note."""
        positions = []
        for string_idx, open_note in enumerate(self.STANDARD_TUNING):
            fret = midi - open_note
            if 0 <= fret <= self.frets:
                positions.append((string_idx, fret))
        return positions


# ============================
# EXERCISE TYPES
# ============================

class ExerciseType(Enum):
    SCALE_ASCENDING = "scale_asc"
    SCALE_DESCENDING = "scale_desc"
    SCALE_BOTH = "scale_both"
    ARPEGGIO = "arpeggio"
    FIND_NOTE = "find_note"
    FIND_OCTAVES = "find_octaves"
    INTERVAL = "interval"
    RANDOM_SCALE = "random_scale"


@dataclass
class Exercise:
    """Represents a training exercise."""
    name: str
    exercise_type: ExerciseType
    root_note: str
    pattern_name: str
    target_notes: List[int]  # MIDI notes to play in order
    current_index: int = 0
    correct_count: int = 0
    wrong_count: int = 0
    
    @property
    def is_complete(self) -> bool:
        return self.current_index >= len(self.target_notes)
    
    @property
    def current_target(self) -> Optional[int]:
        if self.current_index < len(self.target_notes):
            return self.target_notes[self.current_index]
        return None
    
    def advance(self):
        self.current_index += 1
    
    def record_attempt(self, correct: bool):
        if correct:
            self.correct_count += 1
        else:
            self.wrong_count += 1


# ============================
# MAIN APPLICATION
# ============================

class FretTrainerGUI:
    """Main GUI application for guitar training."""
    
    def __init__(self, root):
        self.root = root
        self.root.title(f"FretTrainer v{VERSION} - Guitar Practice Tool")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.configure(bg="#1e1e1e")
        
        # State
        self.is_running = False
        self.stream: Optional[sd.InputStream] = None
        self.output_stream: Optional[sd.OutputStream] = None
        self.state = None  # Model state
        
        # Thread-safe communication
        self.audio_queue = queue.Queue()
        self.message_queue = queue.Queue()
        
        # Audio monitoring
        self.monitor_enabled = True
        self.monitor_audio = None
        self._stopping = False
        
        # Audio devices
        self.device_list: List[Tuple[int, str]] = []
        self.channel_list: List[int] = []
        
        # Capture state
        self.capture_state = {
            'mode': 'stopped',
            'pre_buffer': None,
            'pre_idx': 0,
            'recording_parts': [],
            'recorded_post': 0,
            'silence_ctr': 0,
        }
        
        # Training state
        self.current_exercise: Optional[Exercise] = None
        self.last_detected_midi: Optional[int] = None
        self.detection_cooldown = 0
        
        # Default parameters
        self.params = {
            'checkpoint': 'artifacts/models/note_20251211_164726/note_cnn_best.pt',
            'trigger_threshold': 0.01,
            'pre_ms': 50.0,
            'post_ms': 500.0,
            'silence_threshold': 0.006,
            'min_silence_ms': 80.0,
            'sweep_step_ms': 5.0,
            'margin': 0.10,
            'device_idx': None,
            'channel_idx': 0
        }
        
        self._setup_ui()
        self._refresh_devices()
        self._update_message_display()
    
    def _setup_ui(self):
        """Set up the UI."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Colors
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
        
        style.map('TCombobox',
                  fieldbackground=[('readonly', bg_mid)],
                  foreground=[('readonly', fg)],
                  selectbackground=[('readonly', bg_mid)],
                  selectforeground=[('readonly', fg)])
        style.configure("TCombobox", fieldbackground=bg_mid, foreground=fg, background=bg_mid)
        
        style.configure("TButton", font=("Arial", 11), padding=(15, 10))
        style.configure("Big.TButton", font=("Arial", 14, "bold"), padding=(20, 15))
        
        # Main container
        main = ttk.Frame(self.root, padding="15")
        main.pack(fill="both", expand=True)
        
        # === LEFT COLUMN ===
        left_col = ttk.Frame(main)
        left_col.pack(side="left", fill="y", padx=(0, 15))
        
        # Model/Device section
        device_frame = ttk.LabelFrame(left_col, text="Audio Setup", padding="10")
        device_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(device_frame, text="Model:").pack(anchor="w")
        self.checkpoint_var = tk.StringVar(value=self.params['checkpoint'])
        ckpt_frame = ttk.Frame(device_frame)
        ckpt_frame.pack(fill="x", pady=(2, 8))
        ttk.Entry(ckpt_frame, textvariable=self.checkpoint_var, width=25).pack(side="left", fill="x", expand=True)
        ttk.Button(ckpt_frame, text="...", width=3, command=self._browse_checkpoint).pack(side="left", padx=(5, 0))
        
        ttk.Label(device_frame, text="Device:").pack(anchor="w")
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(device_frame, textvariable=self.device_var, state="readonly", width=28)
        self.device_combo.pack(fill="x", pady=(2, 8))
        self.device_combo.bind("<<ComboboxSelected>>", lambda e: self._on_device_selected())
        
        ttk.Label(device_frame, text="Channel:").pack(anchor="w")
        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(device_frame, textvariable=self.channel_var, state="readonly", width=28)
        self.channel_combo.pack(fill="x", pady=(2, 8))
        
        self.monitor_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(device_frame, text="Enable audio monitoring",
                       variable=self.monitor_var,
                       command=self._sync_monitor).pack(anchor="w", pady=(5, 0))
        
        # Exercise selection
        exercise_frame = ttk.LabelFrame(left_col, text="Exercise", padding="10")
        exercise_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(exercise_frame, text="Type:").pack(anchor="w")
        self.exercise_type_var = tk.StringVar(value="Scale Practice")
        exercise_types = ["Scale Practice", "Arpeggio Practice", "Find Note", "Find Octaves", "Interval Training"]
        self.exercise_type_combo = ttk.Combobox(exercise_frame, textvariable=self.exercise_type_var,
                                                values=exercise_types, state="readonly", width=28)
        self.exercise_type_combo.pack(fill="x", pady=(2, 8))
        self.exercise_type_combo.bind("<<ComboboxSelected>>", lambda e: self._on_exercise_type_changed())
        
        ttk.Label(exercise_frame, text="Root Note:").pack(anchor="w")
        self.root_note_var = tk.StringVar(value="C")
        self.root_note_combo = ttk.Combobox(exercise_frame, textvariable=self.root_note_var,
                                           values=NOTE_NAMES, state="readonly", width=28)
        self.root_note_combo.pack(fill="x", pady=(2, 8))
        
        ttk.Label(exercise_frame, text="Octave:").pack(anchor="w")
        self.octave_var = tk.IntVar(value=3)
        octave_frame = ttk.Frame(exercise_frame)
        octave_frame.pack(fill="x", pady=(2, 8))
        ttk.Spinbox(octave_frame, from_=2, to=5, textvariable=self.octave_var, width=5).pack(side="left")
        ttk.Label(octave_frame, text="(2=low E string range)", foreground=fg_dim).pack(side="left", padx=10)
        
        ttk.Label(exercise_frame, text="Pattern:").pack(anchor="w")
        self.pattern_var = tk.StringVar(value="Major")
        self.pattern_combo = ttk.Combobox(exercise_frame, textvariable=self.pattern_var,
                                         values=list(SCALES.keys()), state="readonly", width=28)
        self.pattern_combo.pack(fill="x", pady=(2, 8))
        
        ttk.Label(exercise_frame, text="Direction:").pack(anchor="w")
        self.direction_var = tk.StringVar(value="Ascending")
        self.direction_combo = ttk.Combobox(exercise_frame, textvariable=self.direction_var,
                                           values=["Ascending", "Descending", "Up and Down"],
                                           state="readonly", width=28)
        self.direction_combo.pack(fill="x", pady=(2, 8))
        
        # Control buttons
        control_frame = ttk.LabelFrame(left_col, text="Control", padding="10")
        control_frame.pack(fill="x", pady=(0, 10))
        
        self.start_btn = ttk.Button(control_frame, text="START EXERCISE",
                                   command=self.start_exercise, style="Big.TButton")
        self.start_btn.pack(fill="x", pady=(0, 8))
        
        self.stop_btn = ttk.Button(control_frame, text="STOP",
                                  command=self.stop_exercise, state="disabled")
        self.stop_btn.pack(fill="x", pady=(0, 8))
        
        self.reset_btn = ttk.Button(control_frame, text="Reset Exercise",
                                   command=self.reset_exercise, state="disabled")
        self.reset_btn.pack(fill="x")
        
        # Stats
        stats_frame = ttk.LabelFrame(left_col, text="Session Stats", padding="10")
        stats_frame.pack(fill="x", pady=(0, 10))
        
        self.stats_correct_var = tk.StringVar(value="Correct: 0")
        self.stats_wrong_var = tk.StringVar(value="Wrong: 0")
        self.stats_accuracy_var = tk.StringVar(value="Accuracy: --")
        
        ttk.Label(stats_frame, textvariable=self.stats_correct_var, foreground="#4ade80").pack(anchor="w")
        ttk.Label(stats_frame, textvariable=self.stats_wrong_var, foreground="#f87171").pack(anchor="w")
        ttk.Label(stats_frame, textvariable=self.stats_accuracy_var).pack(anchor="w")
        
        # === RIGHT COLUMN ===
        right_col = ttk.Frame(main)
        right_col.pack(side="left", fill="both", expand=True)
        
        # Status
        status_frame = ttk.LabelFrame(right_col, text="Status", padding="10")
        status_frame.pack(fill="x", pady=(0, 10))
        
        status_content = ttk.Frame(status_frame)
        status_content.pack(fill="x")
        
        self.status_light = StatusLight(status_content, size=40, bg=bg_dark)
        self.status_light.pack(side="left", padx=(0, 15))
        
        self.status_var = tk.StringVar(value="Ready - Configure and start an exercise")
        tk.Label(status_content, textvariable=self.status_var,
                font=("Arial", 16, "bold"), fg=fg, bg=bg_dark).pack(side="left", anchor="w")
        
        # Current target display
        target_frame = ttk.LabelFrame(right_col, text="Target Note", padding="10")
        target_frame.pack(fill="x", pady=(0, 10))
        
        target_content = ttk.Frame(target_frame)
        target_content.pack(fill="x")
        
        self.target_note_var = tk.StringVar(value="--")
        tk.Label(target_content, textvariable=self.target_note_var,
                font=("Arial", 72, "bold"), fg="#fbbf24", bg=bg_dark).pack(side="left", padx=(0, 30))
        
        target_info = ttk.Frame(target_content)
        target_info.pack(side="left", fill="y")
        
        self.target_midi_var = tk.StringVar(value="MIDI: --")
        ttk.Label(target_info, textvariable=self.target_midi_var, font=("Arial", 14)).pack(anchor="w")
        
        self.progress_var = tk.StringVar(value="Progress: 0 / 0")
        ttk.Label(target_info, textvariable=self.progress_var, font=("Arial", 14)).pack(anchor="w")
        
        self.last_played_var = tk.StringVar(value="Last played: --")
        ttk.Label(target_info, textvariable=self.last_played_var,
                 font=("Arial", 12), foreground=fg_dim).pack(anchor="w", pady=(10, 0))
        
        # Fretboard
        fretboard_frame = ttk.LabelFrame(right_col, text="Fretboard", padding="5")
        fretboard_frame.pack(fill="x", pady=(0, 10))
        
        self.fretboard = TrainingFretboard(fretboard_frame, frets=15)
        self.fretboard.pack()
        
        # Exercise info / Log
        log_frame = ttk.LabelFrame(right_col, text="Exercise Info", padding="5")
        log_frame.pack(fill="both", expand=True)
        
        self.log_text = tk.Text(log_frame, height=6, wrap="word",
                               bg=bg_mid, fg=fg, insertbackground=fg,
                               selectbackground=accent, font=("Consolas", 10))
        self.log_text.pack(fill="both", expand=True, side="left")
        
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(fill="y", side="right")
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # Initial setup
        self._on_exercise_type_changed()
    
    def _log(self, msg: str):
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
    
    def _browse_checkpoint(self):
        filename = filedialog.askopenfilename(
            title="Select checkpoint file",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if filename:
            self.checkpoint_var.set(filename)
    
    def _refresh_devices(self):
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
        except Exception as e:
            messagebox.showerror("Error", f"Failed to query devices: {e}")
    
    def _on_device_selected(self):
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
    
    def _sync_monitor(self):
        self.monitor_enabled = bool(self.monitor_var.get())
    
    def _on_exercise_type_changed(self):
        """Update UI based on exercise type selection."""
        ex_type = self.exercise_type_var.get()
        
        if ex_type == "Scale Practice":
            self.pattern_combo['values'] = list(SCALES.keys())
            self.pattern_var.set("Major")
            self.direction_combo.config(state="readonly")
        elif ex_type == "Arpeggio Practice":
            self.pattern_combo['values'] = list(ARPEGGIOS.keys())
            self.pattern_var.set("Major")
            self.direction_combo.config(state="readonly")
        elif ex_type in ("Find Note", "Find Octaves"):
            self.pattern_combo['values'] = ["Single Note"]
            self.pattern_var.set("Single Note")
            self.direction_combo.config(state="disabled")
        elif ex_type == "Interval Training":
            self.pattern_combo['values'] = list(INTERVALS.keys())
            self.pattern_var.set("Perfect 5th")
            self.direction_combo.config(state="disabled")
    
    def _create_exercise(self) -> Exercise:
        """Create an exercise based on current settings."""
        ex_type = self.exercise_type_var.get()
        root_name = self.root_note_var.get()
        root_idx = NOTE_NAMES.index(root_name)
        octave = self.octave_var.get()
        root_midi = root_idx + (octave + 1) * 12  # MIDI note number
        
        pattern_name = self.pattern_var.get()
        direction = self.direction_var.get()
        
        target_notes = []
        
        if ex_type == "Scale Practice":
            intervals = SCALES.get(pattern_name, [0])
            # Build one octave of the scale
            notes = [root_midi + i for i in intervals]
            notes.append(root_midi + 12)  # Add octave
            
            if direction == "Ascending":
                target_notes = notes
            elif direction == "Descending":
                target_notes = list(reversed(notes))
            else:  # Up and Down
                target_notes = notes + list(reversed(notes[:-1]))
            
            exercise_type = ExerciseType.SCALE_ASCENDING
            
        elif ex_type == "Arpeggio Practice":
            intervals = ARPEGGIOS.get(pattern_name, [0])
            notes = [root_midi + i for i in intervals]
            
            if direction == "Ascending":
                target_notes = notes
            elif direction == "Descending":
                target_notes = list(reversed(notes))
            else:
                target_notes = notes + list(reversed(notes[:-1]))
            
            exercise_type = ExerciseType.ARPEGGIO
            
        elif ex_type == "Find Note":
            # Find all positions for this note on the fretboard
            positions = self.fretboard.midi_to_positions(root_midi)
            # Extend to multiple octaves
            for offset in [-12, 12, 24]:
                positions.extend(self.fretboard.midi_to_positions(root_midi + offset))
            
            # Create target notes (same note, different positions)
            target_notes = [root_midi] * min(len(positions), 6)  # Find it 6 times
            exercise_type = ExerciseType.FIND_NOTE
            
        elif ex_type == "Find Octaves":
            # Find the same note across octaves
            target_notes = []
            for octave_offset in range(4):
                midi = root_midi + octave_offset * 12
                if 40 <= midi <= 84:  # Within guitar range
                    target_notes.append(midi)
            
            exercise_type = ExerciseType.FIND_OCTAVES
            
        elif ex_type == "Interval Training":
            interval = INTERVALS.get(pattern_name, 7)
            target_notes = [root_midi, root_midi + interval]
            exercise_type = ExerciseType.INTERVAL
        
        return Exercise(
            name=f"{root_name} {pattern_name}",
            exercise_type=exercise_type,
            root_note=root_name,
            pattern_name=pattern_name,
            target_notes=target_notes
        )
    
    def start_exercise(self):
        """Start a training exercise."""
        if self.is_running:
            return
        
        # Validate checkpoint
        checkpoint_path = Path(self.checkpoint_var.get())
        if not checkpoint_path.exists():
            messagebox.showerror("Error", f"Checkpoint not found: {checkpoint_path}")
            return
        
        # Get device
        current = self.device_combo.current()
        if current < 0 or current >= len(self.device_list):
            messagebox.showerror("Error", "Please select an audio device")
            return
        
        self.params['device_idx'], _ = self.device_list[current]
        self.params['channel_idx'] = self.channel_combo.current()
        self.params['checkpoint'] = self.checkpoint_var.get()
        
        # Create exercise
        self.current_exercise = self._create_exercise()
        
        self.is_running = True
        self._stopping = False
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.reset_btn.config(state="normal")
        
        self._log(f"Starting exercise: {self.current_exercise.name}")
        self._log(f"Notes to play: {len(self.current_exercise.target_notes)}")
        
        self._update_exercise_display()
        
        # Start audio in background thread
        threading.Thread(target=self._run_audio, daemon=True).start()
    
    def _run_audio(self):
        """Run audio capture (background thread)."""
        try:
            self._send_message('log', "Loading model...")
            model_device = self._get_model_device()
            
            checkpoint_path = Path(self.params['checkpoint'])
            self.state = self._load_checkpoint(checkpoint_path, model_device)
            
            sr = self.state["sample_rate"]
            attack = self.state["attack_samples"]
            
            pre_samples = int(round((self.params['pre_ms'] / 1000.0) * sr))
            post_samples = int(round((self.params['post_ms'] / 1000.0) * sr))
            capture_total = pre_samples + post_samples
            
            sweep_step_samples = max(1, int(round((self.params['sweep_step_ms'] / 1000.0) * sr)))
            min_silence_samples = max(1, int(round((self.params['min_silence_ms'] / 1000.0) * sr)))
            
            device_idx = self.params['device_idx']
            device_info = sd.query_devices(device_idx)
            max_channels = device_info['max_input_channels']
            ch = self.params['channel_idx']
            
            self.capture_state = {
                'mode': 'armed',
                'pre_buffer': np.zeros((pre_samples, max_channels), dtype=np.float32) if pre_samples > 0 else None,
                'pre_idx': 0,
                'recording_parts': [],
                'recorded_post': 0,
                'silence_ctr': 0,
                'sr': sr,
                'ch': ch,
                'pre_samples': pre_samples,
                'post_samples': post_samples,
                'capture_total': capture_total,
                'min_silence_samples': min_silence_samples,
                'sweep_step_samples': sweep_step_samples,
                'attack': attack,
            }
            
            self.audio_queue = queue.Queue()
            self.monitor_audio = None
            
            def audio_callback(indata, frames, time_info, status):
                if indata.ndim == 2:
                    mono = indata[:, ch].astype(np.float32, copy=True)
                else:
                    mono = indata.astype(np.float32, copy=True)
                
                if self.monitor_enabled:
                    self.monitor_audio = mono
                
                self.audio_queue.put((indata.copy(), mono))
            
            def output_callback(outdata, frames, time_info, status):
                if self._stopping or not self.monitor_enabled or self.monitor_audio is None:
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
                channels=max_channels,
                samplerate=sr,
                blocksize=ANALYSIS_HOP,
                dtype='float32',
                callback=audio_callback
            )
            
            if self.monitor_enabled:
                try:
                    self.output_stream = sd.OutputStream(
                        samplerate=sr,
                        blocksize=ANALYSIS_HOP,
                        channels=2,
                        dtype='float32',
                        callback=output_callback
                    )
                    self.output_stream.start()
                except:
                    self.output_stream = None
            
            self.stream.start()
            
            self._send_message('status', "Play the highlighted note!")
            self._send_message('state', 'ready')
            
            self.root.after(10, self._process_audio_queue)
            
        except Exception as e:
            self._send_message('error', f"Failed to start: {e}")
            self.is_running = False
    
    def _process_audio_queue(self):
        """Process audio from queue (main thread)."""
        if not self.is_running:
            return
        
        while not self.audio_queue.empty():
            try:
                indata, mono = self.audio_queue.get_nowait()
            except queue.Empty:
                break
            
            self._process_audio_block(indata, mono)
        
        if self.is_running:
            self.root.after(10, self._process_audio_queue)
    
    def _process_audio_block(self, indata, mono):
        """Process an audio block."""
        cs = self.capture_state
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        frames = mono.shape[0]
        
        # Cooldown to prevent double-detection
        if self.detection_cooldown > 0:
            self.detection_cooldown -= frames
        
        if cs['mode'] == 'armed':
            if cs['pre_buffer'] is not None:
                self._update_prebuffer(indata)
            
            if peak >= self.params['trigger_threshold'] and self.detection_cooldown <= 0:
                pre_audio = self._get_prebuffer_channel()
                cs['recording_parts'] = [pre_audio, mono.copy()]
                cs['recorded_post'] = frames
                cs['mode'] = 'capturing'
                self.status_light.set_state("listening")
                
        elif cs['mode'] == 'capturing':
            cs['recording_parts'].append(mono.copy())
            cs['recorded_post'] += frames
            
            if cs['recorded_post'] >= cs['post_samples']:
                cap = np.concatenate(cs['recording_parts']).astype(np.float32)
                
                if cap.shape[0] > cs['capture_total']:
                    cap = cap[:cs['capture_total']]
                elif cap.shape[0] < cs['capture_total']:
                    cap = np.pad(cap, (0, cs['capture_total'] - cap.shape[0]), mode='constant')
                
                self._process_captured_audio(cap)
                
                cs['mode'] = 'waiting_silence'
                cs['silence_ctr'] = 0
                
        elif cs['mode'] == 'waiting_silence':
            if peak <= self.params['silence_threshold']:
                cs['silence_ctr'] += frames
            else:
                cs['silence_ctr'] = 0
            
            if cs['silence_ctr'] >= cs['min_silence_samples']:
                cs['mode'] = 'armed'
                cs['recording_parts'] = []
                cs['recorded_post'] = 0
                cs['pre_idx'] = 0
                
                if not self.current_exercise.is_complete:
                    self.status_light.set_state("ready")
    
    def _process_captured_audio(self, captured):
        """Process captured audio and check against target."""
        cs = self.capture_state
        
        # Run inference
        best_start, best_window, best_topk = self._sweep_best_window(
            captured=captured,
            sweep_step_samples=cs['sweep_step_samples'],
            topk=5
        )
        
        if not best_topk:
            return
        
        midi, name, prob, reason, was_ambiguous, f0_hz = self._decide_with_margin_and_pitch(
            best_window=best_window,
            best_topk=best_topk,
            sr=cs['sr'],
            margin=self.params['margin']
        )
        
        self.last_detected_midi = midi
        self.last_played_var.set(f"Last played: {name} (MIDI {midi})")
        
        # Check against exercise
        if self.current_exercise and not self.current_exercise.is_complete:
            target = self.current_exercise.current_target
            
            # For "Find Note" exercises, accept any octave of the target
            if self.current_exercise.exercise_type == ExerciseType.FIND_NOTE:
                correct = (midi % 12) == (target % 12)
            else:
                correct = (midi == target)
            
            self.current_exercise.record_attempt(correct)
            
            if correct:
                self._log(f"✓ Correct! Played {name}")
                self.status_light.set_state("correct")
                self.current_exercise.advance()
                
                # Set cooldown to prevent double-detection
                self.detection_cooldown = int(cs['sr'] * 0.3)
                
                if self.current_exercise.is_complete:
                    self._exercise_complete()
                else:
                    self.root.after(500, self._update_exercise_display)
            else:
                target_name = self._midi_to_name(target)
                self._log(f"✗ Wrong - played {name}, expected {target_name}")
                self.status_light.set_state("wrong")
                self.root.after(300, lambda: self.status_light.set_state("ready"))
            
            self._update_stats()
    
    def _exercise_complete(self):
        """Handle exercise completion."""
        self._log("🎉 Exercise complete!")
        self.status_var.set("Exercise Complete!")
        self.status_light.set_state("complete")
        self.target_note_var.set("✓")
        self.fretboard.clear_notes()
    
    def _update_exercise_display(self):
        """Update the display for current exercise state."""
        if not self.current_exercise:
            return
        
        ex = self.current_exercise
        
        if ex.is_complete:
            return
        
        target = ex.current_target
        target_name = self._midi_to_name(target)
        
        self.target_note_var.set(target_name)
        self.target_midi_var.set(f"MIDI: {target}")
        self.progress_var.set(f"Progress: {ex.current_index + 1} / {len(ex.target_notes)}")
        self.status_var.set(f"Play: {target_name}")
        
        # Update fretboard
        if ex.exercise_type in (ExerciseType.FIND_NOTE, ExerciseType.FIND_OCTAVES):
            self.fretboard.show_target_note(target, 'target')
        else:
            self.fretboard.show_sequence(ex.target_notes, ex.current_index)
    
    def _update_stats(self):
        """Update statistics display."""
        if not self.current_exercise:
            return
        
        ex = self.current_exercise
        self.stats_correct_var.set(f"Correct: {ex.correct_count}")
        self.stats_wrong_var.set(f"Wrong: {ex.wrong_count}")
        
        total = ex.correct_count + ex.wrong_count
        if total > 0:
            accuracy = (ex.correct_count / total) * 100
            self.stats_accuracy_var.set(f"Accuracy: {accuracy:.1f}%")
        else:
            self.stats_accuracy_var.set("Accuracy: --")
    
    def stop_exercise(self):
        """Stop the current exercise."""
        if not self.is_running:
            return
        
        self.is_running = False
        self._stopping = True
        
        if self.output_stream:
            try:
                self.output_stream.stop()
                self.output_stream.close()
            except:
                pass
            self.output_stream = None
        
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
            self.stream = None
        
        self.capture_state['mode'] = 'stopped'
        
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Stopped")
        self.status_light.set_state("stopped")
        
        self._stopping = False
        self._log("Exercise stopped")
    
    def reset_exercise(self):
        """Reset the current exercise."""
        if self.current_exercise:
            self.current_exercise.current_index = 0
            self.current_exercise.correct_count = 0
            self.current_exercise.wrong_count = 0
            self._update_exercise_display()
            self._update_stats()
            self._log("Exercise reset")
    
    def _send_message(self, msg_type, data):
        self.message_queue.put((msg_type, data))
    
    def _update_message_display(self):
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()
                
                if msg_type == 'status':
                    self.status_var.set(data)
                elif msg_type == 'state':
                    self.status_light.set_state(data)
                elif msg_type == 'log':
                    self._log(data)
                elif msg_type == 'error':
                    messagebox.showerror("Error", data)
                    self.stop_exercise()
        except queue.Empty:
            pass
        
        self.root.after(50, self._update_message_display)
    
    def on_closing(self):
        if self.is_running:
            self.stop_exercise()
        self.root.destroy()
    
    # ============================
    # AUDIO HELPER METHODS
    # ============================
    
    def _update_prebuffer(self, block):
        cs = self.capture_state
        pre_buffer = cs['pre_buffer']
        pre_samples = cs['pre_samples']
        pre_idx = cs['pre_idx']
        
        frames = block.shape[0]
        if frames <= 0:
            return
        
        if frames > pre_samples:
            block = block[-pre_samples:]
            frames = block.shape[0]
        
        end = pre_idx + frames
        if end <= pre_samples:
            pre_buffer[pre_idx:end, :] = block.astype(np.float32)
        else:
            first = pre_samples - pre_idx
            pre_buffer[pre_idx:, :] = block[:first].astype(np.float32)
            remaining = frames - first
            pre_buffer[:remaining, :] = block[first:].astype(np.float32)
        
        cs['pre_idx'] = (pre_idx + frames) % pre_samples
    
    def _get_prebuffer_channel(self):
        cs = self.capture_state
        pre_buffer = cs['pre_buffer']
        ch = cs['ch']
        pre_samples = cs['pre_samples']
        pre_idx = cs['pre_idx']
        
        if pre_buffer is None or pre_samples <= 0:
            return np.zeros((0,), dtype=np.float32)
        
        if pre_idx == 0:
            return pre_buffer[:, ch].astype(np.float32, copy=True)
        
        return np.concatenate([
            pre_buffer[pre_idx:, ch],
            pre_buffer[:pre_idx, ch]
        ]).astype(np.float32)
    
    # ============================
    # MODEL & INFERENCE METHODS
    # ============================
    
    def _get_model_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _load_checkpoint(self, path: Path, model_device: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        
        mel_params = ckpt["mel_params"]
        num_classes = int(ckpt["num_classes"])
        n_mels = int(mel_params["n_mels"])
        
        model = NoteCNN(num_classes=num_classes, n_mels=n_mels, dropout_p=0.0)
        
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt["model_state_dict"])
        
        model.to(model_device)
        model.eval()
        
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=int(ckpt["sample_rate"]),
            n_fft=int(mel_params["n_fft"]),
            hop_length=int(mel_params["hop_length"]),
            n_mels=int(mel_params["n_mels"]),
            f_min=float(mel_params["f_min"]),
            f_max=float(mel_params["f_max"]),
        )
        db = torchaudio.transforms.AmplitudeToDB()
        
        idx_to_midi = ckpt["idx_to_midi"]
        if isinstance(idx_to_midi, dict):
            idx_to_midi = {int(k): int(v) for k, v in idx_to_midi.items()}
        
        return {
            "model": model,
            "model_device": model_device,
            "mel": mel,
            "db": db,
            "idx_to_midi": idx_to_midi,
            "sample_rate": int(ckpt["sample_rate"]),
            "attack_samples": int(ckpt["attack_samples"]),
            "num_frames": int(mel_params["num_frames"]),
        }
    
    def _waveform_to_logmel(self, wav_1d):
        state = self.state
        wav = wav_1d.astype(np.float32)
        wav = wav - float(np.mean(wav))
        
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        
        spec = state["mel"](wav_t)
        spec_db = state["db"](spec)
        
        mean = spec_db.mean(dim=2, keepdim=True)
        std = spec_db.std(dim=2, keepdim=True) + 1e-6
        spec_db = (spec_db - mean) / std
        
        num_frames = state["num_frames"]
        _, _, t = spec_db.shape
        if t < num_frames:
            spec_db = torch.nn.functional.pad(spec_db, (0, num_frames - t))
        elif t > num_frames:
            spec_db = spec_db[:, :, :num_frames]
        
        return spec_db.unsqueeze(0)
    
    def _peak_normalize(self, window):
        w = window.astype(np.float32)
        peak = float(np.max(np.abs(w))) if w.size else 0.0
        if peak > 0:
            w = (w / peak) * 0.95
        return w
    
    def _predict_topk_window(self, window_1d, topk, peak_normalize=True):
        w = self._peak_normalize(window_1d) if peak_normalize else window_1d.astype(np.float32)
        
        spec = self._waveform_to_logmel(w)
        spec = spec.to(self.state["model_device"], dtype=torch.float32)
        
        with torch.no_grad():
            logits = self.state["model"](spec)
            probs = torch.softmax(logits, dim=1)
        
        probs_cpu = probs.detach().to("cpu")
        k = min(int(topk), probs_cpu.shape[1])
        vals, idxs = torch.topk(probs_cpu, k=k, dim=1)
        
        out = []
        for p, idx in zip(vals[0].tolist(), idxs[0].tolist()):
            midi = int(self.state["idx_to_midi"][int(idx)])
            out.append((midi, self._midi_to_name(midi), float(p)))
        
        return out, probs_cpu
    
    def _sweep_best_window(self, captured, sweep_step_samples, topk):
        attack = self.state["attack_samples"]
        
        if captured.shape[0] < attack:
            padded = np.zeros(attack, dtype=np.float32)
            padded[:captured.shape[0]] = captured.astype(np.float32)
            captured = padded
        
        best_conf = -1.0
        best_start = 0
        best_window = captured[:attack].copy()
        best_topk = []
        
        last_start = max(0, captured.shape[0] - attack)
        step = max(1, int(sweep_step_samples))
        
        for start in range(0, last_start + 1, step):
            window = captured[start:start + attack]
            
            topk_list, _ = self._predict_topk_window(window, topk, peak_normalize=True)
            if not topk_list:
                continue
            
            conf = topk_list[0][2]
            if conf > best_conf:
                best_conf = conf
                best_start = start
                best_window = window.copy()
                best_topk = topk_list
        
        return best_start, best_window, best_topk
    
    def _decide_with_margin_and_pitch(self, best_window, best_topk, sr, margin):
        if not best_topk:
            return -1, "UNKNOWN", 0.0, "no_candidates", True, None
        
        m1, n1, p1 = best_topk[0]
        if len(best_topk) < 2:
            return m1, n1, p1, "only_one_candidate", False, None
        
        m2, n2, p2 = best_topk[1]
        gap = p1 - p2
        
        if gap >= margin:
            return m1, n1, p1, f"margin_ok(gap={gap:.3f})", False, None
        
        if abs(m1 - m2) == 1:
            chosen, f0 = self._pitch_tiebreak(best_window, sr, m1, m2)
            if chosen is not None:
                if chosen == m1:
                    return m1, n1, p1, "pitch->top1", True, f0
                return m2, n2, p2, "pitch->top2", True, f0
            return m1, n1, p1, "ambiguous_adjacent", True, None
        
        return m1, n1, p1, "ambiguous_non_adjacent", True, None
    
    def _pitch_tiebreak(self, window, sr, midi_a, midi_b):
        f0 = self._estimate_f0(window, sr)
        if f0 is None:
            return None, None
        
        fa = self._midi_to_freq(midi_a)
        fb = self._midi_to_freq(midi_b)
        
        da = abs(f0 - fa)
        db = abs(f0 - fb)
        
        return (midi_a if da <= db else midi_b), float(f0)
    
    def _estimate_f0(self, x, sr, fmin=70.0, fmax=1200.0):
        if x.size < int(sr * 0.02):
            return None
        
        y = x.astype(np.float32)
        y = y - float(np.mean(y))
        y = self._peak_normalize(y)
        
        win = np.hanning(y.size).astype(np.float32)
        y = y * win
        
        r = np.correlate(y, y, mode="full")[y.size - 1:]
        
        lag_min = int(sr / fmax)
        lag_max = int(sr / fmin)
        if lag_max <= lag_min + 2 or lag_max >= r.size:
            return None
        
        seg = r[lag_min:lag_max]
        if seg.size == 0:
            return None
        
        i = int(np.argmax(seg))
        best_lag = lag_min + i
        if best_lag <= 0:
            return None
        
        peak = float(seg[i])
        if peak <= 0:
            return None
        
        return float(sr / best_lag)
    
    def _midi_to_freq(self, midi):
        return 440.0 * (2.0 ** ((midi - 69) / 12.0))
    
    def _midi_to_name(self, midi):
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        note = names[midi % 12]
        octave = (midi // 12) - 1
        return f"{note}{octave}"


# ============================
# MAIN
# ============================

def main():
    root = tk.Tk()
    root.minsize(1100, 750)
    app = FretTrainerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()