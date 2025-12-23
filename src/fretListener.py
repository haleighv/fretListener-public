#!/usr/bin/env python3
"""
fretListener.py - Real-time Guitar Position Inference

A unified GUI application for real-time guitar note and string inference.

Modes:
  - note: Note classification only (position 0-N)
  - note+string: Combined note + string classification for accurate position

Features:
  - Queue-based audio capture for thread-safe processing
  - Audio monitoring (hear what you're playing)
  - Modern dark-mode UI with 12-fret fretboard display
  - Configurable trigger and inference parameters
  - Audio preprocessing matching training data (DC offset, HPF, fades)

Dependencies:
  pip install sounddevice soundfile numpy torch torchaudio scipy
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import sys

import numpy as np
import sounddevice as sd
import torch
import torchaudio
from scipy.io import wavfile

# Add paths for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
COMMON_DIR = REPO_ROOT / "common"

for p in [str(SRC_DIR), str(COMMON_DIR), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Import models
from note_model import NoteCNN
from string_model import StringCNN
from disambiguator_model import DisambiguatorCNN, STRING_OPEN_MIDI, ALL_PAIRS

# Import common utilities
from common.audio import (
    compute_rms,
    save_wav_int16,
    DEFAULT_SAMPLE_RATE,
)
from common.audio_inference import (
    peak_normalize,
    waveform_to_logmel_torch,
    create_mel_transforms,
)
from common.guitar import (
    STANDARD_TUNING_LIST,
    STRING_NAMES_BY_INDEX,
    STRING_NAMES_TOP_TO_BOTTOM,
    midi_to_note_name,
    string_idx_to_num,
    string_num_to_idx,
)


# ============================
# VERSION
# ============================

VERSION = "3.0"


# ============================
# INFERENCE MODES
# ============================

MODE_NOTE_ONLY = "note"
MODE_NOTE_STRING = "note+string"
MODE_NOTE_DISAMBIG = "note+disambig"

MODES = [MODE_NOTE_ONLY, MODE_NOTE_STRING, MODE_NOTE_DISAMBIG]
MODE_LABELS = {
    MODE_NOTE_ONLY: "Note Only",
    MODE_NOTE_STRING: "Note + String",
    MODE_NOTE_DISAMBIG: "Note + Disambig",
}


# ============================
# AUDIO CONFIG
# ============================

ANALYSIS_HOP = 1024  # Block size for audio processing


# ============================
# STATUS LIGHT WIDGET
# ============================

class StatusLight(tk.Canvas):
    """A status indicator light widget."""
    
    COLORS = {
        "armed": "#4ade80",       # Green - ready to detect
        "capturing": "#facc15",   # Yellow - capturing audio
        "processing": "#f97316",  # Orange - running inference
        "detected": "#3b82f6",    # Blue - note detected
        "stopped": "#4b5563",     # Dark gray - not running
        "error": "#ef4444",       # Red - error state
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
        """Draw the status light."""
        self.delete("all")
        padding = 2
        
        color = self.COLORS.get(self.state, self.COLORS["stopped"])
        
        # Pulse effect for certain states
        if self.state in ("armed", "capturing") and not self.pulse_on:
            color = self._dim_color(color, 0.6)
        
        # Draw outer ring
        self.create_oval(padding, padding,
                        self.size + padding, self.size + padding,
                        fill=color, outline="#1e1e1e", width=2)
    
    def _dim_color(self, hex_color: str, factor: float) -> str:
        """Dim a hex color by a factor."""
        try:
            hex_color = hex_color.lstrip('#')
            r = int(int(hex_color[0:2], 16) * factor)
            g = int(int(hex_color[2:4], 16) * factor)
            b = int(int(hex_color[4:6], 16) * factor)
            return f"#{r:02x}{g:02x}{b:02x}"
        except:
            return hex_color
    
    def set_state(self, state: str):
        """Set the status light state."""
        if self.pulse_job:
            self.after_cancel(self.pulse_job)
            self.pulse_job = None
        
        self.state = state
        self.pulse_on = True
        
        if state in ("armed", "capturing"):
            self._start_pulse()
        else:
            self._draw()
    
    def _start_pulse(self):
        """Start pulsing animation."""
        self.pulse_on = not self.pulse_on
        self._draw()
        self.pulse_job = self.after(500, self._start_pulse)


# ============================
# FRETBOARD WIDGET (12 FRETS)
# ============================

class GuitarFretboard(tk.Canvas):
    """
    Visual guitar fretboard widget (12 frets).
    
    Display layout:
    - Row 0 (top) = E4 (high E, string 1)
    - Row 5 (bottom) = E2 (low E, string 6)
    
    Internal string indexing:
    - string_idx 0 = E2 (low E, bottom of display)
    - string_idx 5 = E4 (high E, top of display)
    """
    
    STANDARD_TUNING = STANDARD_TUNING_LIST
    STRING_NAMES_TOP_TO_BOTTOM = STRING_NAMES_TOP_TO_BOTTOM

    def __init__(self, parent, frets: int = 12, **kwargs):
        super().__init__(parent, **kwargs)
        self.frets = int(frets)

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

        # Draw frets
        for fret in range(self.frets + 1):
            x = 45 + fret * self.fret_spacing
            y_start = 25
            y_end = 25 + self.string_spacing * 5

            if fret == 0:
                self.create_line(x, y_start, x, y_end, width=5, fill="#d4a574")
            else:
                self.create_line(x, y_start, x, y_end, width=2, fill="#8b7355")

        # Fret markers
        for fret in [3, 5, 7, 9]:
            if fret <= self.frets:
                x = 45 + fret * self.fret_spacing - self.fret_spacing // 2
                marker_y = 25 + self.string_spacing * 2.5
                self.create_oval(x - 4, marker_y - 4, x + 4, marker_y + 4,
                               fill="#555", outline="")
        
        # Double dot at 12
        if 12 <= self.frets:
            x = 45 + 12 * self.fret_spacing - self.fret_spacing // 2
            marker_y1 = 25 + self.string_spacing * 1.5
            marker_y2 = 25 + self.string_spacing * 3.5
            self.create_oval(x - 4, marker_y1 - 4, x + 4, marker_y1 + 4,
                           fill="#555", outline="")
            self.create_oval(x - 4, marker_y2 - 4, x + 4, marker_y2 + 4,
                           fill="#555", outline="")

        # Draw strings (top=E4 thin, bottom=E2 thick)
        for s in range(6):
            y = 25 + s * self.string_spacing
            x_start = 45
            x_end = 45 + self.frets * self.fret_spacing

            thickness = 1 + s * 0.4
            self.create_line(x_start, y, x_end, y, width=thickness, fill="#c0c0c0")

            self.create_text(18, y, text=self.STRING_NAMES_TOP_TO_BOTTOM[s],
                           font=("Arial", 9), fill="#888")

    def clear_notes(self):
        """Clear all active notes."""
        self.delete("note")

    def midi_to_fretboard(self, midi_note: int) -> list:
        """Convert MIDI note to all possible (string_idx, fret) positions.
        
        Returns positions where string_idx 0 = low E (bottom), 5 = high E (top).
        """
        positions = []
        for string_idx, open_note in enumerate(self.STANDARD_TUNING):
            fret = midi_note - open_note
            if 0 <= fret <= self.frets:
                positions.append((string_idx, fret))
        return positions

    def show_note(self, midi_note: int):
        """Highlight ALL positions where a MIDI note can be played."""
        self.clear_notes()
        positions = self.midi_to_fretboard(midi_note)
        
        for string_idx, fret in positions:
            # Convert string_idx to display row
            display_row = 5 - string_idx
            self._draw_note(display_row, fret)

    def show_position(self, string_idx: int, fret: int):
        """
        Show a note at the given position.
        
        Args:
            string_idx: 0-5 (0=low E, 5=high E)
            fret: 0-12
        """
        self.clear_notes()
        # Convert string_idx to display row
        # string_idx 0 (low E) -> display row 5 (bottom)
        # string_idx 5 (high E) -> display row 0 (top)
        display_row = 5 - string_idx
        self._draw_note(display_row, fret)

    def _draw_note(self, display_row: int, fret: int):
        """Draw a note marker at display position."""
        if fret < 0 or fret > self.frets:
            return
        if display_row < 0 or display_row >= 6:
            return

        y = 25 + display_row * self.string_spacing
        
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
# TOOLTIP
# ============================

class ToolTip:
    """Tooltip widget for hover help."""
    
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)
    
    def _show(self, event=None):
        if self.tip_window:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(tw, text=self.text, justify='left',
                        background="#2d2d2d", foreground="#e0e0e0",
                        relief='solid', borderwidth=1,
                        font=("Arial", 9), padx=8, pady=4)
        label.pack()
    
    def _hide(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


# ============================
# MAIN APPLICATION
# ============================

class FretListenerGUI:
    """Main GUI application for guitar inference."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"fretListener v{VERSION}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.configure(bg="#1e1e1e")

        # State
        self.is_running = False
        self._stopping = False
        self.stream = None
        self.output_stream = None
        self.state = None  # Model state after loading
        self.audio_queue = None
        self.monitor_audio = None
        self.capture_state = {}
        
        # Message queue for thread-safe UI updates
        self.message_queue = queue.Queue()

        # Available devices
        self.device_list = []
        self.channel_list = []

        # Monitor toggle (cached for audio thread)
        self.monitor_enabled = True

        # Inference mode
        self.inference_mode = MODE_NOTE_STRING

        # Default parameters
        self.params = {
            'note_checkpoint': 'artifacts/models/note_cnn_best.pt',
            'string_checkpoint': 'artifacts/models/string_cnn_best.pt',
            'disambig_dir': 'artifacts/models/disambig_batch',
            'trigger_threshold': 0.01,
            'pre_ms': 50.0,
            'post_ms': 500.0,
            'silence_threshold': 0.006,
            'min_silence_ms': 80.0,
            'sweep_step_ms': 5.0,
            'margin': 0.10,
            'topk': 5,
            'save_dir': '',
            'device_idx': None,
            'channel_idx': 0
        }

        self._setup_ui()
        self._refresh_devices()
        self._update_message_display()

    def _to_display_string_num(self, internal_string_idx: int) -> int:
        """Convert internal index (0=low E) to display number (6=low E)."""
        return 6 - int(internal_string_idx)

    def _string_open_name(self, internal_string_idx: int) -> str:
        """Get open string note name from internal index."""
        si = int(internal_string_idx)
        return STRING_NAMES_BY_INDEX[si] if 0 <= si < 6 else f"S{si}"

    def _setup_ui(self):
        """Set up the modern dark-mode UI."""
        style = ttk.Style()
        style.theme_use('clam')

        # Color scheme
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

        style.configure("TEntry", fieldbackground=bg_mid, foreground=fg)
        style.configure("TSpinbox", fieldbackground=bg_mid, foreground=fg)
        style.configure("TRadiobutton", background=bg_dark, foreground=fg)

        style.configure("Start.TButton", font=("Arial", 14, "bold"), padding=(20, 15))
        style.configure("Stop.TButton", font=("Arial", 14, "bold"), padding=(20, 15))
        style.configure("Action.TButton", font=("Arial", 11), padding=(15, 10))

        # Main container
        main = ttk.Frame(self.root, padding="15")
        main.pack(fill="both", expand=True)

        # === LEFT COLUMN ===
        left_col = ttk.Frame(main)
        left_col.pack(side="left", fill="y", padx=(0, 15))

        # Mode selection
        mode_frame = ttk.LabelFrame(left_col, text="Inference Mode", padding="10")
        mode_frame.pack(fill="x", pady=(0, 10))

        self.mode_var = tk.StringVar(value=MODE_NOTE_STRING)
        for mode in MODES:
            rb = ttk.Radiobutton(mode_frame, text=MODE_LABELS[mode],
                                variable=self.mode_var, value=mode,
                                command=self._on_mode_change)
            rb.pack(anchor="w", pady=2)
            if mode == MODE_NOTE_ONLY:
                ToolTip(rb, "Use only the note model.\nFret position from position class.")
            else:
                ToolTip(rb, "Use both note and string models.\nMore accurate fret detection.")

        # Models section
        model_frame = ttk.LabelFrame(left_col, text="Models", padding="10")
        model_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(model_frame, text="Note checkpoint:").pack(anchor="w")
        self.note_checkpoint_var = tk.StringVar(value=self.params['note_checkpoint'])
        note_ckpt_frame = ttk.Frame(model_frame)
        note_ckpt_frame.pack(fill="x", pady=(2, 8))
        ttk.Entry(note_ckpt_frame, textvariable=self.note_checkpoint_var, width=30).pack(side="left", fill="x", expand=True)
        ttk.Button(note_ckpt_frame, text="Browse", command=self._browse_note_checkpoint, style="Action.TButton").pack(side="left", padx=(5, 0))

        self.string_model_frame = ttk.Frame(model_frame)
        self.string_model_frame.pack(fill="x")
        ttk.Label(self.string_model_frame, text="String checkpoint:").pack(anchor="w")
        self.string_checkpoint_var = tk.StringVar(value=self.params['string_checkpoint'])
        string_ckpt_frame = ttk.Frame(self.string_model_frame)
        string_ckpt_frame.pack(fill="x", pady=(2, 0))
        ttk.Entry(string_ckpt_frame, textvariable=self.string_checkpoint_var, width=30).pack(side="left", fill="x", expand=True)
        ttk.Button(string_ckpt_frame, text="Browse", command=self._browse_string_checkpoint, style="Action.TButton").pack(side="left", padx=(5, 0))

        # Disambiguator model frame (for note+disambig mode)
        self.disambig_model_frame = ttk.Frame(model_frame)
        ttk.Label(self.disambig_model_frame, text="Disambig directory:").pack(anchor="w")
        self.disambig_dir_var = tk.StringVar(value=self.params.get('disambig_dir', ''))
        disambig_dir_frame = ttk.Frame(self.disambig_model_frame)
        disambig_dir_frame.pack(fill="x", pady=(2, 0))
        ttk.Entry(disambig_dir_frame, textvariable=self.disambig_dir_var, width=30).pack(side="left", fill="x", expand=True)
        ttk.Button(disambig_dir_frame, text="Browse", command=self._browse_disambig_dir, style="Action.TButton").pack(side="left", padx=(5, 0))

        # Audio Device
        device_frame = ttk.LabelFrame(left_col, text="Audio Device", padding="10")
        device_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(device_frame, text="Device:").pack(anchor="w")
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(device_frame, textvariable=self.device_var, state="readonly", width=30)
        self.device_combo.pack(fill="x", pady=(2, 8))
        self.device_combo.bind("<<ComboboxSelected>>", lambda e: self._on_device_selected())

        ttk.Label(device_frame, text="Channel:").pack(anchor="w")
        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(device_frame, textvariable=self.channel_var, state="readonly", width=30)
        self.channel_combo.pack(fill="x", pady=(2, 8))

        ttk.Button(device_frame, text="Refresh", command=self._refresh_devices, style="Action.TButton").pack(anchor="e", pady=(5, 0))

        # Trigger Settings
        trigger_frame = ttk.LabelFrame(left_col, text="Trigger Settings", padding="10")
        trigger_frame.pack(fill="x", pady=(0, 10))

        self.param_vars = {}
        param_config = [
            ('trigger_threshold', 'Trigger:', 0.001, 0.1, 0.001),
            ('pre_ms', 'Pre (ms):', 0, 200, 10),
            ('post_ms', 'Post (ms):', 100, 2000, 50),
            ('silence_threshold', 'Silence:', 0.001, 0.05, 0.001),
            ('min_silence_ms', 'Re-arm (ms):', 10, 500, 10),
        ]

        for key, label, min_val, max_val, step in param_config:
            row = ttk.Frame(trigger_frame)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=12).pack(side="left")
            var = tk.DoubleVar(value=self.params[key])
            self.param_vars[key] = var
            spinbox = ttk.Spinbox(row, from_=min_val, to=max_val, increment=step,
                                 textvariable=var, width=10)
            spinbox.pack(side="left", padx=(5, 0))

        # Monitor checkbox
        monitor_frame = ttk.Frame(left_col)
        monitor_frame.pack(fill="x", pady=(0, 10))
        self.monitor_var = tk.BooleanVar(value=True)
        self.monitor_var.trace_add("write", lambda *a: self._sync_monitor_enabled())
        ttk.Checkbutton(monitor_frame, text="Enable audio monitoring", variable=self.monitor_var).pack(anchor="w")

        # Save captures
        save_frame = ttk.LabelFrame(left_col, text="Save Captures", padding="10")
        save_frame.pack(fill="x", pady=(0, 10))

        self.save_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(save_frame, text="Save captured audio", variable=self.save_enabled_var).pack(anchor="w")

        self.save_dir_var = tk.StringVar(value="artifacts/captures")
        save_dir_row = ttk.Frame(save_frame)
        save_dir_row.pack(fill="x", pady=(5, 0))
        ttk.Entry(save_dir_row, textvariable=self.save_dir_var, width=25).pack(side="left", fill="x", expand=True)
        ttk.Button(save_dir_row, text="Browse", command=self._browse_save_dir).pack(side="left", padx=(5, 0))

        # Control buttons
        control_frame = ttk.Frame(left_col)
        control_frame.pack(fill="x", pady=(10, 0))

        self.start_btn = tk.Button(control_frame, text="START", command=self.start_inference,
                                  bg="#4ade80", fg="#000000", font=("Arial", 14, "bold"),
                                  relief="flat", padx=20, pady=10)
        self.start_btn.pack(fill="x", pady=(0, 5))

        self.stop_btn = tk.Button(control_frame, text="STOP", command=self.stop_inference,
                                 bg="#ef4444", fg="#ffffff", font=("Arial", 14, "bold"),
                                 relief="flat", padx=20, pady=10, state="disabled")
        self.stop_btn.pack(fill="x")

        # === RIGHT COLUMN ===
        right_col = ttk.Frame(main)
        right_col.pack(side="left", fill="both", expand=True)

        # Status row
        status_row = ttk.Frame(right_col)
        status_row.pack(fill="x", pady=(0, 10))

        self.status_light = StatusLight(status_row, size=40, bg="#1e1e1e")
        self.status_light.pack(side="left", padx=(0, 15))

        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_row, textvariable=self.status_var, font=("Arial", 18, "bold"))
        status_label.pack(side="left", fill="x", expand=True)

        # Detection display
        detect_frame = ttk.LabelFrame(right_col, text="Detection", padding="10")
        detect_frame.pack(fill="x", pady=(0, 10))

        self.note_display_var = tk.StringVar(value="--")
        note_label = tk.Label(detect_frame, textvariable=self.note_display_var,
                             font=("Arial", 48, "bold"), fg="#4fc3f7", bg="#1e1e1e")
        note_label.pack()

        self.confidence_var = tk.StringVar(value="")
        ttk.Label(detect_frame, textvariable=self.confidence_var, font=("Arial", 12)).pack()

        self.midi_var = tk.StringVar(value="")
        ttk.Label(detect_frame, textvariable=self.midi_var, font=("Arial", 11), foreground=fg_dim).pack()

        self.reason_var = tk.StringVar(value="")
        ttk.Label(detect_frame, textvariable=self.reason_var, font=("Arial", 10), foreground=fg_dim).pack()

        # Fretboard
        fb_frame = ttk.LabelFrame(right_col, text="Fretboard", padding="5")
        fb_frame.pack(fill="x", pady=(0, 10))

        self.fretboard = GuitarFretboard(fb_frame, frets=12)
        self.fretboard.pack()

        # Log
        log_frame = ttk.LabelFrame(right_col, text="Log", padding="5")
        log_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(log_frame, height=10, wrap="word",
                               bg=bg_mid, fg=fg, insertbackground=fg,
                               selectbackground=accent, font=("Consolas", 10))
        self.log_text.pack(fill="both", expand=True, side="left")

        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(fill="y", side="right")
        self.log_text.config(yscrollcommand=scrollbar.set)

        # Initial mode update
        self._on_mode_change()

    def _on_mode_change(self):
        """Handle inference mode change."""
        mode = self.mode_var.get()
        self.inference_mode = mode
        
        # Hide all optional model frames first
        self.string_model_frame.pack_forget()
        self.disambig_model_frame.pack_forget()
        
        # Show relevant frame based on mode
        if mode == MODE_NOTE_STRING:
            self.string_model_frame.pack(fill="x")
        elif mode == MODE_NOTE_DISAMBIG:
            self.disambig_model_frame.pack(fill="x")

    def _sync_monitor_enabled(self):
        """Sync monitor enabled state."""
        self.monitor_enabled = bool(self.monitor_var.get())

    def _browse_note_checkpoint(self):
        """Browse for note checkpoint."""
        filename = filedialog.askopenfilename(
            title="Select NOTE checkpoint",
            filetypes=[("PyTorch", "*.pt"), ("All", "*.*")]
        )
        if filename:
            self.note_checkpoint_var.set(filename)

    def _browse_string_checkpoint(self):
        """Browse for string checkpoint."""
        filename = filedialog.askopenfilename(
            title="Select STRING checkpoint",
            filetypes=[("PyTorch", "*.pt"), ("All", "*.*")]
        )
        if filename:
            self.string_checkpoint_var.set(filename)

    def _browse_disambig_dir(self):
        """Browse for disambiguator models directory."""
        dirname = filedialog.askdirectory(title="Select disambiguator models directory")
        if dirname:
            self.disambig_dir_var.set(dirname)

    def _browse_save_dir(self):
        """Browse for save directory."""
        dirname = filedialog.askdirectory(title="Select save directory")
        if dirname:
            self.save_dir_var.set(dirname)

    def _refresh_devices(self):
        """Refresh available audio devices."""
        try:
            devices = sd.query_devices()
            self.device_list = [
                (i, d['name'])
                for i, d in enumerate(devices)
                if d.get('max_input_channels', 0) > 0
            ]
            self.device_combo['values'] = [name for _, name in self.device_list]
            if self.device_list:
                self.device_combo.current(0)
                self._on_device_selected()
        except Exception as e:
            self._log(f"Error refreshing devices: {e}")

    def _on_device_selected(self):
        """Handle device selection."""
        idx = self.device_combo.current()
        if idx < 0 or idx >= len(self.device_list):
            return
        
        device_idx, _ = self.device_list[idx]
        try:
            info = sd.query_devices(device_idx)
            n_ch = info.get('max_input_channels', 1)
            self.channel_list = [f"Channel {i+1}" for i in range(n_ch)]
            self.channel_combo['values'] = self.channel_list
            if self.channel_list:
                self.channel_combo.current(0)
        except Exception as e:
            self._log(f"Error querying device: {e}")

    def _send_message(self, msg_type: str, data: Any = None):
        """Send message to main thread."""
        self.message_queue.put((msg_type, data))

    def _update_message_display(self):
        """Process messages from background threads."""
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()
                if msg_type == 'log':
                    self._log(data)
                elif msg_type == 'status':
                    self.status_var.set(data)
                elif msg_type == 'light':
                    self.status_light.set_state(data)
                elif msg_type == 'note':
                    self.note_display_var.set(data)
                elif msg_type == 'confidence':
                    self.confidence_var.set(data)
                elif msg_type == 'midi':
                    self.midi_var.set(data)
                elif msg_type == 'reason':
                    self.reason_var.set(data)
                elif msg_type == 'fretboard':
                    # Show single position (note+string mode)
                    self.fretboard.show_position(data[0], data[1])
                elif msg_type == 'fretboard_note':
                    # Show all positions for a MIDI note (note-only mode)
                    self.fretboard.show_note(data)
        except queue.Empty:
            pass
        
        if not self._stopping:
            self.root.after(50, self._update_message_display)

    def _log(self, msg: str):
        """Add message to log."""
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def _update_params(self):
        """Update parameters from GUI."""
        self.params['note_checkpoint'] = self.note_checkpoint_var.get()
        self.params['string_checkpoint'] = self.string_checkpoint_var.get()
        self.params['disambig_dir'] = self.disambig_dir_var.get()
        self.params['save_dir'] = self.save_dir_var.get() if self.save_enabled_var.get() else ""
        
        for key, var in self.param_vars.items():
            self.params[key] = var.get()
        
        current = self.device_combo.current()
        if current >= 0 and current < len(self.device_list):
            self.params['device_idx'], _ = self.device_list[current]
        
        ch_current = self.channel_combo.current()
        if ch_current >= 0:
            self.params['channel_idx'] = ch_current

    def _get_model_device(self):
        """Get the best available compute device."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _create_sequential_note_model(self, num_classes: int, n_mels: int, device):
        """Create a Sequential-style note model matching older checkpoints."""
        import torch.nn as nn
        
        class SequentialNoteCNN(nn.Module):
            def __init__(self, num_classes: int, n_mels: int = 48):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.classifier = nn.Linear(128, num_classes)
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        return SequentialNoteCNN(num_classes, n_mels).to(device)

    def _load_checkpoints(self, note_path: Path, string_path: Optional[Path], device, disambig_dir: Optional[Path] = None):
        """Load model checkpoints."""
        # Load note model
        note_ckpt = torch.load(note_path, map_location=device)
        
        # Get mel params from checkpoint
        mel_params = note_ckpt.get("mel_params", {})
        n_mels = mel_params.get("n_mels", 48)
        hop_length = mel_params.get("hop_length", 256)
        num_frames = mel_params.get("num_frames", 32)
        sample_rate = mel_params.get("sample_rate", 48000)
        attack_samples = mel_params.get("attack_samples", 12000)
        n_fft = mel_params.get("n_fft", 1024)
        f_min = mel_params.get("f_min", 55.0)
        f_max = mel_params.get("f_max", 3000.0)
        
        num_classes = note_ckpt.get("num_classes", 48)
        
        # Support both old (idx_to_midi) and new (idx_to_pos/pos_to_meta) checkpoint formats
        idx_to_midi = note_ckpt.get("idx_to_midi", None)
        if idx_to_midi is not None:
            # Old format: direct idx -> MIDI mapping
            if isinstance(idx_to_midi, dict):
                idx_to_midi = {int(k): int(v) for k, v in idx_to_midi.items()}
        
        idx_to_pos = note_ckpt.get("idx_to_pos", {i: i for i in range(num_classes)})
        pos_to_meta = note_ckpt.get("pos_to_meta", {})
        
        # Detect model architecture from state dict keys
        state_dict = note_ckpt["model_state"]
        if "features.0.weight" in state_dict:
            # Sequential-style architecture - create compatible model
            note_model = self._create_sequential_note_model(num_classes, n_mels, device)
        else:
            # Standard NoteCNN architecture
            note_model = NoteCNN(num_classes=num_classes, n_mels=n_mels).to(device)
        
        note_model.load_state_dict(state_dict)
        note_model.eval()
        
        # Build mel transform using common module
        mel_spec, db_transform = create_mel_transforms(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
        
        state = {
            "note_model": note_model,
            "string_model": None,
            "mel_spec": mel_spec,
            "db_transform": db_transform,
            "device": device,
            "sample_rate": sample_rate,
            "attack_samples": attack_samples,
            "n_mels": n_mels,
            "hop_length": hop_length,
            "num_frames": num_frames,
            "num_classes": num_classes,
            "idx_to_midi": idx_to_midi,  # Old format (may be None)
            "idx_to_pos": idx_to_pos,
            "pos_to_meta": pos_to_meta,
        }
        
        # Load string model if in note+string mode
        if string_path and string_path.exists():
            string_ckpt = torch.load(string_path, map_location=device)
            string_mel_params = string_ckpt.get("mel_params", {})
            
            string_n_mels = string_mel_params.get("n_mels", 64)
            string_hop_length = string_mel_params.get("hop_length", 128)
            string_num_frames = string_mel_params.get("num_frames", 96)
            string_n_fft = string_mel_params.get("n_fft", 1024)
            string_f_min = string_mel_params.get("f_min", 55.0)
            string_f_max = string_mel_params.get("f_max", 3000.0)
            
            num_strings = string_ckpt.get("num_strings", 6)
            
            string_model = StringCNN(num_strings=num_strings, n_mels=string_n_mels).to(device)
            string_model.load_state_dict(string_ckpt["model_state"])
            string_model.eval()
            
            # Build string mel transforms using common module
            string_mel_spec, string_db_transform = create_mel_transforms(
                sample_rate=sample_rate,
                n_fft=string_n_fft,
                hop_length=string_hop_length,
                n_mels=string_n_mels,
                f_min=string_f_min,
                f_max=string_f_max,
            )
            
            state["string_model"] = string_model
            state["string_mel_spec"] = string_mel_spec
            state["string_db_transform"] = string_db_transform
            state["string_n_mels"] = string_n_mels
            state["string_hop_length"] = string_hop_length
            state["string_num_frames"] = string_num_frames
        
        # Load disambiguator models if in note+disambig mode
        if disambig_dir and disambig_dir.exists():
            state["disambig_models"] = {}
            state["disambig_mel_transforms"] = {}
            
            # Map string pair names to directory names
            pair_names = ['E_A', 'A_D', 'D_G', 'G_B', 'B_e']
            
            for pair_name in pair_names:
                # Try both batch directory structure and single model structure
                ckpt_path = disambig_dir / pair_name / "disambig_best.pt"
                if not ckpt_path.exists():
                    ckpt_path = disambig_dir / f"disambig_{pair_name.lower()}_best.pt"
                if not ckpt_path.exists():
                    continue
                
                try:
                    ckpt = torch.load(ckpt_path, map_location='cpu')
                    disambig_mel_params = ckpt.get("mel_params", {})
                    
                    disambig_n_mels = disambig_mel_params.get("n_mels", 64)
                    disambig_hop_length = disambig_mel_params.get("hop_length", 128)
                    disambig_num_frames = disambig_mel_params.get("num_frames", 32)
                    disambig_n_fft = disambig_mel_params.get("n_fft", 1024)
                    disambig_f_min = disambig_mel_params.get("f_min", 55.0)
                    disambig_f_max = disambig_mel_params.get("f_max", 3000.0)
                    
                    model = DisambiguatorCNN(n_mels=disambig_n_mels, num_frets=13)
                    # Initialize with dummy forward pass
                    dummy = torch.zeros(1, 1, disambig_n_mels, disambig_num_frames)
                    dummy_fret = torch.zeros(1, dtype=torch.long)
                    with torch.no_grad():
                        model(dummy, dummy_fret, dummy_fret)
                    model.load_state_dict(ckpt["model_state"])
                    model = model.to(device)
                    model.eval()
                    
                    mel_spec, db_transform = create_mel_transforms(
                        sample_rate=sample_rate,
                        n_fft=disambig_n_fft,
                        hop_length=disambig_hop_length,
                        n_mels=disambig_n_mels,
                        f_min=disambig_f_min,
                        f_max=disambig_f_max,
                    )
                    
                    state["disambig_models"][pair_name] = {
                        "model": model,
                        "mel_spec": mel_spec,
                        "db_transform": db_transform,
                        "n_mels": disambig_n_mels,
                        "hop_length": disambig_hop_length,
                        "num_frames": disambig_num_frames,
                    }
                except Exception as e:
                    print(f"Warning: Failed to load disambiguator {pair_name}: {e}")
        
        return state

    def start_inference(self):
        """Start the inference system."""
        if self.is_running:
            return
        
        self._update_params()
        self._sync_monitor_enabled()
        
        # Validate checkpoints
        note_path = Path(self.params['note_checkpoint'])
        if not note_path.exists():
            messagebox.showerror("Error", f"Note checkpoint not found: {note_path}")
            return
        
        if self.inference_mode == MODE_NOTE_STRING:
            string_path = Path(self.params['string_checkpoint'])
            if not string_path.exists():
                messagebox.showerror("Error", f"String checkpoint not found: {string_path}")
                return
        
        if self.inference_mode == MODE_NOTE_DISAMBIG:
            disambig_dir = Path(self.params['disambig_dir'])
            if not disambig_dir.exists():
                messagebox.showerror("Error", f"Disambiguator directory not found: {disambig_dir}")
                return
        
        if self.params['device_idx'] is None:
            messagebox.showerror("Error", "Please select an audio input device")
            return
        
        self.is_running = True
        self._stopping = False
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("Loading models...")
        self.status_light.set_state("processing")
        
        self.log_text.delete(1.0, tk.END)
        
        # Restart message display loop (may have stopped on previous stop)
        self._update_message_display()
        
        threading.Thread(target=self._run_inference, daemon=True).start()

    def _run_inference(self):
        """Run inference in background thread."""
        try:
            self._send_message('log', "Loading models...")
            model_device = self._get_model_device()
            self._send_message('log', f"Using compute device: {model_device}")
            
            note_path = Path(self.params['note_checkpoint'])
            string_path = None
            disambig_dir = None
            if self.inference_mode == MODE_NOTE_STRING:
                string_path = Path(self.params['string_checkpoint'])
            elif self.inference_mode == MODE_NOTE_DISAMBIG:
                disambig_dir = Path(self.params['disambig_dir'])
            
            self.state = self._load_checkpoints(note_path, string_path, model_device, disambig_dir)
            
            sr = self.state["sample_rate"]
            attack = self.state["attack_samples"]
            
            pre_samples = int(round((self.params['pre_ms'] / 1000.0) * sr))
            post_samples = int(round((self.params['post_ms'] / 1000.0) * sr))
            capture_total = pre_samples + post_samples
            
            sweep_step_samples = max(1, int(round((self.params.get('sweep_step_ms', 5.0) / 1000.0) * sr)))
            min_silence_samples = max(1, int(round((self.params['min_silence_ms'] / 1000.0) * sr)))
            
            self._send_message('log', f"Model sample rate: {sr} Hz")
            self._send_message('log', f"Attack window: {attack} samples ({attack/sr:.3f}s)")
            self._send_message('log', f"Mode: {MODE_LABELS[self.inference_mode]}")
            
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
                if status:
                    print(f"Audio status: {status}")
                
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
                
                src = self.monitor_audio
                if src is not None:
                    out_len = outdata.shape[0]
                    if len(src) >= out_len:
                        outdata[:, 0] = src[:out_len]
                        if outdata.shape[1] > 1:
                            outdata[:, 1] = src[:out_len]
                    else:
                        outdata[:len(src), 0] = src
                        outdata[len(src):, 0] = 0
                        if outdata.shape[1] > 1:
                            outdata[:len(src), 1] = src
                            outdata[len(src):, 1] = 0
            
            self.stream = sd.InputStream(
                device=device_idx,
                samplerate=sr,
                channels=max_channels,
                blocksize=ANALYSIS_HOP,
                callback=audio_callback,
            )
            
            if self.monitor_enabled:
                self.output_stream = sd.OutputStream(
                    samplerate=sr,
                    channels=2,
                    blocksize=ANALYSIS_HOP,
                    callback=output_callback,
                )
            
            self.stream.start()
            if self.output_stream:
                self.output_stream.start()
            
            self._send_message('status', 'Armed')
            self._send_message('light', 'armed')
            self._send_message('log', 'Re-armed')
            
            self._process_audio_queue()
            
        except Exception as e:
            import traceback
            self._send_message('log', f"Error: {e}\n{traceback.format_exc()}")
            self._send_message('status', 'Error')
            self._send_message('light', 'error')
            self.is_running = False

    def _process_audio_queue(self):
        """Process audio from queue."""
        while self.is_running and not self._stopping:
            try:
                block, mono = self.audio_queue.get(timeout=0.1)
                self._process_audio_block(block, mono)
            except queue.Empty:
                continue
            except Exception as e:
                self._send_message('log', f"Processing error: {e}")

    def _process_audio_block(self, indata, mono):
        """Process a block of audio."""
        cs = self.capture_state
        mode = cs['mode']
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        frames = mono.shape[0]
        
        if mode == 'armed':
            if cs['pre_buffer'] is not None:
                self._update_prebuffer(indata)
            
            if peak >= self.params['trigger_threshold']:
                # Get pre-buffer audio
                pre_audio = self._get_prebuffer_channel() if cs['pre_buffer'] is not None else np.array([], dtype=np.float32)
                cs['recording_parts'] = [pre_audio, mono.copy()]
                cs['recorded_post'] = frames
                cs['mode'] = 'capturing'
                
                self._send_message('log', f"Triggered! Peak={peak:.4f}")
                self._send_message('status', 'Capturing...')
                self._send_message('light', 'capturing')
        
        elif mode == 'capturing':
            cs['recording_parts'].append(mono.copy())
            cs['recorded_post'] += frames
            
            # Record for fixed post_samples duration (no early cutoff)
            if cs['recorded_post'] >= cs['post_samples']:
                # Assemble captured audio
                captured = np.concatenate(cs['recording_parts']).astype(np.float32)
                
                # Trim/pad to exact size
                if captured.shape[0] > cs['capture_total']:
                    captured = captured[:cs['capture_total']]
                elif captured.shape[0] < cs['capture_total']:
                    captured = np.pad(captured, (0, cs['capture_total'] - captured.shape[0]), mode='constant')
                
                self._send_message('status', 'Processing...')
                self._send_message('light', 'processing')
                
                self._process_captured_audio(captured)
                
                cs['mode'] = 'waiting_silence'
                cs['silence_ctr'] = 0
        
        elif mode == 'waiting_silence':
            # Wait for silence before re-arming (use peak, not RMS)
            if peak <= self.params['silence_threshold']:
                cs['silence_ctr'] += frames
            else:
                cs['silence_ctr'] = 0
            
            if cs['silence_ctr'] >= cs['min_silence_samples']:
                cs['mode'] = 'armed'
                cs['recording_parts'] = []
                cs['recorded_post'] = 0
                cs['pre_idx'] = 0
                
                self._send_message('status', 'Armed')
                self._send_message('light', 'armed')
                self._send_message('log', 'Re-armed')

    def _apply_fades(self, audio: np.ndarray, sr: int, fade_ms: float = 3.0) -> np.ndarray:
        """Apply preprocessing matching data_collection.py exactly."""
        return preprocess_audio(audio, sr, fade_ms=fade_ms)

    def _update_prebuffer(self, block):
        """Update circular pre-buffer."""
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
        """Get pre-buffer for selected channel."""
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

    def _process_captured_audio(self, captured):
        """Process captured audio and run inference."""
        cs = self.capture_state
        
        if self.inference_mode == MODE_NOTE_ONLY:
            self._process_note_only(captured)
        elif self.inference_mode == MODE_NOTE_DISAMBIG:
            self._process_note_and_disambig(captured)
        else:
            self._process_note_and_string(captured)

    def _process_note_only(self, captured):
        """Process with note model only - shows ALL positions for detected note."""
        cs = self.capture_state
        
        # Sweep for best window
        best_start, best_window, best_topk = self._sweep_best_window_note(
            captured=captured,
            sweep_step_samples=cs['sweep_step_samples'],
            topk=5
        )
        
        if not best_topk:
            self._send_message('log', "No note candidates found")
            return
        
        # Get top prediction
        # First element is MIDI (old format) or pos (new format)
        first_elem, name, prob = best_topk[0]
        
        # Determine MIDI note
        if self.state.get("idx_to_midi") is not None:
            # Old format: first_elem is already MIDI
            midi = first_elem
        else:
            # New format: first_elem is pos, convert to MIDI
            midi = self._pos_to_midi(first_elem)
        
        # Update UI
        self._send_message('note', name)
        self._send_message('confidence', f"Conf: {prob:.3f}")
        self._send_message('midi', f"MIDI: {midi}")
        self._send_message('reason', "")
        
        # Show ALL positions where this note can be played
        self._send_message('fretboard_note', midi)
        self._send_message('light', 'detected')
        
        topk_str = ", ".join([f"{n}({p:.2f})" for _, n, p in best_topk[:5]])
        self._send_message('log', f"Detected: {name} (MIDI {midi}) | TopK: {topk_str}")
        
        if self.params['save_dir']:
            self._save_captured_audio(captured, best_window, f"{name}", prob)

    def _process_note_and_string(self, captured):
        """Process with both note and string models."""
        cs = self.capture_state
        
        # Sweep for best window using note model
        best_start, best_window, best_note_topk = self._sweep_best_window_note(
            captured=captured,
            sweep_step_samples=cs['sweep_step_samples'],
            topk=5
        )
        
        if not best_note_topk:
            self._send_message('log', "No note candidates found")
            return
        
        # Run string model on same window
        best_string_topk = self._predict_topk_string(best_window, topk=5)
        
        # Get note prediction
        midi, name, note_prob = self._decide_note(best_note_topk, self.params.get('margin', 0.1))
        
        # Get string prediction
        string_idx, string_prob = self._decide_string(best_string_topk, self.params.get('margin', 0.1))
        
        # Resolve position
        final_string_idx, final_fret = self._resolve_position(midi, string_idx, best_string_topk)
        
        # Update UI
        self._send_message('note', name)
        self._send_message('confidence', f"Note: {note_prob:.3f} | String: {string_prob:.3f}")
        
        disp_s = self._to_display_string_num(final_string_idx)
        open_name = self._string_open_name(final_string_idx)
        self._send_message('midi', f"MIDI: {midi} | String: {disp_s} ({open_name}) | Fret: {final_fret}")
        self._send_message('reason', "")
        
        self._send_message('fretboard', (final_string_idx, final_fret))
        self._send_message('light', 'detected')
        
        topk_note_str = ", ".join([f"{n}({p:.2f})" for _, n, p in best_note_topk[:5]])
        topk_str_str = ", ".join([f"S{self._to_display_string_num(si)}({p:.2f})" for si, _, p in best_string_topk[:5]])
        
        self._send_message('log', f"Detected: {name} | Pos: S{disp_s} F{final_fret} | NoteTopK: {topk_note_str} | StringTopK: {topk_str_str}")
        
        if self.params['save_dir']:
            self._save_captured_audio(captured, best_window, f"{name}_S{final_string_idx+1}_F{final_fret}", note_prob)

    def _process_note_and_disambig(self, captured):
        """Process with note model + disambiguator models."""
        cs = self.capture_state
        
        # Sweep for best window using note model
        best_start, best_window, best_note_topk = self._sweep_best_window_note(
            captured=captured,
            sweep_step_samples=cs['sweep_step_samples'],
            topk=5
        )
        
        if not best_note_topk:
            self._send_message('log', "No note candidates found")
            return
        
        # Get note prediction
        midi, name, note_prob = self._decide_note(best_note_topk, self.params.get('margin', 0.1))
        
        # Find which string pairs this note is ambiguous for
        final_string_idx, final_fret, disambig_info = self._disambiguate_note(midi, best_window)
        
        # Update UI
        self._send_message('note', name)
        self._send_message('confidence', f"Note: {note_prob:.3f}{disambig_info}")
        
        disp_s = self._to_display_string_num(final_string_idx)
        open_name = self._string_open_name(final_string_idx)
        self._send_message('midi', f"MIDI: {midi} | String: {disp_s} ({open_name}) | Fret: {final_fret}")
        self._send_message('reason', "")
        
        self._send_message('fretboard', (final_string_idx, final_fret))
        self._send_message('light', 'detected')
        
        topk_note_str = ", ".join([f"{n}({p:.2f})" for _, n, p in best_note_topk[:5]])
        self._send_message('log', f"Detected: {name} | Pos: S{disp_s} F{final_fret} | TopK: {topk_note_str}")
        
        if self.params['save_dir']:
            self._save_captured_audio(captured, best_window, f"{name}_S{final_string_idx+1}_F{final_fret}", note_prob)

    def _disambiguate_note(self, midi, window):
        """Use disambiguator models to determine string/fret position.
        
        Returns: (string_idx, fret, info_string)
        """
        # Get all possible positions for this MIDI note
        positions = []
        for string_num, open_midi in STRING_OPEN_MIDI.items():
            fret = midi - open_midi
            if 0 <= fret <= 12:
                positions.append((string_num, fret))
        
        if not positions:
            # Should not happen, but fallback
            return 0, 0, " | No positions"
        
        if len(positions) == 1:
            # Unambiguous - only one position
            string_num, fret = positions[0]
            return string_num - 1, fret, " | Unambiguous"  # Convert to 0-indexed
        
        # Check which disambiguators we have and which are relevant
        disambig_models = self.state.get("disambig_models", {})
        
        # Find which string pair(s) this note belongs to
        # Format: pair_name -> (string_a, string_b) where both are 1-indexed
        relevant_pairs = []
        for pair_name, pair_info in ALL_PAIRS.items():
            if midi in pair_info:
                relevant_pairs.append(pair_name)
        
        if not relevant_pairs:
            # No relevant disambiguator - pick lowest fret position
            positions.sort(key=lambda x: x[1])
            string_num, fret = positions[0]
            return string_num - 1, fret, " | No disambig"
        
        # Run relevant disambiguators
        results = {}
        for pair_name in relevant_pairs:
            if pair_name not in disambig_models:
                continue
            
            dm = disambig_models[pair_name]
            pair_info = ALL_PAIRS[pair_name]
            
            if midi not in pair_info:
                continue
            
            midi_info = pair_info[midi]
            string_a, fret_a = midi_info[0]
            string_b, fret_b = midi_info[1]
            
            # Run inference
            prob_a, prob_b = self._run_disambiguator(window, dm, fret_a, fret_b)
            
            if prob_a > prob_b:
                results[pair_name] = (string_a, fret_a, prob_a)
            else:
                results[pair_name] = (string_b, fret_b, prob_b)
        
        if not results:
            # No disambiguator succeeded - pick lowest fret
            positions.sort(key=lambda x: x[1])
            string_num, fret = positions[0]
            return string_num - 1, fret, " | Disambig N/A"
        
        # Use first result (should only be one pair per note typically)
        pair_name = list(results.keys())[0]
        string_num, fret, prob = results[pair_name]
        
        return string_num - 1, fret, f" | {pair_name}: {prob:.2f}"

    def _run_disambiguator(self, window, dm, fret_a, fret_b):
        """Run a single disambiguator model."""
        import torch.nn.functional as F
        
        device = self.state["device"]
        model = dm["model"]
        mel_spec = dm["mel_spec"]
        db_transform = dm["db_transform"]
        num_frames = dm["num_frames"]
        
        # Preprocess using common module
        spec = waveform_to_logmel_torch(
            window, mel_spec, db_transform, device, num_frames
        )
        
        fret_a_t = torch.tensor([fret_a], dtype=torch.long).to(device)
        fret_b_t = torch.tensor([fret_b], dtype=torch.long).to(device)
        
        with torch.no_grad():
            logits = model(spec, fret_a_t, fret_b_t)
            probs = F.softmax(logits, dim=1)
        
        return probs[0, 0].item(), probs[0, 1].item()

    def _sweep_best_window_note(self, captured, sweep_step_samples, topk=5):
        """Find the best window for note classification."""
        attack = self.state["attack_samples"]
        
        # Pad if too short
        if len(captured) < attack:
            padded = np.zeros(attack, dtype=np.float32)
            padded[:len(captured)] = captured.astype(np.float32)
            captured = padded
        
        best_start = 0
        best_window = captured[:attack].copy()
        best_prob = -1.0
        best_topk = []
        
        last_start = max(0, len(captured) - attack)
        step = max(1, int(sweep_step_samples))
        
        for start in range(0, last_start + 1, step):
            window = captured[start:start + attack]
            topk_results = self._predict_topk_note(window, topk)
            
            if topk_results:
                _, _, prob = topk_results[0]
                if prob > best_prob:
                    best_prob = prob
                    best_start = start
                    best_window = window.copy()
                    best_topk = topk_results
        
        return best_start, best_window, best_topk

    def _predict_topk_note(self, window, topk=5):
        """Run note model inference."""
        # Peak normalize before inference (matches training)
        w = peak_normalize(window)
        
        spec = waveform_to_logmel_torch(
            w, 
            self.state["mel_spec"],
            self.state["db_transform"],
            self.state["num_frames"],
            self.state["device"]
        )
        
        with torch.no_grad():
            logits = self.state["note_model"](spec)
            probs = torch.softmax(logits, dim=1)
            top_probs, top_idxs = torch.topk(probs, topk, dim=1)
        
        results = []
        idx_to_midi = self.state.get("idx_to_midi")
        
        for i in range(topk):
            idx = int(top_idxs[0, i].item())
            prob = float(top_probs[0, i].item())
            
            # Use idx_to_midi if available (old checkpoint format)
            if idx_to_midi is not None:
                midi = int(idx_to_midi.get(idx, idx))
                name = midi_to_note_name(midi)
                results.append((midi, name, prob))  # Return MIDI as first element for old format
            else:
                # New format: idx -> pos -> midi
                pos = self.state["idx_to_pos"].get(idx, idx)
                midi = self._pos_to_midi(pos)
                name = midi_to_note_name(midi)
                results.append((pos, name, prob))
        
        return results

    def _predict_topk_string(self, window, topk=5):
        """Run string model inference."""
        if self.state.get("string_model") is None:
            return []
        
        # Peak normalize before inference (matches training)
        w = peak_normalize(window)
        
        spec = waveform_to_logmel_torch(
            w,
            self.state["string_mel_spec"],
            self.state["string_db_transform"],
            self.state["string_num_frames"],
            self.state["device"]
        )
        
        with torch.no_grad():
            logits = self.state["string_model"](spec)
            probs = torch.softmax(logits, dim=1)
            top_probs, top_idxs = torch.topk(probs, topk, dim=1)
        
        results = []
        for i in range(topk):
            idx = int(top_idxs[0, i].item())
            prob = float(top_probs[0, i].item())
            name = STRING_NAMES_BY_INDEX[idx] if 0 <= idx < 6 else f"S{idx}"
            results.append((idx, name, prob))
        
        return results

    def _pos_to_midi(self, pos: int) -> int:
        """Convert position to MIDI note."""
        meta = self.state.get("pos_to_meta", {}).get(pos, {})
        if "midi" in meta:
            return meta["midi"]
        
        # Fallback: assume 8 frets per string
        frets_per_string = 8
        string_idx = pos // frets_per_string
        fret = pos % frets_per_string
        return STANDARD_TUNING_LIST[string_idx] + fret

    def _decide_note(self, topk, margin=0.1):
        """Decide final note from top-k."""
        if not topk:
            return 0, "--", 0.0
        
        pos, name, prob = topk[0]
        midi = self._pos_to_midi(pos)
        return midi, name, prob

    def _decide_string(self, topk, margin=0.1):
        """Decide final string from top-k."""
        if not topk:
            return 0, 0.0
        
        string_idx, _, prob = topk[0]
        return string_idx, prob

    def _resolve_position(self, midi: int, string_idx_pred: int, string_topk):
        """Resolve final (string, fret) from predictions."""
        tuning = STANDARD_TUNING_LIST
        max_fret = self.fretboard.frets
        
        # Try predicted string first
        fret = midi - tuning[string_idx_pred]
        if 0 <= fret <= max_fret:
            return string_idx_pred, fret
        
        # Try other strings by probability
        for string_idx, _, _ in string_topk:
            fret = midi - tuning[string_idx]
            if 0 <= fret <= max_fret:
                return string_idx, fret
        
        # Fallback: find any valid position
        for string_idx in range(6):
            fret = midi - tuning[string_idx]
            if 0 <= fret <= max_fret:
                return string_idx, fret
        
        # Last resort
        return string_idx_pred, max(0, min(midi - tuning[string_idx_pred], max_fret))

    def _save_captured_audio(self, captured, best_window, name, prob):
        """Save captured audio files."""
        save_dir = Path(self.params['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        sr = self.capture_state['sr']
        
        raw_path = save_dir / f"cap_{timestamp}_raw.wav"
        save_wav_int16(raw_path, sr, captured)
        self._send_message('log', f"Saved: {raw_path.name}")

    def stop_inference(self):
        """Stop the inference system."""
        if not self.is_running:
            return
        
        self._stopping = True
        self.is_running = False
        
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
            self.stream = None
        
        if self.output_stream:
            try:
                self.output_stream.stop()
                self.output_stream.close()
            except:
                pass
            self.output_stream = None
        
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Stopped")
        self.status_light.set_state("stopped")
        self._log("Inference stopped")

    def _on_close(self):
        """Handle window close."""
        self.stop_inference()
        self._stopping = True
        self.root.destroy()


# ============================
# MAIN
# ============================

def main():
    root = tk.Tk()
    root.minsize(1000, 700)
    app = FretListenerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()