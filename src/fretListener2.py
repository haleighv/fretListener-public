#!/usr/bin/env python3
"""
fretListener2.py - Real-time Guitar Position Inference (Note + String)

A GUI application for real-time guitar note + string inference using two neural networks.
Features:
- Queue-based audio capture for thread-safe processing
- Audio monitoring (hear what you're playing)
- Modern dark-mode UI
- Visual fretboard display (single inferred position dot)
- Configurable trigger and inference parameters

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
from typing import Dict, Any, Tuple, Optional, List, Set
import json

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
from string_model import StringCNN


# ============================
# VERSION
# ============================

VERSION = "2.1"


# ============================
# AUDIO CONFIG
# ============================

DEFAULT_SAMPLE_RATE = 48000
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
            # Dim the color slightly
            color = self._dim_color(color, 0.6)
        
        # Draw outer ring
        self.create_oval(padding, padding,
                        self.size + padding, self.size + padding,
                        fill=color, outline="#1e1e1e", width=2)
    
    def _dim_color(self, hex_color: str, factor: float) -> str:
        """Dim a hex color by a factor."""
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def set_state(self, state: str):
        """Set the status light state."""
        if self.pulse_job:
            self.after_cancel(self.pulse_job)
            self.pulse_job = None
        
        self.state = state
        self.pulse_on = True
        
        if state in ("armed",):
            self._start_pulse()
        else:
            self._draw()
    
    def _start_pulse(self):
        """Start a subtle pulse animation."""
        self.pulse_on = not self.pulse_on
        self._draw()
        self.pulse_job = self.after(800, self._start_pulse)


# ============================
# GUITAR FRETBOARD WIDGET
# ============================

class GuitarFretboard(tk.Canvas):
    """Visual guitar fretboard widget showing a single inferred position."""
    
    # Display order: index 0 = top of screen (high E), index 5 = bottom (low E)
    STRING_NAMES_TOP_TO_BOTTOM = ["E4", "B3", "G3", "D3", "A2", "E2"]
    
    # Standard tuning MIDI notes (string_idx 0 = low E, 5 = high E)
    STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4
    
    def __init__(self, parent, frets: int = 12, **kwargs):
        super().__init__(parent, **kwargs)
        self.frets = int(frets)
        self.active_pos: Optional[Tuple[int, int]] = None  # (string_idx, fret)

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

            thickness = 1 + s * 0.4
            self.create_line(x_start, y, x_end, y, width=thickness, fill="#c0c0c0")

            self.create_text(18, y, text=self.STRING_NAMES_TOP_TO_BOTTOM[s],
                           font=("Arial", 9), fill="#888")

        # Fret numbers
        for fret in range(self.frets + 1):
            if fret > 0:
                x = 45 + fret * self.fret_spacing - self.fret_spacing // 2
                self.create_text(x, 25 + self.string_spacing * 5 + 15,
                               text=str(fret), font=("Arial", 9), fill="#666")

    def clear_notes(self):
        """Clear active position."""
        self.delete("note")
        self.active_pos = None

    def show_position(self, string_idx: int, fret: int):
        """Show a single (string_idx, fret) position."""
        self.clear_notes()
        self.active_pos = (int(string_idx), int(fret))
        self._draw_note(int(string_idx), int(fret))

    def _draw_note(self, string_idx: int, fret: int):
        """Draw a note marker at position.

        string_idx: 0=low E (display at bottom), 5=high E (display at top)
        """
        if fret < 0 or fret > self.frets:
            return
        if string_idx < 0 or string_idx >= 6:
            return

        display_idx = 5 - string_idx
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
# MAIN APPLICATION
# ============================

class NoteInferenceGUI:
    """Main GUI application for real-time position inference (note + string)."""
    
    def __init__(self, root):
        self.root = root
        self.root.title(f"FretListener v{VERSION} - Real-time Position Inference")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.configure(bg="#1e1e1e")
        
        # Parameter descriptions for tooltips
        self.param_descriptions = {
            'trigger_threshold': 'Peak amplitude level that triggers audio capture (0.001-0.1)\nLower = more sensitive',
            'pre_ms': 'Milliseconds of audio to capture BEFORE the trigger point\nCaptures the attack transient',
            'post_ms': 'Milliseconds of audio to capture AFTER the trigger point\nEnsures full note is captured',
            'silence_threshold': 'Peak level below which audio is considered silence\nUsed to detect when to re-arm',
            'min_silence_ms': 'How long the signal must stay silent before re-arming\nPrevents retriggering on note decay',
            'sweep_step_ms': 'Step size when searching for best inference window\nSmaller = more accurate but slower',
            'margin': 'Required confidence gap between top 2 predictions\nHigher = more certain, but may miss ambiguous predictions',
            'topk': 'Number of top candidate notes to display in debug output',
            'save_dir': 'Directory to save captured WAV files for analysis',
            'filter': 'Analyzes audio features to reject non-picked sounds:\n- Finger slides (poor harmonics)\n- Muted notes (weak attack)\n- Silence or noise'
        }
        
        # State
        self.is_running = False
        self.stream: Optional[sd.InputStream] = None
        self.output_stream: Optional[sd.OutputStream] = None
        self.state = None
        
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
            'mode': 'stopped',  # stopped, armed, capturing, waiting_silence
            'pre_buffer': None,
            'pre_idx': 0,
            'recording_parts': [],
            'recorded_post': 0,
            'silence_ctr': 0,
        }
        
        # Default parameters
        self.params = {
            'note_checkpoint': 'artifacts/models/note_20251211_164726/note_cnn_best.pt',
            'string_checkpoint': 'artifacts/models/string_20251217_135254/string_cnn_best.pt',
            'trigger_threshold': 0.01,
            'pre_ms': 50.0,
            'post_ms': 500.0,
            'silence_threshold': 0.006,
            'min_silence_ms': 80.0,
            'sweep_step_ms': 5.0,
            'margin': 0.10,
            'topk': 5,
            'save_dir': 'artifacts/captures',
            'device_idx': None,
            'channel_idx': 0
        }
        
        self._setup_ui()
        self._refresh_devices()
        self._update_message_display()
    
    def _to_display_string_num(self, internal_string_idx: int) -> int:
        # internal: 0=low E ... 5=high E
        # display:  6=low E ... 1=high E
        return 6 - int(internal_string_idx)

    def _string_open_name(self, internal_string_idx: int) -> str:
        names = ["E2", "A2", "D3", "G3", "B3", "E4"]  # internal: 0=low E ... 5=high E
        si = int(internal_string_idx)
        return names[si] if 0 <= si < 6 else f"S{si}"

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
        
        # Combobox colors
        style.map('TCombobox',
                  fieldbackground=[('readonly', bg_mid)],
                  foreground=[('readonly', fg)],
                  selectbackground=[('readonly', bg_mid)],
                  selectforeground=[('readonly', fg)])
        style.configure("TCombobox", fieldbackground=bg_mid, foreground=fg, background=bg_mid)
        
        style.configure("TEntry", fieldbackground=bg_mid, foreground=fg)
        style.configure("TSpinbox", fieldbackground=bg_mid, foreground=fg)
        
        # Progress bar / Scale
        style.configure("Accent.Horizontal.TScale",
                       troughcolor=bg_mid,
                       background=accent)
        
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

        # Model section
        model_frame = ttk.LabelFrame(left_col, text="Models", padding="10")
        model_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(model_frame, text="Note checkpoint:").pack(anchor="w")
        self.note_checkpoint_var = tk.StringVar(value=self.params['note_checkpoint'])
        note_ckpt_frame = ttk.Frame(model_frame)
        note_ckpt_frame.pack(fill="x", pady=(2, 8))
        note_ckpt_entry = ttk.Entry(note_ckpt_frame, textvariable=self.note_checkpoint_var, width=30)
        note_ckpt_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(note_ckpt_frame, text="Browse",
                  command=self._browse_note_checkpoint,
                  style="Action.TButton").pack(side="left", padx=(5, 0))

        ttk.Label(model_frame, text="String checkpoint:").pack(anchor="w")
        self.string_checkpoint_var = tk.StringVar(value=self.params['string_checkpoint'])
        string_ckpt_frame = ttk.Frame(model_frame)
        string_ckpt_frame.pack(fill="x", pady=(2, 0))
        string_ckpt_entry = ttk.Entry(string_ckpt_frame, textvariable=self.string_checkpoint_var, width=30)
        string_ckpt_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(string_ckpt_frame, text="Browse",
                  command=self._browse_string_checkpoint,
                  style="Action.TButton").pack(side="left", padx=(5, 0))

        # Audio Device section
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
            ('sweep_step_ms', 'Sweep (ms):', 1, 50, 1),
            ('margin', 'Margin:', 0.0, 0.5, 0.01),
        ]
        
        for key, label, min_val, max_val, step in param_config:
            row = ttk.Frame(trigger_frame)
            row.pack(fill="x", pady=2)
            
            lbl = ttk.Label(row, text=label, width=10)
            lbl.pack(side="left")
            if key in self.param_descriptions:
                self._create_tooltip(lbl, self.param_descriptions[key])
            
            var = tk.DoubleVar(value=self.params[key])
            self.param_vars[key] = var
            
            slider = ttk.Scale(row, from_=min_val, to=max_val,
                              variable=var, orient=tk.HORIZONTAL, length=150)
            slider.pack(side="left", padx=5)
            
            value_label = ttk.Label(row, text=f"{var.get():.3f}", width=7)
            value_label.pack(side="left")
            
            def make_update_func(v, l):
                return lambda *args: l.config(text=f"{v.get():.3f}")
            var.trace_add('write', make_update_func(var, value_label))

        # Other Settings
        other_frame = ttk.LabelFrame(left_col, text="Other Settings", padding="10")
        other_frame.pack(fill="x", pady=(0, 10))
        
        # Top-K
        topk_row = ttk.Frame(other_frame)
        topk_row.pack(fill="x", pady=2)
        ttk.Label(topk_row, text="Top-K:").pack(side="left")
        self.topk_var = tk.IntVar(value=self.params['topk'])
        ttk.Spinbox(topk_row, from_=1, to=10, textvariable=self.topk_var, width=8).pack(side="left", padx=5)
        
        # Save directory
        save_row = ttk.Frame(other_frame)
        save_row.pack(fill="x", pady=2)
        ttk.Label(save_row, text="Save Dir:").pack(side="left")
        self.save_dir_var = tk.StringVar(value=self.params['save_dir'])
        ttk.Entry(save_row, textvariable=self.save_dir_var, width=20).pack(side="left", padx=5)
        
        self.save_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(save_row, text="Save", variable=self.save_enabled_var).pack(side="left")

        # Audio Monitor
        monitor_frame = ttk.LabelFrame(left_col, text="Audio Monitor", padding="10")
        monitor_frame.pack(fill="x", pady=(0, 10))
        
        self.monitor_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(monitor_frame, text="Enable monitoring (hear input)",
                       variable=self.monitor_var,
                       command=self._sync_monitor_enabled).pack(anchor="w")

        # Filter settings
        filter_frame = ttk.LabelFrame(left_col, text="Note Filtering", padding="10")
        filter_frame.pack(fill="x", pady=(0, 10))
        
        self.filter_enabled_var = tk.BooleanVar(value=False)
        filter_cb = ttk.Checkbutton(filter_frame, text="Enable filter (experimental)",
                                   variable=self.filter_enabled_var,
                                   command=self._on_filter_toggle)
        filter_cb.pack(anchor="w")
        self._create_tooltip(filter_cb, self.param_descriptions['filter'])
        
        ttk.Label(filter_frame, text="(Rejects slides and muted notes)",
                 font=("Arial", 9)).pack(anchor="w", padx=20)
        
        # Advanced filter settings (hidden by default)
        self.filter_advanced_frame = ttk.Frame(filter_frame)
        
        self.filter_min_rms = tk.DoubleVar(value=0.0005)
        self.filter_min_peak_ratio = tk.DoubleVar(value=2.0)
        self.filter_min_harmonics = tk.DoubleVar(value=0.1)
        
        ttk.Label(self.filter_advanced_frame, text="Min RMS:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(self.filter_advanced_frame, from_=0.0001, to=0.01, increment=0.0001,
                   textvariable=self.filter_min_rms, width=8).grid(row=0, column=1, padx=5)
        
        ttk.Label(self.filter_advanced_frame, text="Min Peak/RMS:").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(self.filter_advanced_frame, from_=1.5, to=5.0, increment=0.1,
                   textvariable=self.filter_min_peak_ratio, width=8).grid(row=1, column=1, padx=5)
        
        ttk.Label(self.filter_advanced_frame, text="Min Harmonics:").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(self.filter_advanced_frame, from_=0.0, to=0.5, increment=0.05,
                   textvariable=self.filter_min_harmonics, width=8).grid(row=2, column=1, padx=5)

        # Control buttons
        control_frame = ttk.LabelFrame(left_col, text="Control", padding="10")
        control_frame.pack(fill="x", pady=(0, 10))
        
        self.start_btn = ttk.Button(control_frame, text="START",
                                   command=self.start_inference,
                                   style="Start.TButton")
        self.start_btn.pack(fill="x", pady=(0, 8))
        
        self.stop_btn = ttk.Button(control_frame, text="STOP",
                                  command=self.stop_inference,
                                  state="disabled",
                                  style="Stop.TButton")
        self.stop_btn.pack(fill="x")

        # === RIGHT COLUMN: Display ===
        right_col = ttk.Frame(main)
        right_col.pack(side="left", fill="both", expand=True)

        # Status Box
        status_box = ttk.LabelFrame(right_col, text="Status", padding="10")
        status_box.pack(fill="x", pady=(0, 10))
        
        status_content = ttk.Frame(status_box)
        status_content.pack(fill="x")
        
        self.status_light = StatusLight(status_content, size=40, bg=bg_dark)
        self.status_light.pack(side="left", padx=(0, 15))
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(status_content, textvariable=self.status_var,
                                    font=("Arial", 20, "bold"),
                                    fg=fg, bg=bg_dark)
        self.status_label.pack(side="left", fill="x", expand=True, anchor="w")

        # Detected Position Display
        note_frame = ttk.LabelFrame(right_col, text="Detected Position", padding="10")
        note_frame.pack(fill="x", pady=(0, 10))
        
        note_content = ttk.Frame(note_frame)
        note_content.pack(fill="x")
        
        # Big note display
        self.note_display_var = tk.StringVar(value="--")
        note_label = tk.Label(note_content, textvariable=self.note_display_var,
                             font=("Arial", 56, "bold"),
                             fg="#ff6b6b", bg=bg_dark)
        note_label.pack(side="left", padx=(0, 20))
        
        # Info frame
        info_frame = ttk.Frame(note_content)
        info_frame.pack(side="left", fill="y")
        
        self.confidence_var = tk.StringVar(value="Note conf: -- | String conf: --")
        ttk.Label(info_frame, textvariable=self.confidence_var,
                 font=("Arial", 14)).pack(anchor="w")
        
        self.midi_var = tk.StringVar(value="MIDI: -- | String: -- | Fret: --")
        ttk.Label(info_frame, textvariable=self.midi_var,
                 font=("Arial", 12), foreground=fg_dim).pack(anchor="w")
        
        self.reason_var = tk.StringVar(value="")
        ttk.Label(info_frame, textvariable=self.reason_var,
                 font=("Arial", 10), foreground=fg_dim).pack(anchor="w")

        # Fretboard
        fretboard_frame = ttk.LabelFrame(right_col, text="Fretboard", padding="5")
        fretboard_frame.pack(fill="x", pady=(0, 10))
        
        self.fretboard = GuitarFretboard(fretboard_frame, frets=12)
        self.fretboard.pack()

        # Debug Log
        log_frame = ttk.LabelFrame(right_col, text="Log", padding="5")
        log_frame.pack(fill="both", expand=True)
        
        self.log_text = tk.Text(log_frame, height=10, wrap="word",
                               bg=bg_mid, fg=fg,
                               insertbackground=fg,
                               selectbackground=accent,
                               font=("Consolas", 10))
        self.log_text.pack(fill="both", expand=True, side="left")
        
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(fill="y", side="right")
        self.log_text.config(yscrollcommand=scrollbar.set)

    def _create_tooltip(self, widget, text):
        """Create a tooltip for a widget."""
        def show_tooltip(event):
            tip = tk.Toplevel(self.root)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
            
            label = tk.Label(tip, text=text, font=("Arial", 10),
                           background="lightyellow", foreground="black",
                           relief="solid", borderwidth=1, padx=8, pady=6,
                           justify=tk.LEFT)
            label.pack()
            
            widget._tooltip = tip
        
        def hide_tooltip(event):
            if hasattr(widget, '_tooltip') and widget._tooltip:
                widget._tooltip.destroy()
                widget._tooltip = None
        
        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)

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

    def _browse_note_checkpoint(self):
        """Browse for note checkpoint file."""
        filename = filedialog.askopenfilename(
            title="Select NOTE checkpoint file",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if filename:
            self.note_checkpoint_var.set(filename)

    def _browse_string_checkpoint(self):
        """Browse for string checkpoint file."""
        filename = filedialog.askopenfilename(
            title="Select STRING checkpoint file",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if filename:
            self.string_checkpoint_var.set(filename)

    def _sync_monitor_enabled(self):
        """Sync monitor enabled state."""
        self.monitor_enabled = bool(self.monitor_var.get())

    def _on_filter_toggle(self):
        """Show/hide advanced filter settings."""
        if self.filter_enabled_var.get():
            self.filter_advanced_frame.pack(fill="x", pady=(5, 0))
        else:
            self.filter_advanced_frame.pack_forget()

    def _log(self, msg: str):
        """Log a message to the debug area."""
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")

    def _update_params(self):
        """Update parameters from GUI."""
        self.params['note_checkpoint'] = self.note_checkpoint_var.get()
        self.params['string_checkpoint'] = self.string_checkpoint_var.get()
        self.params['save_dir'] = self.save_dir_var.get() if self.save_enabled_var.get() else ""
        self.params['topk'] = self.topk_var.get()
        
        for key, var in self.param_vars.items():
            self.params[key] = var.get()
        
        # Get device and channel
        current = self.device_combo.current()
        if current >= 0 and current < len(self.device_list):
            self.params['device_idx'], _ = self.device_list[current]
        
        ch_current = self.channel_combo.current()
        if ch_current >= 0:
            self.params['channel_idx'] = ch_current

    def start_inference(self):
        """Start the inference system."""
        if self.is_running:
            return
        
        self._update_params()
        self._sync_monitor_enabled()
        
        # Validate checkpoints
        note_path = Path(self.params['note_checkpoint'])
        string_path = Path(self.params['string_checkpoint'])
        if not note_path.exists():
            messagebox.showerror("Error", f"Note checkpoint not found: {note_path}")
            return
        if not string_path.exists():
            messagebox.showerror("Error", f"String checkpoint not found: {string_path}")
            return
        
        # Validate device
        if self.params['device_idx'] is None:
            messagebox.showerror("Error", "Please select an audio input device")
            return
        
        self.is_running = True
        self._stopping = False
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("Loading models...")
        self.status_light.set_state("processing")
        
        # Clear log
        self.log_text.delete(1.0, tk.END)
        
        # Start in background thread
        threading.Thread(target=self._run_inference, daemon=True).start()

    def _run_inference(self):
        """Run inference (in background thread)."""
        try:
            self._send_message('log', "Loading models...")
            model_device = self._get_model_device()
            self._send_message('log', f"Using compute device: {model_device}")
            
            note_path = Path(self.params['note_checkpoint'])
            string_path = Path(self.params['string_checkpoint'])
            self.state = self._load_checkpoints(note_path, string_path, model_device)
            
            sr = self.state["sample_rate"]
            attack = self.state["attack_samples"]
            
            # Calculate sample counts
            pre_samples = int(round((self.params['pre_ms'] / 1000.0) * sr))
            post_samples = int(round((self.params['post_ms'] / 1000.0) * sr))
            capture_total = pre_samples + post_samples
            
            sweep_step_samples = max(1, int(round((self.params['sweep_step_ms'] / 1000.0) * sr)))
            min_silence_samples = max(1, int(round((self.params['min_silence_ms'] / 1000.0) * sr)))
            
            self._send_message('log', f"Model sample rate: {sr} Hz")
            self._send_message('log', f"Attack window: {attack} samples ({attack/sr:.3f}s)")
            self._send_message('log', f"Capture: pre={pre_samples} + post={post_samples}")
            
            # Get device info
            device_idx = self.params['device_idx']
            device_info = sd.query_devices(device_idx)
            max_channels = device_info['max_input_channels']
            ch = self.params['channel_idx']
            
            # Initialize capture state
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
                
                # Extract mono channel
                if indata.ndim == 2:
                    mono = indata[:, ch].astype(np.float32, copy=True)
                else:
                    mono = indata.astype(np.float32, copy=True)
                
                # Update monitor buffer
                if self.monitor_enabled:
                    self.monitor_audio = mono
                
                # Queue for processing
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
                except Exception as e:
                    self._send_message('log', f"Monitor unavailable: {e}")
                    self.output_stream = None
            
            self.stream.start()
            
            self._send_message('status', "Armed - Play notes!")
            self._send_message('state', 'armed')
            
            self.root.after(10, self._process_audio_queue)
            
        except Exception as e:
            import traceback
            self._send_message('error', f"Failed to start: {e}\n{traceback.format_exc()}")
            self.is_running = False

    def _process_audio_queue(self):
        """Process audio data from the queue (main thread)."""
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
        
        if cs['mode'] == 'armed':
            if cs['pre_buffer'] is not None:
                self._update_prebuffer(indata)
            
            if peak >= self.params['trigger_threshold']:
                pre_audio = self._get_prebuffer_channel()
                cs['recording_parts'] = [pre_audio, mono.copy()]
                cs['recorded_post'] = frames
                cs['mode'] = 'capturing'
                
                self._log(f"Triggered! Peak={peak:.4f}")
                self.status_var.set("Capturing...")
                self.status_light.set_state("capturing")
                
        elif cs['mode'] == 'capturing':
            cs['recording_parts'].append(mono.copy())
            cs['recorded_post'] += frames
            
            if cs['recorded_post'] >= cs['post_samples']:
                cap = np.concatenate(cs['recording_parts']).astype(np.float32)
                
                if cap.shape[0] > cs['capture_total']:
                    cap = cap[:cs['capture_total']]
                elif cap.shape[0] < cs['capture_total']:
                    cap = np.pad(cap, (0, cs['capture_total'] - cap.shape[0]), mode='constant')
                
                # Apply fade-in/fade-out to eliminate edge pops
                cap = self._apply_fades(cap, cs['sr'])
                
                self.status_var.set("Processing...")
                self.status_light.set_state("processing")
                self._process_captured_audio(cap)
                
                cs['mode'] = 'waiting_silence'
                cs['silence_ctr'] = 0
                
        elif cs['mode'] == 'waiting_silence':
            if peak <= self.params['silence_threshold']:
                cs['silence_ctr'] += frames
            else:
                cs['silence_ctr'] = 0
            
            if cs['silence_ctr'] >= cs['min_silence_samples']:
                self._log("Re-armed")
                self.status_var.set("Armed")
                self.status_light.set_state("armed")
                cs['mode'] = 'armed'
                cs['recording_parts'] = []
                cs['recorded_post'] = 0
                cs['pre_idx'] = 0

    def _apply_fades(self, audio: np.ndarray, sr: int, fade_ms: float = 5.0) -> np.ndarray:
        """Apply short fade-in and fade-out to eliminate edge discontinuities."""
        fade_samples = int(sr * fade_ms / 1000.0)
        fade_samples = min(fade_samples, len(audio) // 4)  # Don't fade more than 25% of audio
        
        if fade_samples < 2:
            return audio
        
        audio = audio.copy()
        
        # Fade-in (first few ms)
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        audio[:fade_samples] *= fade_in
        
        # Fade-out (last few ms)  
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        audio[-fade_samples:] *= fade_out
        
        return audio

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
        """Process captured audio and run inference (note + string)."""
        cs = self.capture_state
        
        if self.filter_enabled_var.get():
            is_picked, features = self._analyze_audio_features(captured, cs['sr'])
            
            feature_msg = f"RMS={features['rms']:.4f}, Peak/RMS={features['peak_to_rms']:.1f}, "
            feature_msg += f"Harmonics={features['harmonic_strength']:.2f}"
            self._log(feature_msg)
            
            if not is_picked:
                self._log(f"FILTERED: {', '.join(features['rejection_reasons'])}")
                self.status_var.set("Filtered")
                return
        
        # Sweep for best window using note model
        best_start, best_window, best_note_topk = self._sweep_best_window_note(
            captured=captured,
            sweep_step_samples=cs['sweep_step_samples'],
            topk=max(2, self.params['topk'])
        )
        
        if not best_note_topk:
            self._log("No note candidates found")
            return

        # Run string model on the SAME best window
        best_string_topk = self._predict_topk_string(best_window, topk=max(2, self.params['topk']))

        midi, name, note_prob, note_reason, note_ambig, f0_hz = self._decide_with_margin_and_pitch(
            best_window=best_window,
            best_topk=best_note_topk,
            sr=cs['sr'],
            margin=self.params['margin']
        )

        string_idx, string_name, string_prob, string_reason, string_ambig = self._decide_string_with_margin(
            best_topk=best_string_topk,
            margin=self.params['margin']
        )

        final_string_idx, final_fret, resolve_reason = self._resolve_position(midi, string_idx, best_string_topk)

        # UI updates
        self.note_display_var.set(f"{name}")
        self.confidence_var.set(f"Note conf: {note_prob:.3f} | String conf: {string_prob:.3f}")
        disp_s = self._to_display_string_num(final_string_idx)
        open_name = self._string_open_name(final_string_idx)
        self.midi_var.set(f"MIDI: {midi} | String: {disp_s} ({open_name}) | Fret: {final_fret}")

        reasons = []
        if note_ambig:
            reasons.append(f"note:{note_reason}")
        if string_ambig:
            reasons.append(f"string:{string_reason}")
        if resolve_reason:
            reasons.append(resolve_reason)
        self.reason_var.set(" | ".join(reasons))

        self.fretboard.show_position(final_string_idx, final_fret)
        self.status_light.set_state("detected")

        topk_note_str = ", ".join([f"{n}({p:.2f})" for m, n, p in best_note_topk[:self.params['topk']]])
        topk_str_str = ", ".join([f"S{self._to_display_string_num(si)}({p:.2f})" for si, sn, p in best_string_topk[:self.params['topk']]])

        disp_s = self._to_display_string_num(final_string_idx)
        self._log(f"Detected: {name} | Pos: S{disp_s} F{final_fret} | NoteTopK: {topk_note_str} | StringTopK: {topk_str_str}")

        if self.params['save_dir']:
            self._save_captured_audio(captured, best_window, f"{name}_S{final_string_idx+1}_F{final_fret}", note_prob)

    def _resolve_position(self, midi: int, string_idx_pred: int, string_topk):
        """Resolve final (string, fret) position from note and string predictions.
        
        Priority:
        1. Trust the string model's prediction if it yields a valid fret
        2. Try other strings by probability order
        3. Fall back to lowest valid fret position
        """
        tuning = GuitarFretboard.STANDARD_TUNING
        max_fret = int(self.fretboard.frets)

        # 1. Try predicted string first (trust the model!)
        fret = midi - tuning[string_idx_pred]
        if 0 <= fret <= max_fret:
            return string_idx_pred, fret, ""

        # 2. Try remaining strings by probability
        for si, _, _ in string_topk:
            if si == string_idx_pred:
                continue  # already tried
            fret = midi - tuning[si]
            if 0 <= fret <= max_fret:
                return si, fret, "pos: string_fallback"

        # 3. Final fallback: any valid position, prefer lowest fret
        best = None
        for si in range(6):
            fret = midi - tuning[si]
            if 0 <= fret <= max_fret:
                if best is None or fret < best[0]:
                    best = (fret, si)

        if best:
            fret, si = best
            return si, fret, "pos: lowest_fret_fallback"

        return 0, 0, "pos: no_valid_position"


    def _analyze_audio_features(self, audio, sr):
        """Analyze audio to determine if it's a picked note."""
        features = {}
        
        rms = np.sqrt(np.mean(audio ** 2))
        features['rms'] = float(rms)
        
        peak = float(np.max(np.abs(audio)))
        features['peak'] = peak
        features['peak_to_rms'] = peak / rms if rms > 1e-6 else 0
        
        zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        features['zcr'] = float(zcr)
        
        attack_samples = int(0.03 * sr)
        if len(audio) > attack_samples * 2:
            early_energy = np.mean(audio[:attack_samples] ** 2)
            later_energy = np.mean(audio[attack_samples:attack_samples*2] ** 2)
            features['attack_ratio'] = float(later_energy / early_energy) if early_energy > 1e-6 else 1.0
        else:
            features['attack_ratio'] = 1.0
        
        features['harmonic_strength'] = 0
        if len(audio) > sr // 50:
            norm_audio = audio - np.mean(audio)
            if np.std(norm_audio) > 0:
                norm_audio = norm_audio / np.std(norm_audio)
                min_lag = int(sr / 1200)
                max_lag = min(int(sr / 50), len(norm_audio) // 2)
                
                if max_lag > min_lag:
                    autocorr = np.correlate(norm_audio, norm_audio, mode='full')
                    autocorr = autocorr[len(autocorr)//2:]
                    if autocorr[0] > 0:
                        autocorr = autocorr / autocorr[0]
                    if max_lag < len(autocorr) and min_lag < len(autocorr):
                        autocorr_range = autocorr[min_lag:max_lag]
                        if len(autocorr_range) > 0:
                            features['harmonic_strength'] = float(np.max(np.abs(autocorr_range)))
        
        min_rms = self.filter_min_rms.get()
        min_peak_ratio = self.filter_min_peak_ratio.get()
        min_harmonics = self.filter_min_harmonics.get()
        max_attack_ratio = 2.0
        
        is_picked = True
        rejection_reasons = []
        
        if features['rms'] < min_rms:
            is_picked = False
            rejection_reasons.append(f"silence(rms={features['rms']:.5f})")
        elif features['peak_to_rms'] < min_peak_ratio:
            is_picked = False
            rejection_reasons.append(f"no_attack(p2r={features['peak_to_rms']:.1f})")
        elif features['attack_ratio'] > max_attack_ratio:
            is_picked = False
            rejection_reasons.append(f"reverse_attack({features['attack_ratio']:.2f})")
        elif features['harmonic_strength'] < min_harmonics:
            is_picked = False
            rejection_reasons.append(f"no_harmonics({features['harmonic_strength']:.2f})")
        
        features['is_picked'] = is_picked
        features['rejection_reasons'] = rejection_reasons
        
        return is_picked, features

    def _save_captured_audio(self, captured, best_window, name, prob):
        """Save captured audio."""
        try:
            save_dir = Path(self.params['save_dir'])
            save_dir.mkdir(parents=True, exist_ok=True)
            
            ts = time.strftime("%Y%m%d_%H%M%S")
            sr = self.capture_state['sr']
            attack = self.capture_state['attack']
            
            raw_path = save_dir / f"cap_{ts}_raw.wav"
            self._save_wav_int16(raw_path, sr, captured)
            
            win = best_window
            if win.shape[0] < attack:
                win = np.pad(win, (0, attack - win.shape[0]), mode='constant')
            else:
                win = win[:attack]
            
            best_path = save_dir / f"cap_{ts}_best_{name}_prob{prob:.2f}.wav"
            self._save_wav_int16(best_path, sr, win)
            
            self._log(f"Saved: {raw_path.name}")
            
        except Exception as e:
            self._log(f"Save failed: {e}")

    def _save_wav_int16(self, path, sr, audio_float32):
        """Save audio as 16-bit WAV."""
        x = np.clip(audio_float32.astype(np.float32), -1.0, 1.0)
        x_i16 = np.int16(x * 32767)
        wavfile.write(str(path), sr, x_i16)

    def stop_inference(self):
        """Stop the inference system."""
        if not self.is_running:
            return
        
        self.is_running = False
        self._stopping = True
        self.status_var.set("Stopping...")
        
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
        
        self.capture_state['mode'] = 'stopped'
        
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Ready")
        self.status_light.set_state("stopped")
        self.note_display_var.set("--")
        self.confidence_var.set("Note conf: -- | String conf: --")
        self.midi_var.set("MIDI: -- | String: -- | Fret: --")
        self.reason_var.set("")
        self.fretboard.clear_notes()
        
        self._stopping = False

    def _send_message(self, msg_type, data):
        """Send message to main thread."""
        self.message_queue.put((msg_type, data))

    def _update_message_display(self):
        """Process messages from background thread."""
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
                    self.stop_inference()
                    
        except queue.Empty:
            pass
        
        self.root.after(50, self._update_message_display)

    def on_closing(self):
        """Handle window close."""
        if self.is_running:
            self.stop_inference()
        self.root.destroy()

    # ============================
    # MODEL & INFERENCE METHODS
    # ============================

    def _get_model_device(self):
        """Get the best available compute device."""
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_checkpoints(self, note_path: Path, string_path: Path, model_device: str):
        """Load note+string checkpoints."""
        note_ckpt = torch.load(note_path, map_location="cpu", weights_only=False)
        string_ckpt = torch.load(string_path, map_location="cpu", weights_only=False)

        # ---- NOTE MODEL ----
        note_mel_params = note_ckpt["mel_params"]
        if "num_classes" in note_ckpt:
            note_num_classes = int(note_ckpt["num_classes"])
        elif "n_classes" in note_ckpt:
            note_num_classes = int(note_ckpt["n_classes"])
        else:
            if "idx_to_midi" in note_ckpt:
                note_num_classes = len(note_ckpt["idx_to_midi"])
            else:
                raise KeyError("Cannot determine num_classes for note model. "
                              f"Available keys: {list(note_ckpt.keys())}")
        note_n_mels = int(note_mel_params["n_mels"])

        note_model = NoteCNN(num_classes=note_num_classes, n_mels=note_n_mels, dropout_p=0.0)
        if "model_state" in note_ckpt:
            note_model.load_state_dict(note_ckpt["model_state"])
        else:
            note_model.load_state_dict(note_ckpt["model_state_dict"])
        note_model.to(model_device)
        note_model.eval()

        note_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=int(note_ckpt["sample_rate"]),
            n_fft=int(note_mel_params["n_fft"]),
            hop_length=int(note_mel_params["hop_length"]),
            n_mels=int(note_mel_params["n_mels"]),
            f_min=float(note_mel_params["f_min"]),
            f_max=float(note_mel_params["f_max"]),
        )
        note_db = torchaudio.transforms.AmplitudeToDB()

        idx_to_midi = note_ckpt["idx_to_midi"]
        if isinstance(idx_to_midi, dict):
            idx_to_midi = {int(k): int(v) for k, v in idx_to_midi.items()}

        # ---- STRING MODEL ----
        string_mel_params = string_ckpt["mel_params"]
        if "num_strings" in string_ckpt:
            string_num_classes = int(string_ckpt["num_strings"])
        elif "num_classes" in string_ckpt:
            string_num_classes = int(string_ckpt["num_classes"])
        elif "n_classes" in string_ckpt:
            string_num_classes = int(string_ckpt["n_classes"])
        else:
            string_num_classes = 6
        string_n_mels = int(string_mel_params["n_mels"])

        string_model = StringCNN(num_strings=string_num_classes, n_mels=string_n_mels, dropout_p=0.0)
        if "model_state" in string_ckpt:
            string_model.load_state_dict(string_ckpt["model_state"])
        else:
            string_model.load_state_dict(string_ckpt["model_state_dict"])
        string_model.to(model_device)
        string_model.eval()

        string_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=int(string_ckpt["sample_rate"]),
            n_fft=int(string_mel_params["n_fft"]),
            hop_length=int(string_mel_params["hop_length"]),
            n_mels=int(string_mel_params["n_mels"]),
            f_min=float(string_mel_params["f_min"]),
            f_max=float(string_mel_params["f_max"]),
        )
        string_db = torchaudio.transforms.AmplitudeToDB()

        # Ensure shared SR/attack for a single capture pipeline
        sample_rate = int(note_ckpt["sample_rate"])
        attack_samples = int(note_ckpt["attack_samples"])
        if int(string_ckpt["sample_rate"]) != sample_rate:
            raise RuntimeError(f"SR mismatch: note={sample_rate}, string={int(string_ckpt['sample_rate'])}")
        if int(string_ckpt["attack_samples"]) != attack_samples:
            raise RuntimeError(f"Attack mismatch: note={attack_samples}, string={int(string_ckpt['attack_samples'])}")

        return {
            "model_device": model_device,

            "note_model": note_model,
            "note_mel": note_mel,
            "note_db": note_db,
            "note_idx_to_midi": idx_to_midi,
            "note_num_frames": int(note_mel_params["num_frames"]),

            "string_model": string_model,
            "string_mel": string_mel,
            "string_db": string_db,
            "string_num_frames": int(string_mel_params["num_frames"]),

            "sample_rate": sample_rate,
            "attack_samples": attack_samples,
        }

    def _waveform_to_logmel(self, wav_1d: np.ndarray, mel, db, num_frames: int) -> torch.Tensor:
        """Convert waveform to log-mel spectrogram with per-sample normalization."""
        wav = wav_1d.astype(np.float32)
        wav = wav - float(np.mean(wav))
        wav_t = torch.from_numpy(wav).unsqueeze(0)

        spec = mel(wav_t)
        spec_db = db(spec)

        mean = spec_db.mean(dim=2, keepdim=True)
        std = spec_db.std(dim=2, keepdim=True) + 1e-6
        spec_db = (spec_db - mean) / std

        _, _, t = spec_db.shape
        if t < num_frames:
            spec_db = torch.nn.functional.pad(spec_db, (0, num_frames - t))
        elif t > num_frames:
            spec_db = spec_db[:, :, :num_frames]

        return spec_db.unsqueeze(0)  # [1,1,n_mels,num_frames]

    def _peak_normalize(self, window):
        """Peak normalize audio."""
        w = window.astype(np.float32)
        peak = float(np.max(np.abs(w))) if w.size else 0.0
        if peak > 0:
            w = (w / peak) * 0.95
        return w

    def _predict_topk_note(self, window_1d, topk, peak_normalize=True):
        """Run NOTE model inference on a window."""
        w = self._peak_normalize(window_1d) if peak_normalize else window_1d.astype(np.float32)

        state = self.state
        spec = self._waveform_to_logmel(
            w,
            mel=state["note_mel"],
            db=state["note_db"],
            num_frames=state["note_num_frames"],
        )
        spec = spec.to(state["model_device"], dtype=torch.float32)

        with torch.no_grad():
            logits = state["note_model"](spec)
            probs = torch.softmax(logits, dim=1)

        probs_cpu = probs.detach().to("cpu")
        k = min(int(topk), probs_cpu.shape[1])
        vals, idxs = torch.topk(probs_cpu, k=k, dim=1)

        out = []
        for p, idx in zip(vals[0].tolist(), idxs[0].tolist()):
            midi = int(state["note_idx_to_midi"][int(idx)])
            out.append((midi, self._midi_to_name(midi), float(p)))

        return out

    def _predict_topk_string(self, window_1d, topk, peak_normalize=True) -> List[Tuple[int, str, float]]:
        """Run STRING model inference on a window. Returns (string_idx, string_label, prob).
        
        Training data convention:
          CSV string 1 = high E (E4) -> model class 0
          CSV string 6 = low E (E2) -> model class 5
          
        Internal convention (for fretboard display):
          string_idx 0 = low E (E2)
          string_idx 5 = high E (E4)
          
        So we need to invert: model_class 0 -> string_idx 5, model_class 5 -> string_idx 0
        """
        w = self._peak_normalize(window_1d) if peak_normalize else window_1d.astype(np.float32)

        state = self.state
        spec = self._waveform_to_logmel(
            w,
            mel=state["string_mel"],
            db=state["string_db"],
            num_frames=state["string_num_frames"],
        )
        spec = spec.to(state["model_device"], dtype=torch.float32)

        with torch.no_grad():
            logits = state["string_model"](spec)
            probs = torch.softmax(logits, dim=1)

        probs_cpu = probs.detach().to("cpu")
        k = min(int(topk), probs_cpu.shape[1])
        vals, idxs = torch.topk(probs_cpu, k=k, dim=1)

        # Model class to string name (model class 0 = CSV string 1 = high E)
        model_class_to_name = ["E4", "B3", "G3", "D3", "A2", "E2"]

        out: List[Tuple[int, str, float]] = []
        for p, idx in zip(vals[0].tolist(), idxs[0].tolist()):
            model_class = int(idx)
            
            if not (0 <= model_class < 6):
                continue  # safety guard

            # Convert model class to internal string_idx (invert the order)
            # model_class 0 (high E) -> string_idx 5
            # model_class 5 (low E) -> string_idx 0
            string_idx = 5 - model_class
            
            label = model_class_to_name[model_class]
            out.append((string_idx, label, float(p)))

        return out

    def _sweep_best_window_note(self, captured, sweep_step_samples, topk):
        """Sweep to find best inference window based on NOTE confidence."""
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
            topk_list = self._predict_topk_note(window, topk, peak_normalize=True)
            if not topk_list:
                continue

            conf = topk_list[0][2]
            if conf > best_conf:
                best_conf = conf
                best_start = start
                best_window = window.copy()
                best_topk = topk_list

        return best_start, best_window, best_topk

    def _decide_string_with_margin(self, best_topk: List[Tuple[int, str, float]], margin: float):
        """Apply margin check for string prediction."""
        if not best_topk:
            return 0, "E2", 0.0, "no_candidates", True

        s1, n1, p1 = best_topk[0]
        if len(best_topk) < 2:
            return s1, n1, p1, "only_one_candidate", False

        s2, n2, p2 = best_topk[1]
        gap = p1 - p2
        if gap >= margin:
            return s1, n1, p1, f"margin_ok(gap={gap:.3f})", False

        return s1, n1, p1, f"ambiguous(gap={gap:.3f})", True

    def _decide_with_margin_and_pitch(self, best_window, best_topk, sr, margin):
        """Apply margin check and pitch tie-break (note model)."""
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
        """Tie-break between adjacent semitones using pitch detection."""
        f0 = self._estimate_f0(window, sr)
        if f0 is None:
            return None, None

        fa = self._midi_to_freq(midi_a)
        fb = self._midi_to_freq(midi_b)

        da = abs(f0 - fa)
        db = abs(f0 - fb)

        return (midi_a if da <= db else midi_b), float(f0)

    def _estimate_f0(self, x, sr, fmin=70.0, fmax=1200.0):
        """Estimate fundamental frequency using autocorrelation."""
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
        """Convert MIDI note to frequency."""
        return 440.0 * (2.0 ** ((midi - 69) / 12.0))

    def _midi_to_name(self, midi):
        """Convert MIDI note number to note name."""
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        note = names[midi % 12]
        octave = (midi // 12) - 1
        return f"{note}{octave}"


# ============================
# MAIN
# ============================

def main():
    root = tk.Tk()
    root.minsize(1000, 700)
    app = NoteInferenceGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()