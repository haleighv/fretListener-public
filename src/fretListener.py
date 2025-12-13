#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
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


class GuitarFretboard(tk.Canvas):
    """Visual guitar fretboard widget showing played notes"""
    
    # Standard tuning: E2, A2, D3, G3, B3, E4
    STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # MIDI notes
    STRING_NAMES = ['E', 'A', 'D', 'G', 'B', 'E']
    
    def __init__(self, parent, frets=7, **kwargs):
        super().__init__(parent, **kwargs)
        self.frets = frets
        self.active_notes: Set[Tuple[int, int]] = set()  # (string, fret)
        
        # Visual parameters
        self.string_spacing = 30
        self.fret_spacing = 60
        self.margin_left = 40
        self.margin_top = 30
        self.dot_radius = 12
        
        # Calculate canvas size
        width = self.margin_left + (self.frets + 1) * self.fret_spacing + 20
        height = 2 * self.margin_top + 5 * self.string_spacing
        self.config(width=width, height=height)
        
        self.draw_fretboard()
        
    def draw_fretboard(self):
        """Draw the fretboard with strings, frets, and markers"""
        self.delete("all")
        
        # Draw frets
        for fret in range(self.frets + 1):
            x = self.margin_left + fret * self.fret_spacing
            y1 = self.margin_top
            y2 = self.margin_top + 5 * self.string_spacing
            
            # Nut (thicker line)
            if fret == 0:
                self.create_line(x, y1, x, y2, width=4, fill="#8B4513")
            else:
                self.create_line(x, y1, x, y2, width=2, fill="#D2691E")
            
            # Fret numbers
            if fret > 0:
                self.create_text(x, y2 + 15, text=str(fret), 
                               font=("Arial", 10), fill="#666")
        
        # Draw strings
        for string in range(6):
            y = self.margin_top + string * self.string_spacing
            x1 = self.margin_left
            x2 = self.margin_left + self.frets * self.fret_spacing
            
            # String thickness varies
            thickness = 1 + (5 - string) * 0.3
            self.create_line(x1, y, x2, y, width=thickness, fill="#C0C0C0")
            
            # String labels
            self.create_text(x1 - 20, y, text=self.STRING_NAMES[5-string],
                           font=("Arial", 12, "bold"), fill="#333")
        
        # Draw fret markers (dots)
        marker_frets = [3, 5, 7]
        for fret in marker_frets:
            if fret <= self.frets:
                x = self.margin_left + (fret - 0.5) * self.fret_spacing
                y = self.margin_top + 2.5 * self.string_spacing
                self.create_oval(x - 5, y - 5, x + 5, y + 5,
                               fill="#E0E0E0", outline="#C0C0C0")
        
        # Redraw active notes
        for string, fret in self.active_notes:
            self._draw_note(string, fret, active=True)
    
    def _draw_note(self, string: int, fret: int, active: bool = True):
        """Draw a note on the fretboard"""
        if fret > self.frets or string < 0 or string > 5:
            return
            
        # Position calculation
        if fret == 0:
            x = self.margin_left - 15  # Open string position
        else:
            x = self.margin_left + (fret - 0.5) * self.fret_spacing
            
        y = self.margin_top + (5 - string) * self.string_spacing
        
        # Draw note circle
        color = "#FF4444" if active else "#FFD700"
        self.create_oval(
            x - self.dot_radius, y - self.dot_radius,
            x + self.dot_radius, y + self.dot_radius,
            fill=color, outline="#333", width=2, tags="note"
        )
    
    def midi_to_fretboard(self, midi_note: int) -> List[Tuple[int, int]]:
        """Convert MIDI note to possible (string, fret) positions"""
        positions = []
        for string, open_note in enumerate(self.STANDARD_TUNING):
            fret = midi_note - open_note
            if 0 <= fret <= self.frets:
                positions.append((string, fret))
        return positions
    
    def show_note(self, midi_note: int):
        """Highlight a MIDI note on the fretboard"""
        self.clear_notes()
        positions = self.midi_to_fretboard(midi_note)
        
        # Show all possible positions for this note
        for string, fret in positions:
            self.active_notes.add((string, fret))
            self._draw_note(string, fret, active=True)
    
    def clear_notes(self):
        """Clear all active notes"""
        self.delete("note")
        self.active_notes.clear()


class NoteInferenceGUI:
    """Main GUI application for real-time note inference"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("FretListener - Real-time Note Inference")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Configure style for dark mode compatibility
        self.setup_styles()
        
        # State
        self.inference_thread = None
        self.is_running = False
        self.message_queue = queue.Queue()
        self.state = None
        
        # Default parameters matching command line defaults
        self.params = {
            'checkpoint': 'artifacts/models/note_20251211_164726/note_cnn_best.pt',
            'trigger_threshold': 0.01,
            'pre_ms': 50.0,
            'post_ms': 500.0,
            'silence_threshold': 0.006,
            'min_silence_ms': 80.0,
            'sweep_step_ms': 5.0,
            'block_size': 512,
            'margin': 0.10,
            'topk': 5,
            'save_dir': 'artifacts/captures',
            'device_idx': None,
            'channel_idx': 0
        }
        
        self.setup_ui()
        self.update_message_display()
    
    def setup_styles(self):
        """Configure styles for better macOS compatibility"""
        # Define button styling for native tk buttons
        self.button_style = {
            'bg': '#f0f0f0',
            'fg': 'black',
            'activebackground': '#e0e0e0',
            'activeforeground': 'black',
            'highlightbackground': '#f0f0f0',
            'borderwidth': 1,
            'relief': tk.RAISED,
            'padx': 10,
            'pady': 5,
            'font': ('Arial', 11)
        }
        
        self.button_disabled_style = {
            'bg': '#f5f5f5',
            'fg': '#808080'
        }
        
    def setup_ui(self):
        """Create the GUI layout"""
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        
        # LEFT PANEL - Parameters
        left_panel = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Model section
        ttk.Label(left_panel, text="Model", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=3, pady=(0, 5))
        
        ttk.Label(left_panel, text="Checkpoint:").grid(row=1, column=0, sticky=tk.W)
        self.checkpoint_var = tk.StringVar(value=self.params['checkpoint'])
        checkpoint_entry = tk.Entry(left_panel, textvariable=self.checkpoint_var, width=30,
                                   bg='white', fg='black', insertbackground='black')
        checkpoint_entry.grid(row=1, column=1, padx=5)
        tk.Button(left_panel, text="Browse", command=self.browse_checkpoint,
                 **self.button_style).grid(row=1, column=2)
        
        # Audio Device section
        ttk.Label(left_panel, text="Audio Input", font=("Arial", 10, "bold")).grid(row=2, column=0, columnspan=3, pady=(15, 5))
        
        ttk.Label(left_panel, text="Device:").grid(row=3, column=0, sticky=tk.W)
        self.device_var = tk.StringVar()
        self.device_list = []
        self.device_menu = tk.OptionMenu(left_panel, self.device_var, "")
        self.device_menu.config(width=25, bg='white', fg='black', 
                               activebackground='#e0e0e0', activeforeground='black',
                               highlightbackground='white')
        self.device_menu.grid(row=3, column=1, padx=5, sticky=tk.W)
        self.device_var.trace('w', lambda *args: self.on_device_selected())
        
        ttk.Label(left_panel, text="Channel:").grid(row=4, column=0, sticky=tk.W)
        self.channel_var = tk.StringVar()
        self.channel_list = []
        self.channel_menu = tk.OptionMenu(left_panel, self.channel_var, "")
        self.channel_menu.config(width=25, bg='white', fg='black',
                                activebackground='#e0e0e0', activeforeground='black',
                                highlightbackground='white')
        self.channel_menu.grid(row=4, column=1, padx=5, sticky=tk.W)
        
        tk.Button(left_panel, text="Refresh Devices", command=self.refresh_devices,
                 **self.button_style).grid(row=3, column=2)
        
        # Trigger Parameters section
        ttk.Label(left_panel, text="Trigger Settings", font=("Arial", 10, "bold")).grid(row=5, column=0, columnspan=3, pady=(15, 5))
        
        params_frame = ttk.Frame(left_panel)
        params_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        self.param_vars = {}
        param_config = [
            ('trigger_threshold', 'Trigger Threshold:', 0.001, 0.1, 0.001),
            ('pre_ms', 'Pre-trigger (ms):', 0, 200, 10),
            ('post_ms', 'Post-trigger (ms):', 100, 2000, 50),
            ('silence_threshold', 'Silence Threshold:', 0.001, 0.05, 0.001),
            ('min_silence_ms', 'Min Silence (ms):', 10, 500, 10),
            ('sweep_step_ms', 'Sweep Step (ms):', 1, 50, 1),
            ('margin', 'Confidence Margin:', 0.0, 0.5, 0.01),
        ]
        
        for i, (key, label, min_val, max_val, step) in enumerate(param_config):
            ttk.Label(params_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            
            var = tk.DoubleVar(value=self.params[key])
            self.param_vars[key] = var
            
            slider = ttk.Scale(params_frame, from_=min_val, to=max_val, 
                              variable=var, orient=tk.HORIZONTAL, length=200)
            slider.grid(row=i, column=1, padx=5, pady=2)
            
            value_label = ttk.Label(params_frame, text=f"{var.get():.3f}", width=8)
            value_label.grid(row=i, column=2, padx=5)
            
            # Update label when slider moves
            def make_update_func(v, l):
                return lambda x: l.config(text=f"{v.get():.3f}")
            var.trace('w', make_update_func(var, value_label))
        
        # Other settings
        ttk.Label(left_panel, text="Other Settings", font=("Arial", 10, "bold")).grid(row=7, column=0, columnspan=3, pady=(15, 5))
        
        other_frame = ttk.Frame(left_panel)
        other_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        ttk.Label(other_frame, text="Top-K:").grid(row=0, column=0, sticky=tk.W)
        self.topk_var = tk.IntVar(value=self.params['topk'])
        tk.Spinbox(other_frame, from_=1, to=10, textvariable=self.topk_var, width=10,
                  bg='white', fg='black', insertbackground='black').grid(row=0, column=1, padx=5)
        
        ttk.Label(other_frame, text="Block Size:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.blocksize_var = tk.IntVar(value=self.params['block_size'])
        tk.Spinbox(other_frame, from_=128, to=2048, increment=128, 
                  textvariable=self.blocksize_var, width=10,
                  bg='white', fg='black', insertbackground='black').grid(row=1, column=1, padx=5)
        
        ttk.Label(other_frame, text="Save Dir:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.save_dir_var = tk.StringVar(value=self.params['save_dir'])
        tk.Entry(other_frame, textvariable=self.save_dir_var, width=20,
                bg='white', fg='black', insertbackground='black').grid(row=2, column=1, padx=5)
        
        self.save_enabled_var = tk.BooleanVar(value=True)
        tk.Checkbutton(other_frame, text="Save WAVs", 
                      variable=self.save_enabled_var,
                      bg='#f9f9f9', fg='black', activebackground='#f0f0f0',
                      selectcolor='white').grid(row=2, column=2, padx=5)
        
        # Control buttons
        control_frame = ttk.Frame(left_panel)
        control_frame.grid(row=9, column=0, columnspan=3, pady=20)
        
        self.start_button = tk.Button(control_frame, text="Start", command=self.start_inference,
                                      **self.button_style)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = tk.Button(control_frame, text="Stop", command=self.stop_inference,
                                     **self.button_style)
        self.stop_button.config(state="disabled")
        self.stop_button.grid(row=0, column=1, padx=5)
        
        # RIGHT PANEL - Visualization
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_panel.columnconfigure(0, weight=1)
        
        # Fretboard
        fretboard_frame = ttk.LabelFrame(right_panel, text="Guitar Fretboard (Frets 0-7)")
        fretboard_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.fretboard = GuitarFretboard(fretboard_frame, frets=7, bg="white", 
                                        highlightthickness=1, highlightbackground="#ccc")
        self.fretboard.pack(padx=10, pady=10)
        
        # Current Note Display
        note_frame = ttk.LabelFrame(right_panel, text="Detected Note")
        note_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        note_frame.columnconfigure(0, weight=1)
        
        # Create a white background frame for the note display
        note_bg_frame = tk.Frame(note_frame, bg="white", relief=tk.SUNKEN, borderwidth=2)
        note_bg_frame.grid(row=0, column=0, columnspan=1, sticky=(tk.W, tk.E), padx=10, pady=10)
        
        self.note_label = tk.Label(note_bg_frame, text="--", 
                                  font=("Arial", 72, "bold"),
                                  fg="#FF4444", bg="white")
        self.note_label.pack(pady=(20, 10))
        
        self.confidence_label = tk.Label(note_bg_frame, text="Confidence: --", 
                                        font=("Arial", 16),
                                        fg="black", bg="white")
        self.confidence_label.pack()
        
        self.midi_label = tk.Label(note_bg_frame, text="MIDI: --", 
                                  font=("Arial", 12),
                                  fg="black", bg="white")
        self.midi_label.pack(pady=(5, 15))
        
        # Debug Info
        debug_frame = ttk.LabelFrame(right_panel, text="Debug Information")
        debug_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        debug_frame.columnconfigure(0, weight=1)
        debug_frame.rowconfigure(0, weight=1)
        
        # Use a Text widget with explicit colors for better dark mode compatibility
        self.debug_text = scrolledtext.ScrolledText(debug_frame, height=10, width=60,
                                                   font=("Courier", 9),
                                                   bg="white", fg="black",
                                                   insertbackground="black")
        self.debug_text.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Initialize devices
        self.refresh_devices()
        
    def browse_checkpoint(self):
        """Browse for checkpoint file"""
        filename = filedialog.askopenfilename(
            title="Select checkpoint file",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if filename:
            self.checkpoint_var.set(filename)
    
    def refresh_devices(self):
        """Refresh audio device list"""
        try:
            devices = sd.query_devices()
            self.device_list = []
            
            for idx, device in enumerate(devices):
                if device.get('max_input_channels', 0) > 0:
                    device_str = f"{idx}: {device['name']} ({device['max_input_channels']} ch)"
                    self.device_list.append(device_str)
            
            # Update device OptionMenu
            menu = self.device_menu['menu']
            menu.delete(0, 'end')
            
            for device in self.device_list:
                menu.add_command(label=device, 
                               command=tk._setit(self.device_var, device))
            
            if self.device_list:
                self.device_var.set(self.device_list[0])
                self.on_device_selected()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get audio devices: {e}")
    
    def on_device_selected(self, event=None):
        """Handle device selection"""
        if not self.device_var.get():
            return
            
        try:
            device_idx = int(self.device_var.get().split(':')[0])
            device_info = sd.query_devices(device_idx)
            max_channels = device_info.get('max_input_channels', 0)
            
            self.channel_list = [f"Channel {i+1}" for i in range(max_channels)]
            
            # Update channel OptionMenu
            menu = self.channel_menu['menu']
            menu.delete(0, 'end')
            
            for channel in self.channel_list:
                menu.add_command(label=channel,
                               command=tk._setit(self.channel_var, channel))
            
            if self.channel_list:
                self.channel_var.set(self.channel_list[0])
                
        except Exception as e:
            print(f"Error updating channels: {e}")
    
    def update_params(self):
        """Update parameters from GUI"""
        self.params['checkpoint'] = self.checkpoint_var.get()
        self.params['save_dir'] = self.save_dir_var.get() if self.save_enabled_var.get() else ""
        self.params['topk'] = self.topk_var.get()
        self.params['block_size'] = self.blocksize_var.get()
        
        for key, var in self.param_vars.items():
            self.params[key] = var.get()
        
        # Get device and channel
        if self.device_var.get():
            self.params['device_idx'] = int(self.device_var.get().split(':')[0])
        
        if self.channel_var.get():
            channel_num = self.channel_var.get().replace('Channel ', '')
            self.params['channel_idx'] = int(channel_num) - 1
    
    def start_inference(self):
        """Start the inference thread"""
        if self.is_running:
            return
        
        self.update_params()
        
        # Validate checkpoint
        checkpoint_path = Path(self.params['checkpoint'])
        if not checkpoint_path.exists():
            messagebox.showerror("Error", f"Checkpoint file not found: {checkpoint_path}")
            return
        
        # Validate device
        if self.params['device_idx'] is None:
            messagebox.showerror("Error", "Please select an audio input device")
            return
        
        self.is_running = True
        self.start_button.config(state="disabled", **self.button_disabled_style)
        self.stop_button.config(state="normal", **self.button_style)
        self.status_var.set("Starting inference...")
        
        # Clear debug display
        self.debug_text.delete(1.0, tk.END)
        
        # Start inference thread
        self.inference_thread = threading.Thread(target=self.run_inference, daemon=True)
        self.inference_thread.start()
    
    def stop_inference(self):
        """Stop the inference thread"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_button.config(state="disabled", **self.button_disabled_style)
        self.status_var.set("Stopping...")
        
        # Wait a bit for thread to stop
        self.root.after(500, self.on_inference_stopped)
    
    def on_inference_stopped(self):
        """Called after inference stops"""
        self.start_button.config(state="normal", **self.button_style)
        self.status_var.set("Ready")
        self.note_label.config(text="--")
        self.confidence_label.config(text="Confidence: --")
        self.midi_label.config(text="MIDI: --")
        self.fretboard.clear_notes()
    
    def run_inference(self):
        """Run the inference loop (in separate thread)"""
        try:
            # Load model
            self.send_message('status', "Loading model...")
            model_device = self.get_model_device()
            self.send_message('debug', f"Using compute device: {model_device}")
            
            checkpoint_path = Path(self.params['checkpoint'])
            self.state = self.load_checkpoint(checkpoint_path, model_device)
            
            sr = self.state["sample_rate"]
            attack = self.state["attack_samples"]
            
            # Calculate sample counts
            pre_samples = int(round((self.params['pre_ms'] / 1000.0) * sr))
            post_samples = int(round((self.params['post_ms'] / 1000.0) * sr))
            capture_total = pre_samples + post_samples
            
            sweep_step_samples = max(1, int(round((self.params['sweep_step_ms'] / 1000.0) * sr)))
            min_silence_samples = max(1, int(round((self.params['min_silence_ms'] / 1000.0) * sr)))
            
            self.send_message('debug', f"Model sample rate: {sr} Hz")
            self.send_message('debug', f"Training attack window: {attack} samples ({attack/sr:.3f}s)")
            self.send_message('debug', f"Capture: pre={pre_samples} + post={post_samples} = {capture_total} samples")
            self.send_message('status', "Inference armed. Play notes...")
            
            # Setup audio stream with callback
            device_info = sd.query_devices(self.params['device_idx'])
            max_channels = device_info['max_input_channels']
            ch = self.params['channel_idx']
            
            # Shared state for callback
            audio_state = {
                'mode': 'armed',
                'pre_buffer': np.zeros((pre_samples, max_channels), dtype=np.float32) if pre_samples > 0 else None,
                'pre_idx': 0,
                'recording_parts': [],
                'recorded_post': 0,
                'silence_ctr': 0,
                'params': self.params,
                'sr': sr,
                'ch': ch,
                'pre_samples': pre_samples,
                'post_samples': post_samples,
                'capture_total': capture_total,
                'min_silence_samples': min_silence_samples,
                'sweep_step_samples': sweep_step_samples,
                'attack': attack,
            }
            
            def audio_callback(indata, frames, time_info, status):
                if not self.is_running:
                    return
                    
                if status:
                    self.send_message('debug', f"Audio status: {status}")
                
                self.process_audio_block(indata, frames, audio_state)
            
            # Run audio stream
            with sd.InputStream(
                device=self.params['device_idx'],
                channels=max_channels,
                samplerate=sr,
                dtype='float32',
                blocksize=self.params['block_size'],
                callback=audio_callback
            ):
                while self.is_running:
                    time.sleep(0.1)
                    
        except Exception as e:
            self.send_message('error', f"Inference error: {e}")
            self.is_running = False
    
    def process_audio_block(self, indata, frames, audio_state):
        """Process audio block in callback"""
        ch = audio_state['ch']
        chan = indata[:, ch].astype(np.float32, copy=False)
        peak = float(np.max(np.abs(chan))) if chan.size else 0.0
        
        if audio_state['mode'] == 'armed':
            # Update pre-buffer
            if audio_state['pre_buffer'] is not None:
                self.update_prebuffer(indata, audio_state)
            
            # Check for trigger
            if peak >= self.params['trigger_threshold']:
                # Get pre-buffer audio
                pre_audio = self.get_prebuffer_channel(audio_state)
                audio_state['recording_parts'] = [pre_audio, chan.astype(np.float32, copy=True)]
                audio_state['recorded_post'] = int(chan.shape[0])
                audio_state['mode'] = 'capturing'
                
                self.send_message('debug', f"Triggered! Peak={peak:.4f}")
                self.send_message('status', "Capturing...")
                
        elif audio_state['mode'] == 'capturing':
            audio_state['recording_parts'].append(chan.astype(np.float32, copy=True))
            audio_state['recorded_post'] += int(chan.shape[0])
            
            if audio_state['recorded_post'] >= audio_state['post_samples']:
                # Process captured audio
                cap = np.concatenate(audio_state['recording_parts']).astype(np.float32, copy=False)
                
                if cap.shape[0] > audio_state['capture_total']:
                    cap = cap[:audio_state['capture_total']]
                elif cap.shape[0] < audio_state['capture_total']:
                    cap = np.pad(cap, (0, audio_state['capture_total'] - cap.shape[0]), mode='constant')
                
                # Run inference
                self.process_captured_audio(cap, audio_state)
                
                audio_state['mode'] = 'waiting_silence'
                audio_state['silence_ctr'] = 0
                
        elif audio_state['mode'] == 'waiting_silence':
            if peak <= self.params['silence_threshold']:
                audio_state['silence_ctr'] += frames
            else:
                audio_state['silence_ctr'] = 0
            
            if audio_state['silence_ctr'] >= audio_state['min_silence_samples']:
                self.send_message('debug', "Silence detected, re-arming")
                self.send_message('status', "Armed")
                audio_state['mode'] = 'armed'
                audio_state['silence_ctr'] = 0
                audio_state['recording_parts'] = []
                audio_state['recorded_post'] = 0
                audio_state['pre_idx'] = 0
    
    def update_prebuffer(self, block, audio_state):
        """Update circular pre-buffer"""
        pre_buffer = audio_state['pre_buffer']
        pre_samples = audio_state['pre_samples']
        pre_idx = audio_state['pre_idx']
        
        frames = block.shape[0]
        if frames <= 0:
            return
        
        if frames > pre_samples:
            block = block[-pre_samples:]
            frames = block.shape[0]
        
        end = pre_idx + frames
        if end <= pre_samples:
            pre_buffer[pre_idx:end, :] = block.astype(np.float32, copy=False)
        else:
            first = pre_samples - pre_idx
            pre_buffer[pre_idx:, :] = block[:first].astype(np.float32, copy=False)
            remaining = frames - first
            pre_buffer[:remaining, :] = block[first:].astype(np.float32, copy=False)
        
        audio_state['pre_idx'] = (pre_idx + frames) % pre_samples
    
    def get_prebuffer_channel(self, audio_state):
        """Get pre-buffer for selected channel"""
        pre_buffer = audio_state['pre_buffer']
        ch = audio_state['ch']
        pre_samples = audio_state['pre_samples']
        pre_idx = audio_state['pre_idx']
        
        if pre_buffer is None or pre_samples <= 0:
            return np.zeros((0,), dtype=np.float32)
        
        if pre_idx == 0:
            return pre_buffer[:, ch].astype(np.float32, copy=True)
        
        return np.concatenate([
            pre_buffer[pre_idx:, ch],
            pre_buffer[:pre_idx, ch]
        ]).astype(np.float32, copy=True)
    
    def process_captured_audio(self, captured, audio_state):
        """Process captured audio and run inference"""
        # Sweep for best window
        best_start, best_window, best_topk = self.sweep_best_window(
            captured=captured,
            state=self.state,
            sweep_step_samples=audio_state['sweep_step_samples'],
            topk=max(2, self.params['topk'])
        )
        
        if not best_topk:
            self.send_message('debug', "No candidates found")
            return
        
        # Get top result with margin check
        midi, name, prob, reason, was_ambiguous, f0_hz = self.decide_with_margin_and_pitch(
            best_window=best_window,
            best_topk=best_topk,
            sr=audio_state['sr'],
            margin=self.params['margin']
        )
        
        # Update UI
        self.send_message('note', {
            'midi': midi,
            'name': name,
            'confidence': prob,
            'ambiguous': was_ambiguous,
            'reason': reason,
            'f0': f0_hz,
            'topk': best_topk[:self.params['topk']],
            'best_offset': best_start / audio_state['sr']
        })
        
        # Save WAV if enabled
        if self.params['save_dir']:
            self.save_captured_audio(captured, best_window, audio_state, name, prob)
    
    def save_captured_audio(self, captured, best_window, audio_state, name, prob):
        """Save captured audio to WAV files"""
        try:
            save_dir = Path(self.params['save_dir'])
            save_dir.mkdir(parents=True, exist_ok=True)
            
            ts = time.strftime("%Y%m%d_%H%M%S")
            raw_path = save_dir / f"cap_{ts}_raw.wav"
            best_path = save_dir / f"cap_{ts}_best_{name}_prob{prob:.2f}.wav"
            
            # Save raw capture
            self.save_wav_int16(raw_path, audio_state['sr'], captured)
            
            # Save best window
            win = best_window
            attack = audio_state['attack']
            if win.shape[0] < attack:
                win = np.pad(win, (0, attack - win.shape[0]), mode='constant')
            else:
                win = win[:attack]
            self.save_wav_int16(best_path, audio_state['sr'], win)
            
            self.send_message('debug', f"Saved: {raw_path.name} and {best_path.name}")
            
        except Exception as e:
            self.send_message('debug', f"Failed to save WAV: {e}")
    
    def save_wav_int16(self, path, sr, audio_float32):
        """Save audio as 16-bit WAV"""
        x = np.clip(audio_float32.astype(np.float32), -1.0, 1.0)
        x_i16 = np.int16(x * 32767)
        wavfile.write(str(path), sr, x_i16)
    
    def send_message(self, msg_type, data):
        """Send message to main thread"""
        self.message_queue.put((msg_type, data))
    
    def update_message_display(self):
        """Update UI from message queue (main thread)"""
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()
                
                if msg_type == 'status':
                    self.status_var.set(data)
                    
                elif msg_type == 'debug':
                    self.debug_text.insert(tk.END, f"{data}\n")
                    self.debug_text.see(tk.END)
                    
                elif msg_type == 'error':
                    messagebox.showerror("Error", data)
                    self.stop_inference()
                    
                elif msg_type == 'note':
                    # Update note display
                    self.note_label.config(text=data['name'])
                    self.confidence_label.config(text=f"Confidence: {data['confidence']:.3f}")
                    self.midi_label.config(text=f"MIDI: {data['midi']}")
                    
                    # Update fretboard
                    self.fretboard.show_note(data['midi'])
                    
                    # Add debug info
                    debug_msg = f"\nDetected: {data['name']} (MIDI {data['midi']})"
                    debug_msg += f"\nConfidence: {data['confidence']:.3f}"
                    debug_msg += f"\nReason: {data['reason']}"
                    
                    if data['ambiguous']:
                        debug_msg += f"\nAMBIGUOUS (margin < {self.params['margin']:.3f})"
                        if data['f0']:
                            debug_msg += f"\nF0 estimate: {data['f0']:.1f} Hz"
                    
                    debug_msg += f"\nBest offset: {data['best_offset']:.3f}s"
                    debug_msg += f"\nTop-{len(data['topk'])} candidates:"
                    for i, (m, n, p) in enumerate(data['topk'], 1):
                        debug_msg += f"\n  {i}) {n:<4s} MIDI {m:<3d} prob={p:.3f}"
                    
                    debug_msg += "\n" + "-"*50
                    
                    self.debug_text.insert(tk.END, debug_msg + "\n")
                    self.debug_text.see(tk.END)
                    
        except queue.Empty:
            pass
        finally:
            self.root.after(50, self.update_message_display)
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            self.stop_inference()
            time.sleep(0.5)
        self.root.destroy()
    
    # --- Model and inference methods (adapted from original script) ---
    
    def get_model_device(self):
        """Get compute device for model"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    def load_checkpoint(self, checkpoint_path, model_device):
        """Load model checkpoint"""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        
        num_classes = int(ckpt["num_classes"])
        mel_params = ckpt["mel_params"]
        n_mels = int(mel_params["n_mels"])
        
        model = NoteCNN(num_classes=num_classes, n_mels=n_mels, dropout_p=0.0)
        model.load_state_dict(ckpt["model_state"])
        model.to(model_device)
        model.eval()
        
        # Mel+DB transforms on CPU
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
            idx_to_midi_norm = {int(k): int(v) for k, v in idx_to_midi.items()}
        else:
            idx_to_midi_norm = idx_to_midi
        
        return {
            "model": model,
            "model_device": model_device,
            "mel": mel,
            "db": db,
            "idx_to_midi": idx_to_midi_norm,
            "sample_rate": int(ckpt["sample_rate"]),
            "attack_samples": int(ckpt["attack_samples"]),
            "num_frames": int(mel_params["num_frames"]),
        }
    
    def waveform_to_logmel_cpu(self, wav_1d, mel, db, num_frames):
        """Convert waveform to log-mel spectrogram on CPU"""
        wav = wav_1d.astype(np.float32)
        wav = wav - float(np.mean(wav))  # DC removal
        
        wav_t = torch.from_numpy(wav).unsqueeze(0)  # [1, T]
        
        spec = mel(wav_t)         # [1, n_mels, T]
        spec_db = db(spec)        # [1, n_mels, T]
        
        mean = spec_db.mean(dim=2, keepdim=True)
        std = spec_db.std(dim=2, keepdim=True) + 1e-6
        spec_db = (spec_db - mean) / std
        
        _, _, t = spec_db.shape
        if t < num_frames:
            spec_db = torch.nn.functional.pad(spec_db, (0, num_frames - t))
        elif t > num_frames:
            spec_db = spec_db[:, :, :num_frames]
        
        return spec_db.unsqueeze(0)  # [1, 1, n_mels, num_frames]
    
    def peak_normalize(self, window):
        """Peak normalize audio"""
        w = window.astype(np.float32)
        peak = float(np.max(np.abs(w))) if w.size else 0.0
        if peak > 0:
            w = (w / peak) * 0.95
        return w
    
    def predict_topk_window(self, state, window_1d, topk, peak_normalize=True):
        """Run inference on a window"""
        w = self.peak_normalize(window_1d) if peak_normalize else window_1d.astype(np.float32)
        
        spec = self.waveform_to_logmel_cpu(
            wav_1d=w,
            mel=state["mel"],
            db=state["db"],
            num_frames=state["num_frames"],
        )
        
        spec = spec.to(state["model_device"], dtype=torch.float32)
        with torch.no_grad():
            logits = state["model"](spec)
            probs = torch.softmax(logits, dim=1)
        
        probs_cpu = probs.detach().to("cpu")
        k = min(int(topk), probs_cpu.shape[1])
        vals, idxs = torch.topk(probs_cpu, k=k, dim=1)
        
        out = []
        for p, idx in zip(vals[0].tolist(), idxs[0].tolist()):
            midi = int(state["idx_to_midi"][int(idx)])
            out.append((midi, self.midi_to_name(midi), float(p)))
        
        return out, probs_cpu
    
    def sweep_best_window(self, captured, state, sweep_step_samples, topk):
        """Sweep to find best inference window"""
        attack = state["attack_samples"]
        
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
            
            topk_list, _ = self.predict_topk_window(
                state=state,
                window_1d=window,
                topk=topk,
                peak_normalize=True,
            )
            if not topk_list:
                continue
            
            conf = topk_list[0][2]
            if conf > best_conf:
                best_conf = conf
                best_start = start
                best_window = window.copy()
                best_topk = topk_list
        
        return best_start, best_window, best_topk
    
    def decide_with_margin_and_pitch(self, best_window, best_topk, sr, margin):
        """Apply margin check and pitch tie-break"""
        if not best_topk:
            return -1, "UNKNOWN", 0.0, "no_candidates", True, None
        
        m1, n1, p1 = best_topk[0]
        if len(best_topk) < 2:
            return m1, n1, p1, "only_one_candidate", False, None
        
        m2, n2, p2 = best_topk[1]
        gap = p1 - p2
        
        if gap >= margin:
            return m1, n1, p1, f"margin_ok(gap={gap:.3f}>=margin={margin:.3f})", False, None
        
        # Ambiguous under margin
        if abs(m1 - m2) == 1:
            chosen, f0 = self.pitch_tiebreak_adjacent(best_window, sr, m1, m2)
            if chosen is not None:
                if chosen == m1:
                    return m1, n1, p1, f"ambiguous_adjacent_pitch->top1", True, f0
                return m2, n2, p2, f"ambiguous_adjacent_pitch->top2", True, f0
            
            return m1, n1, p1, f"ambiguous_adjacent_no_pitch", True, None
        
        return m1, n1, p1, f"ambiguous_non_adjacent", True, None
    
    def estimate_f0_autocorr_hz(self, x, sr, fmin=70.0, fmax=1200.0):
        """Estimate fundamental frequency using autocorrelation"""
        if x.size < int(sr * 0.02):  # require ~20ms
            return None
        
        y = x.astype(np.float32)
        y = y - float(np.mean(y))
        y = self.peak_normalize(y)
        
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
    
    def midi_to_freq_hz(self, midi):
        """Convert MIDI note to frequency"""
        return 440.0 * (2.0 ** ((midi - 69) / 12.0))
    
    def pitch_tiebreak_adjacent(self, window, sr, midi_a, midi_b):
        """Tie-break between adjacent semitones using pitch detection"""
        f0 = self.estimate_f0_autocorr_hz(window, sr=sr)
        if f0 is None:
            return None, None
        
        fa = self.midi_to_freq_hz(midi_a)
        fb = self.midi_to_freq_hz(midi_b)
        
        da = abs(f0 - fa)
        db = abs(f0 - fb)
        
        return (midi_a if da <= db else midi_b), float(f0)
    
    def midi_to_name(self, midi):
        """Convert MIDI note number to note name"""
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        note = names[midi % 12]
        octave = (midi // 12) - 1
        return f"{note}{octave}"


def main():
    root = tk.Tk()
    app = NoteInferenceGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()