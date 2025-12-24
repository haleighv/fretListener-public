#!/usr/bin/env python3
"""
train_gui.py - Unified Training Mix Builder and Model Trainer

A GUI application for building training mixes and training note/string/disambiguator models.

Features:
- Auto-scans datasets from fretListener/data
- Mode switching between Note, String, and Disambiguator
- Subprocess-based training with live log output
- Early stopping with patience parameter
"""

import sys
import threading
import queue
import subprocess
import os
from pathlib import Path
from typing import Optional, Dict

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from common.guitar import AMBIGUOUS_POSITIONS

# Colors
ACCENT_BLUE = "#4fc3f7"
ACCENT_GREEN = "#81c784"
ACCENT_ORANGE = "#ffb74d"
BG_DARK = "#1e1e1e"
BG_MID = "#2d2d2d"
FG = "#e0e0e0"
FG_DIM = "#888888"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _python_exec() -> str:
    return sys.executable or "python3"


def _is_dataset_root(p: Path) -> bool:
    if not p.is_dir():
        return False
    return (p / "meta" / "metadata.csv").is_file()


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self._tw: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", lambda e: self._show())
        widget.bind("<Leave>", lambda e: self._hide())

    def _show(self) -> None:
        if self._tw:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self._tw = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify="left", background="#ffffe0",
                        relief="solid", borderwidth=1, padx=6, pady=4)
        label.pack()

    def _hide(self) -> None:
        if self._tw:
            self._tw.destroy()
            self._tw = None


class TrainGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("fretListener - Training Mix + Train")
        self.minsize(1100, 800)
        self.configure(bg=BG_DARK)

        self._log_q: "queue.Queue[str]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._stop_requested = False

        self._dataset_vars: Dict[Path, tk.BooleanVar] = {}
        self._mode_var = tk.StringVar(value="note")

        self._build_ui()
        self._scan_and_populate_datasets()
        self._on_mode_change()
        self.after(50, self._drain_log_queue)

    def _build_ui(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".", background=BG_DARK, foreground=FG)
        style.configure("TFrame", background=BG_DARK)
        style.configure("TLabel", background=BG_DARK, foreground=FG)
        style.configure("TLabelframe", background=BG_DARK, foreground=FG)
        style.configure("TLabelframe.Label", background=BG_DARK, foreground=ACCENT_BLUE)
        style.configure("TEntry", fieldbackground=BG_MID, foreground=FG)
        style.configure("TSpinbox", fieldbackground=BG_MID, foreground=FG)
        style.configure("TRadiobutton", background=BG_DARK, foreground=FG)
        style.configure("TCheckbutton", background=BG_DARK, foreground=FG)
        style.map("TCombobox", fieldbackground=[("readonly", BG_MID)],
                  foreground=[("readonly", FG)])
        style.configure("TCombobox", fieldbackground=BG_MID, foreground=FG, background=BG_MID)

        root_frame = ttk.Frame(self, padding=12)
        root_frame.pack(fill="both", expand=True)

        # Mode selector
        mode_frame = ttk.Frame(root_frame)
        mode_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(mode_frame, text="Training Mode:", font=("Arial", 11, "bold")).pack(side="left")

        for val, txt in [("note", "Note"), ("string", "String"), ("disambig", "Disambiguator")]:
            rb = ttk.Radiobutton(mode_frame, text=txt, variable=self._mode_var,
                                value=val, command=self._on_mode_change)
            rb.pack(side="left", padx=(16, 0))

        self.mode_indicator_var = tk.StringVar(value="")
        self.mode_indicator = ttk.Label(mode_frame, textvariable=self.mode_indicator_var,
                                        font=("Arial", 10, "bold"))
        self.mode_indicator.pack(side="right", padx=(0, 10))

        # Top row: Datasets + Mix Params
        top = ttk.Frame(root_frame)
        top.pack(fill="x", pady=(0, 10))

        # Datasets
        ds_frame = ttk.LabelFrame(top, text="Datasets (from fretListener/data)", padding=10)
        ds_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        ds_toolbar = ttk.Frame(ds_frame)
        ds_toolbar.pack(fill="x", pady=(0, 8))
        ttk.Button(ds_toolbar, text="Rescan", command=self._scan_and_populate_datasets).pack(side="left")
        ttk.Button(ds_toolbar, text="Select All", command=self._select_all_datasets).pack(side="left", padx=(8, 0))
        ttk.Button(ds_toolbar, text="Select None", command=self._select_none_datasets).pack(side="left", padx=(8, 0))
        self.ds_count_var = tk.StringVar(value="")
        ttk.Label(ds_toolbar, textvariable=self.ds_count_var, foreground=FG_DIM).pack(side="right")

        self.ds_canvas = tk.Canvas(ds_frame, bg=BG_MID, highlightthickness=0, height=150)
        self.ds_canvas.pack(side="left", fill="both", expand=True)
        ds_scroll = ttk.Scrollbar(ds_frame, orient="vertical", command=self.ds_canvas.yview)
        ds_scroll.pack(side="left", fill="y")
        self.ds_canvas.configure(yscrollcommand=ds_scroll.set)
        self.ds_inner = ttk.Frame(self.ds_canvas)
        self.ds_window = self.ds_canvas.create_window((0, 0), window=self.ds_inner, anchor="nw")
        self.ds_inner.bind("<Configure>", lambda e: self.ds_canvas.configure(scrollregion=self.ds_canvas.bbox("all")))
        self.ds_canvas.bind("<Configure>", lambda e: self.ds_canvas.itemconfig(self.ds_window, width=e.width))

        # Mix parameters frame
        self.mix_frame = ttk.LabelFrame(top, text="Training Mix Parameters", padding=10)
        self.mix_frame.pack(side="left", fill="y")

        # Variables
        self.train_ratio_var = tk.StringVar(value="0.85")
        self.mix_seed_var = tk.StringVar(value="42")
        self.max_per_class_var = tk.StringVar(value="")
        self.mix_output_var = tk.StringVar(value=str(_project_root() / "recipes" / "note_training_mix.csv"))
        self.max_per_string_var = tk.StringVar(value="")
        self.stratify_fret_var = tk.BooleanVar(value=True)
        self.ambiguous_multiplier_var = tk.StringVar(value="2.0")
        self.string_mix_output_var = tk.StringVar(value=str(_project_root() / "recipes" / "string_training_mix.csv"))
        self.disambig_output_dir_var = tk.StringVar(value=str(_project_root() / "recipes" / "disambig"))

        self._build_note_mix_params()
        self._build_string_mix_params()
        self._build_disambig_mix_params()

        # Training parameters
        self.train_frame = ttk.LabelFrame(root_frame, text="Training Parameters", padding=10)
        self.train_frame.pack(fill="x", pady=(0, 10))

        self.batch_size_var = tk.StringVar(value="64")
        self.epochs_var = tk.StringVar(value="30")
        self.lr_var = tk.StringVar(value="1e-3")
        self.num_workers_var = tk.StringVar(value="0")
        self.train_seed_var = tk.StringVar(value="42")
        self.patience_var = tk.StringVar(value="0")
        self.n_mels_var = tk.StringVar(value="64")
        self.hop_length_var = tk.StringVar(value="128")
        self.num_frames_var = tk.StringVar(value="96")

        self.disambig_epochs_var = tk.StringVar(value="50")
        self.disambig_patience_var = tk.StringVar(value="10")

        self._build_training_params()

        # Actions
        actions = ttk.Frame(root_frame)
        actions.pack(fill="x", pady=(0, 10))

        self.build_mix_btn = tk.Button(actions, text="BUILD TRAINING MIX", command=self._start_build_mix,
                                       bg=ACCENT_BLUE, fg="#000", activebackground=ACCENT_BLUE,
                                       relief="flat", padx=14, pady=10)
        self.build_mix_btn.pack(side="left")

        self.run_train_btn = tk.Button(actions, text="RUN TRAINING", command=self._start_training,
                                       bg=ACCENT_GREEN, fg="#000", activebackground=ACCENT_GREEN,
                                       relief="flat", padx=14, pady=10)
        self.run_train_btn.pack(side="left", padx=(10, 0))

        self.stop_btn = tk.Button(actions, text="STOP", command=self._request_stop,
                                  bg="#ef5350", fg="#fff", activebackground="#ef5350",
                                  relief="flat", padx=14, pady=10, state="disabled")
        self.stop_btn.pack(side="left", padx=(10, 0))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(actions, textvariable=self.status_var, foreground=FG_DIM).pack(side="right")

        # Log area
        log_frame = ttk.LabelFrame(root_frame, text="Output Log", padding=10)
        log_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(log_frame, bg=BG_MID, fg=FG, insertbackground=FG,
                               wrap="word", font=("Consolas", 10), height=12)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=log_scroll.set)

    def _build_note_mix_params(self) -> None:
        self.note_mix_frame = ttk.Frame(self.mix_frame)
        row = 0
        ttk.Label(self.note_mix_frame, text="Max per class [blank=no cap]").grid(row=row, column=0, sticky="w")
        ttk.Entry(self.note_mix_frame, textvariable=self.max_per_class_var, width=24).grid(row=row, column=1, sticky="w", padx=(8, 0))
        row += 1
        ttk.Label(self.note_mix_frame, text="Train ratio").grid(row=row, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(self.note_mix_frame, textvariable=self.train_ratio_var, width=24).grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        row += 1
        ttk.Label(self.note_mix_frame, text="Seed").grid(row=row, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(self.note_mix_frame, textvariable=self.mix_seed_var, width=24).grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        row += 1
        ttk.Label(self.note_mix_frame, text="Output CSV").grid(row=row, column=0, sticky="w", pady=(8, 0))
        out_row = ttk.Frame(self.note_mix_frame)
        out_row.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        ttk.Entry(out_row, textvariable=self.mix_output_var, width=40).pack(side="left")
        ttk.Button(out_row, text="Browse…", command=lambda: self._browse_save_csv(self.mix_output_var)).pack(side="left", padx=(8, 0))

    def _build_string_mix_params(self) -> None:
        self.string_mix_frame = ttk.Frame(self.mix_frame)
        row = 0
        ttk.Label(self.string_mix_frame, text="Max per string [blank=no cap]").grid(row=row, column=0, sticky="w")
        ttk.Entry(self.string_mix_frame, textvariable=self.max_per_string_var, width=24).grid(row=row, column=1, sticky="w", padx=(8, 0))
        row += 1
        ttk.Label(self.string_mix_frame, text="Train ratio").grid(row=row, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(self.string_mix_frame, textvariable=self.train_ratio_var, width=24).grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        row += 1
        ttk.Label(self.string_mix_frame, text="Seed").grid(row=row, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(self.string_mix_frame, textvariable=self.mix_seed_var, width=24).grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        row += 1
        ttk.Checkbutton(self.string_mix_frame, text="Stratify by fret", variable=self.stratify_fret_var).grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))
        row += 1
        ttk.Label(self.string_mix_frame, text="Ambiguous multiplier").grid(row=row, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(self.string_mix_frame, textvariable=self.ambiguous_multiplier_var, width=24).grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        row += 1
        ttk.Label(self.string_mix_frame, text="Output CSV").grid(row=row, column=0, sticky="w", pady=(8, 0))
        out_row = ttk.Frame(self.string_mix_frame)
        out_row.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        ttk.Entry(out_row, textvariable=self.string_mix_output_var, width=40).pack(side="left")
        ttk.Button(out_row, text="Browse…", command=lambda: self._browse_save_csv(self.string_mix_output_var)).pack(side="left", padx=(8, 0))

    def _build_disambig_mix_params(self) -> None:
        self.disambig_mix_frame = ttk.Frame(self.mix_frame)
        row = 0
        ttk.Label(self.disambig_mix_frame, text="Output directory").grid(row=row, column=0, sticky="w")
        out_row = ttk.Frame(self.disambig_mix_frame)
        out_row.grid(row=row, column=1, sticky="w", padx=(8, 0))
        ttk.Entry(out_row, textvariable=self.disambig_output_dir_var, width=40).pack(side="left")
        ttk.Button(out_row, text="Browse…", command=self._browse_disambig_output_dir).pack(side="left", padx=(8, 0))
        row += 1
        ttk.Label(self.disambig_mix_frame, text="Seed").grid(row=row, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(self.disambig_mix_frame, textvariable=self.mix_seed_var, width=24).grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        row += 1
        info = ttk.Label(self.disambig_mix_frame, text="Builds mixes for ALL 5 string pairs:\nE/A, A/D, D/G, G/B, B/e", foreground=FG_DIM)
        info.grid(row=row, column=0, columnspan=2, sticky="w", pady=(12, 0))

    def _build_training_params(self) -> None:
        # Shared params frame
        self.shared_train_frame = ttk.Frame(self.train_frame)
        self.shared_train_frame.pack(fill="x")

        grid = self.shared_train_frame
        def add_field(r, c, label, var, width=12):
            ttk.Label(grid, text=label).grid(row=r, column=c, sticky="w", padx=(0, 8), pady=3)
            ttk.Entry(grid, textvariable=var, width=width).grid(row=r, column=c+1, sticky="w", padx=(0, 16), pady=3)

        add_field(0, 0, "Batch size", self.batch_size_var)
        add_field(0, 2, "Epochs", self.epochs_var)
        add_field(0, 4, "LR", self.lr_var)
        add_field(0, 6, "Patience", self.patience_var)

        add_field(1, 0, "Seed", self.train_seed_var)
        add_field(1, 2, "n_mels", self.n_mels_var)
        add_field(1, 4, "hop_length", self.hop_length_var)
        add_field(1, 6, "num_frames", self.num_frames_var)

        # Disambig-specific params frame (hidden by default)
        self.disambig_train_frame = ttk.Frame(self.train_frame)
        dgrid = self.disambig_train_frame
        ttk.Label(dgrid, text="Batch size").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(dgrid, textvariable=self.batch_size_var, width=12).grid(row=0, column=1, sticky="w", padx=(0, 16), pady=3)
        ttk.Label(dgrid, text="Epochs").grid(row=0, column=2, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(dgrid, textvariable=self.disambig_epochs_var, width=12).grid(row=0, column=3, sticky="w", padx=(0, 16), pady=3)
        ttk.Label(dgrid, text="Patience").grid(row=0, column=4, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(dgrid, textvariable=self.disambig_patience_var, width=12).grid(row=0, column=5, sticky="w", padx=(0, 16), pady=3)
        ttk.Label(dgrid, text="Seed").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(dgrid, textvariable=self.train_seed_var, width=12).grid(row=1, column=1, sticky="w", padx=(0, 16), pady=3)
        ttk.Label(dgrid, text="LR").grid(row=1, column=2, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(dgrid, textvariable=self.lr_var, width=12).grid(row=1, column=3, sticky="w", padx=(0, 16), pady=3)

    def _on_mode_change(self) -> None:
        mode = self._mode_var.get()
        # Hide all
        self.note_mix_frame.pack_forget()
        self.string_mix_frame.pack_forget()
        self.disambig_mix_frame.pack_forget()
        self.shared_train_frame.pack_forget()
        self.disambig_train_frame.pack_forget()

        if mode == "note":
            self.note_mix_frame.pack(fill="both", expand=True)
            self.shared_train_frame.pack(fill="x")
            self.mode_indicator_var.set("MODE: Note Classification")
            self.mode_indicator.configure(foreground=ACCENT_BLUE)
            self.train_frame.configure(text="Training Parameters (train_note_model.py)")
        elif mode == "string":
            self.string_mix_frame.pack(fill="both", expand=True)
            self.shared_train_frame.pack(fill="x")
            self.mode_indicator_var.set("MODE: String Classification")
            self.mode_indicator.configure(foreground=ACCENT_GREEN)
            self.train_frame.configure(text="Training Parameters (train_string_model.py)")
        else:  # disambig
            self.disambig_mix_frame.pack(fill="both", expand=True)
            self.disambig_train_frame.pack(fill="x")
            self.mode_indicator_var.set("MODE: Disambiguator (Batch)")
            self.mode_indicator.configure(foreground=ACCENT_ORANGE)
            self.train_frame.configure(text="Training Parameters (train_disambiguator.py --batch)")

    def _scan_and_populate_datasets(self) -> None:
        data_dir = _project_root() / "data"
        for widget in self.ds_inner.winfo_children():
            widget.destroy()
        self._dataset_vars.clear()

        if not data_dir.is_dir():
            self.ds_count_var.set("No data/ directory found")
            return

        datasets = sorted([p for p in data_dir.iterdir() if _is_dataset_root(p)])
        if not datasets:
            self.ds_count_var.set("No datasets found")
            return

        for p in datasets:
            var = tk.BooleanVar(value=True)
            self._dataset_vars[p] = var
            ttk.Checkbutton(self.ds_inner, text=p.name, variable=var).pack(anchor="w", pady=1)

        self.ds_count_var.set(f"{len(datasets)} dataset(s) found")

    def _select_all_datasets(self) -> None:
        for var in self._dataset_vars.values():
            var.set(True)

    def _select_none_datasets(self) -> None:
        for var in self._dataset_vars.values():
            var.set(False)

    def _browse_save_csv(self, var: tk.StringVar) -> None:
        f = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if f:
            var.set(f)

    def _browse_disambig_output_dir(self) -> None:
        d = filedialog.askdirectory()
        if d:
            self.disambig_output_dir_var.set(d)

    def _get_selected_datasets(self):
        return [str(p) for p, var in self._dataset_vars.items() if var.get()]

    def _log(self, msg: str) -> None:
        self._log_q.put(msg)

    def _drain_log_queue(self) -> None:
        try:
            while True:
                msg = self._log_q.get_nowait()
                self.log_text.insert(tk.END, msg + "\n")
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        self.after(50, self._drain_log_queue)

    def _request_stop(self) -> None:
        self._stop_requested = True
        self._log("Stop requested...")

    def _set_buttons_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        self.build_mix_btn.config(state=state)
        self.run_train_btn.config(state=state)
        self.stop_btn.config(state="normal" if running else "disabled")

    def _start_build_mix(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop_requested = False
        self._set_buttons_running(True)
        self.log_text.delete(1.0, tk.END)
        self._worker = threading.Thread(target=self._build_mix_thread, daemon=True)
        self._worker.start()

    def _build_mix_thread(self) -> None:
        try:
            mode = self._mode_var.get()
            datasets = self._get_selected_datasets()
            if not datasets:
                self._log("ERROR: No datasets selected")
                return

            if mode == "disambig":
                self._build_disambig_mix(datasets)
            else:
                self._build_note_or_string_mix(mode, datasets)
        except Exception as e:
            self._log(f"ERROR: {e}")
        finally:
            self.after(0, lambda: self._set_buttons_running(False))
            self.after(0, lambda: self.status_var.set("Ready"))

    def _build_note_or_string_mix(self, mode: str, datasets: list) -> None:
        script = _project_root() / "tools" / "build_training_mix.py"
        output = self.mix_output_var.get() if mode == "note" else self.string_mix_output_var.get()

        cmd = [_python_exec(), str(script), "--mode", mode, "--output", output,
               "--train-ratio", self.train_ratio_var.get(), "--seed", self.mix_seed_var.get(),
               "--datasets"] + datasets

        if mode == "note":
            if self.max_per_class_var.get().strip():
                cmd += ["--max-per-class", self.max_per_class_var.get()]
        else:
            if self.max_per_string_var.get().strip():
                cmd += ["--max-per-string", self.max_per_string_var.get()]
            if self.stratify_fret_var.get():
                cmd += ["--stratify-by-fret"]
            cmd += ["--ambiguous-multiplier", self.ambiguous_multiplier_var.get()]

        self._log(f"Running: {' '.join(cmd)}")
        self.status_var.set("Building mix...")
        self._run_subprocess(cmd)

    def _build_disambig_mix(self, datasets: list) -> None:
        script = _project_root() / "tools" / "build_disambig_mix.py"
        output_dir = self.disambig_output_dir_var.get()

        cmd = [_python_exec(), str(script), "--all-pairs", "--output-dir", output_dir,
               "--seed", self.mix_seed_var.get(), "--datasets"] + datasets

        self._log(f"Running: {' '.join(cmd)}")
        self.status_var.set("Building disambig mixes...")
        self._run_subprocess(cmd)

    def _start_training(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop_requested = False
        self._set_buttons_running(True)
        self.log_text.delete(1.0, tk.END)
        self._worker = threading.Thread(target=self._training_thread, daemon=True)
        self._worker.start()

    def _training_thread(self) -> None:
        try:
            mode = self._mode_var.get()
            if mode == "disambig":
                self._train_disambig()
            else:
                self._train_note_or_string(mode)
        except Exception as e:
            self._log(f"ERROR: {e}")
        finally:
            self.after(0, lambda: self._set_buttons_running(False))
            self.after(0, lambda: self.status_var.set("Ready"))

    def _train_note_or_string(self, mode: str) -> None:
        if mode == "note":
            script = _project_root() / "src" / "train_note_model.py"
            mix_csv = self.mix_output_var.get()
        else:
            script = _project_root() / "src" / "train_string_model.py"
            mix_csv = self.string_mix_output_var.get()

        cmd = [_python_exec(), str(script), "--mix-csv", mix_csv,
               "--batch-size", self.batch_size_var.get(),
               "--epochs", self.epochs_var.get(),
               "--lr", self.lr_var.get(),
               "--seed", self.train_seed_var.get(),
               "--n-mels", self.n_mels_var.get(),
               "--hop-length", self.hop_length_var.get(),
               "--num-frames", self.num_frames_var.get()]

        patience = self.patience_var.get().strip()
        if patience and int(patience) > 0:
            cmd += ["--patience", patience]

        self._log(f"Running: {' '.join(cmd)}")
        self.status_var.set(f"Training {mode} model...")
        self._run_subprocess(cmd)

    def _train_disambig(self) -> None:
        script = _project_root() / "tools" / "train_disambiguator.py"
        mix_dir = self.disambig_output_dir_var.get()

        cmd = [_python_exec(), str(script), "--batch", "--mix-dir", mix_dir,
               "--batch-size", self.batch_size_var.get(),
               "--epochs", self.disambig_epochs_var.get(),
               "--patience", self.disambig_patience_var.get(),
               "--lr", self.lr_var.get(),
               "--seed", self.train_seed_var.get()]

        self._log(f"Running: {' '.join(cmd)}")
        self.status_var.set("Training disambiguators (batch)...")
        self._run_subprocess(cmd)

    def _run_subprocess(self, cmd: list) -> None:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=env)
            for line in proc.stdout:
                if self._stop_requested:
                    proc.terminate()
                    self._log("Process terminated.")
                    break
                self._log(line.rstrip())
            proc.wait()
            if proc.returncode == 0:
                self._log("✓ Completed successfully")
            else:
                self._log(f"✗ Process exited with code {proc.returncode}")
        except Exception as e:
            self._log(f"ERROR: {e}")


def main():
    app = TrainGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
