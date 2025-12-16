#!/usr/bin/env python3

import sys
import threading
import queue
import subprocess
from pathlib import Path
from typing import Optional, List, Dict
import os

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


ACCENT_BLUE = "#4fc3f7"
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
    meta_csv = p / "meta" / "metadata.csv"
    if not meta_csv.is_file():
        return False
    return True


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self._tw: Optional[tk.Toplevel] = None
        self._id_after: Optional[str] = None

        self.widget.bind("<Enter>", self._on_enter, add=True)
        self.widget.bind("<Leave>", self._on_leave, add=True)
        self.widget.bind("<ButtonPress>", self._on_leave, add=True)
        self.widget.bind("<Motion>", self._on_motion, add=True)

    def _on_enter(self, event=None) -> None:
        self._schedule_show()

    def _on_leave(self, event=None) -> None:
        self._cancel_schedule()
        self._hide()

    def _on_motion(self, event=None) -> None:
        if self._tw is not None:
            self._position()

    def _schedule_show(self) -> None:
        self._cancel_schedule()
        self._id_after = self.widget.after(250, self._show)

    def _cancel_schedule(self) -> None:
        if self._id_after is not None:
            try:
                self.widget.after_cancel(self._id_after)
            except Exception:
                pass
            self._id_after = None

    def _show(self) -> None:
        if self._tw is not None:
            return

        try:
            if not self.widget.winfo_viewable():
                return
        except Exception:
            return

        tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_attributes("-topmost", True)
        self._tw = tw

        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background=BG_MID,
            foreground=FG,
            relief="solid",
            borderwidth=1,
            font=("Arial", 10),
            padx=8,
            pady=6,
            wraplength=420,
        )
        label.pack()

        self._position()

    def _position(self) -> None:
        if self._tw is None:
            return
        try:
            x = self.widget.winfo_pointerx() + 12
            y = self.widget.winfo_pointery() + 16
            self._tw.geometry(f"+{x}+{y}")
        except Exception:
            pass

    def _hide(self) -> None:
        if self._tw is not None:
            try:
                self._tw.destroy()
            except Exception:
                pass
            self._tw = None


class TrainGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("fretListener - Training Mix + Train")
        self.minsize(1100, 700)
        self.configure(bg=BG_DARK)

        self._log_q: "queue.Queue[str]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._stop_requested = False

        self._dataset_vars: Dict[Path, tk.BooleanVar] = {}

        self._build_ui()
        self._scan_and_populate_datasets()
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

        style.map(
            "TCombobox",
            fieldbackground=[("readonly", BG_MID)],
            foreground=[("readonly", FG)],
            selectbackground=[("readonly", BG_MID)],
            selectforeground=[("readonly", FG)],
        )
        style.configure("TCombobox", fieldbackground=BG_MID, foreground=FG, background=BG_MID)

        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        top = ttk.Frame(root)
        top.pack(fill="x", pady=(0, 10))

        # --- Datasets (auto-scanned) ---
        ds_frame = ttk.LabelFrame(top, text="Datasets (auto-scanned from fretListener/data)", padding=10)
        ds_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        ds_toolbar = ttk.Frame(ds_frame)
        ds_toolbar.pack(fill="x", pady=(0, 8))

        self.rescan_btn = ttk.Button(ds_toolbar, text="Rescan", command=self._scan_and_populate_datasets)
        self.rescan_btn.pack(side="left")
        self.select_all_btn = ttk.Button(ds_toolbar, text="Select All", command=self._select_all_datasets)
        self.select_all_btn.pack(side="left", padx=(8, 0))
        self.select_none_btn = ttk.Button(ds_toolbar, text="Select None", command=self._select_none_datasets)
        self.select_none_btn.pack(side="left", padx=(8, 0))

        ToolTip(self.rescan_btn, "Rescan fretListener/data and repopulate the dataset list.")
        ToolTip(self.select_all_btn, "Check every dataset in the list.")
        ToolTip(self.select_none_btn, "Uncheck every dataset in the list.")

        self.ds_count_var = tk.StringVar(value="")
        ttk.Label(ds_toolbar, textvariable=self.ds_count_var, foreground=FG_DIM).pack(side="right")

        self.ds_canvas = tk.Canvas(ds_frame, bg=BG_MID, highlightthickness=0)
        self.ds_canvas.pack(side="left", fill="both", expand=True)

        ds_scroll = ttk.Scrollbar(ds_frame, orient="vertical", command=self.ds_canvas.yview)
        ds_scroll.pack(side="left", fill="y")
        self.ds_canvas.configure(yscrollcommand=ds_scroll.set)

        self.ds_inner = ttk.Frame(self.ds_canvas)
        self.ds_window = self.ds_canvas.create_window((0, 0), window=self.ds_inner, anchor="nw")

        def _on_inner_configure(event):
            self.ds_canvas.configure(scrollregion=self.ds_canvas.bbox("all"))

        def _on_canvas_configure(event):
            self.ds_canvas.itemconfig(self.ds_window, width=event.width)

        self.ds_inner.bind("<Configure>", _on_inner_configure)
        self.ds_canvas.bind("<Configure>", _on_canvas_configure)

        # --- Mix parameters ---
        mix_frame = ttk.LabelFrame(top, text="Training Mix Parameters", padding=10)
        mix_frame.pack(side="left", fill="y")

        self.max_per_class_var = tk.StringVar(value="")  # blank => None
        self.train_ratio_var = tk.StringVar(value="0.8")
        self.mix_seed_var = tk.StringVar(value="42")
        self.mix_output_var = tk.StringVar(value=str((_project_root() / "recipes" / "training_mix.csv")))

        row = 0
        ttk.Label(mix_frame, text="Max per class (pos) [blank = no cap]").grid(row=row, column=0, sticky="w")
        self.max_per_class_entry = ttk.Entry(mix_frame, textvariable=self.max_per_class_var, width=24)
        self.max_per_class_entry.grid(row=row, column=1, sticky="w", padx=(8, 0))
        ToolTip(
            self.max_per_class_entry,
            "Optional cap per position class (pos).\n"
            "Blank = no cap (use all samples per class).\n"
            "Use to prevent large datasets dominating training."
        )
        row += 1

        ttk.Label(mix_frame, text="Train ratio").grid(row=row, column=0, sticky="w", pady=(8, 0))
        self.train_ratio_entry = ttk.Entry(mix_frame, textvariable=self.train_ratio_var, width=24)
        self.train_ratio_entry.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        ToolTip(
            self.train_ratio_entry,
            "Train/val split ratio per class.\n"
            "0.8 = 80% train, 20% val.\n"
            "If a class has only 1 sample, it is forced into train."
        )
        row += 1

        ttk.Label(mix_frame, text="Seed").grid(row=row, column=0, sticky="w", pady=(8, 0))
        self.mix_seed_entry = ttk.Entry(mix_frame, textvariable=self.mix_seed_var, width=24)
        self.mix_seed_entry.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        ToolTip(
            self.mix_seed_entry,
            "Random seed used when shuffling samples and creating train/val splits.\n"
            "Same seed = repeatable splits."
        )
        row += 1

        ttk.Label(mix_frame, text="Output CSV").grid(row=row, column=0, sticky="w", pady=(8, 0))
        out_row = ttk.Frame(mix_frame)
        out_row.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        self.mix_output_entry = ttk.Entry(out_row, textvariable=self.mix_output_var, width=40)
        self.mix_output_entry.pack(side="left")
        self.mix_output_browse_btn = ttk.Button(out_row, text="Browse…", command=self._browse_mix_output)
        self.mix_output_browse_btn.pack(side="left", padx=(8, 0))
        ToolTip(
            self.mix_output_entry,
            "Where the training mix CSV will be written.\n"
            "This CSV is passed to the training script via --mix-csv."
        )
        ToolTip(self.mix_output_browse_btn, "Choose an output location for the training mix CSV.")
        row += 1

        # --- Training parameters ---
        train_frame = ttk.LabelFrame(root, text="Training Parameters (src/train_note_model.py)", padding=10)
        train_frame.pack(fill="x", pady=(0, 10))

        self.batch_size_var = tk.StringVar(value="64")
        self.epochs_var = tk.StringVar(value="30")
        self.lr_var = tk.StringVar(value="1e-3")
        self.num_workers_var = tk.StringVar(value="0")
        self.train_seed_var = tk.StringVar(value="42")
        self.ckpt_name_var = tk.StringVar(value="note_cnn_best.pt")
        self.train_script_var = tk.StringVar(value=str((_project_root() / "src" / "train_note_model.py")))

        grid = ttk.Frame(train_frame)
        grid.pack(fill="x")

        def add_field(r: int, c: int, label: str, var: tk.StringVar, width: int = 12) -> tk.Widget:
            ttk.Label(grid, text=label).grid(row=r, column=c, sticky="w", padx=(0, 8), pady=3)
            e = ttk.Entry(grid, textvariable=var, width=width)
            e.grid(row=r, column=c + 1, sticky="w", padx=(0, 16), pady=3)
            return e

        self.batch_size_entry = add_field(0, 0, "Batch size", self.batch_size_var)
        self.epochs_entry = add_field(0, 2, "Epochs", self.epochs_var)
        self.lr_entry = add_field(0, 4, "LR", self.lr_var)
        self.num_workers_entry = add_field(0, 6, "Num workers", self.num_workers_var)

        ToolTip(
            self.batch_size_entry,
            "Batch size: samples per optimization step.\n"
            "Higher = smoother gradients, more memory.\n"
            "Lower = noisier gradients, may generalize better."
        )
        ToolTip(
            self.epochs_entry,
            "Epochs: full passes over the training set.\n"
            "More epochs = longer training.\n"
            "LR scheduler may reduce LR during training."
        )
        ToolTip(
            self.lr_entry,
            "Learning rate: step size for weight updates.\n"
            "Default 1e-3 is a common starting point.\n"
            "Lower = more stable, slower.\n"
            "Higher = faster, may diverge."
        )
        ToolTip(
            self.num_workers_entry,
            "DataLoader workers for loading data.\n"
            "0 = run data loading in the main training process (most compatible on macOS).\n"
            "Higher can improve throughput but may add overhead."
        )

        self.train_seed_entry = add_field(1, 0, "Seed", self.train_seed_var)
        self.ckpt_name_entry = add_field(1, 2, "Checkpoint name", self.ckpt_name_var, width=20)

        ToolTip(
            self.train_seed_entry,
            "Training seed: affects shuffling order and initialization (as used by the training script).\n"
            "Set for repeatability."
        )
        ToolTip(
            self.ckpt_name_entry,
            "Filename of the best checkpoint saved inside the run artifacts directory.\n"
            "This is passed to the training script as --checkpoint-name."
        )

        ttk.Label(grid, text="Train script").grid(row=1, column=4, sticky="w", padx=(0, 8), pady=3)
        script_row = ttk.Frame(grid)
        script_row.grid(row=1, column=5, columnspan=3, sticky="w", pady=3)
        self.train_script_entry = ttk.Entry(script_row, textvariable=self.train_script_var, width=48)
        self.train_script_entry.pack(side="left")
        self.train_script_browse_btn = ttk.Button(script_row, text="Browse…", command=self._browse_train_script)
        self.train_script_browse_btn.pack(side="left", padx=(8, 0))

        ToolTip(
            self.train_script_entry,
            "Python training entrypoint.\n"
            "Must accept the flags passed by this GUI (e.g., --mix-csv, --batch-size, --epochs, --lr, ...)."
        )
        ToolTip(self.train_script_browse_btn, "Choose the training script to run (typically src/train_note_model.py).")

        # --- Actions (tk.Button for macOS-visible text) ---
        actions = ttk.Frame(root)
        actions.pack(fill="x", pady=(0, 10))

        self.build_mix_btn = tk.Button(
            actions,
            text="BUILD TRAINING MIX",
            command=self._start_build_mix,
            bg=ACCENT_BLUE,
            fg="#000000",
            activebackground=ACCENT_BLUE,
            activeforeground="#000000",
            relief="flat",
            padx=14,
            pady=10,
        )
        self.build_mix_btn.pack(side="left")

        self.run_train_btn = tk.Button(
            actions,
            text="RUN TRAINING",
            command=self._start_training,
            bg=ACCENT_BLUE,
            fg="#000000",
            activebackground=ACCENT_BLUE,
            activeforeground="#000000",
            relief="flat",
            padx=14,
            pady=10,
        )
        self.run_train_btn.pack(side="left", padx=(10, 0))

        self.stop_btn = tk.Button(
            actions,
            text="STOP (best-effort)",
            command=self._request_stop,
            bg=BG_MID,
            fg=FG,
            activebackground=BG_MID,
            activeforeground=FG,
            relief="flat",
            padx=14,
            pady=10,
            state="disabled",
        )
        self.stop_btn.pack(side="left", padx=(10, 0))

        ToolTip(
            self.build_mix_btn,
            "Build the training mix CSV using the selected datasets and mix parameters.\n"
            "This calls tools/build_training_mix.py via import and runs in a worker thread."
        )
        ToolTip(
            self.run_train_btn,
            "Run the training script using the current training parameters.\n"
            "Uses the mix CSV path from Output CSV as --mix-csv.\n"
            "Streams stdout/stderr into the log."
        )
        ToolTip(
            self.stop_btn,
            "Request stop.\n"
            "Sends a terminate() to the training subprocess (best-effort).\n"
            "May not immediately stop if the process is in an uninterruptible state."
        )

        # --- Log ---
        log_frame = ttk.LabelFrame(root, text="Log", padding=10)
        log_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(
            log_frame,
            wrap="word",
            height=18,
            bg=BG_MID,
            fg=FG,
            insertbackground=FG,
            selectbackground=ACCENT_BLUE,
            font=("Consolas", 10),
        )
        self.log_text.pack(side="left", fill="both", expand=True)

        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scroll.pack(side="left", fill="y")
        self.log_text.configure(yscrollcommand=log_scroll.set)

    # ------------------------
    # Dataset scanning + UI
    # ------------------------

    def _scan_and_populate_datasets(self) -> None:
        data_root = _project_root() / "data"
        if not data_root.exists():
            self._log(f"Warning: data directory not found: {data_root}")
            return

        datasets = []
        for p in sorted(data_root.iterdir(), key=lambda x: x.name):
            if _is_dataset_root(p):
                datasets.append(p)

        for child in list(self.ds_inner.winfo_children()):
            child.destroy()

        existing = dict(self._dataset_vars)
        self._dataset_vars = {}

        for ds in datasets:
            prev = existing.get(ds)
            var = prev if prev is not None else tk.BooleanVar(value=False)
            self._dataset_vars[ds] = var

            row = ttk.Frame(self.ds_inner)
            row.pack(fill="x", pady=2)

            cb = ttk.Checkbutton(row, variable=var)
            cb.pack(side="left")

            name = ds.name
            ttk.Label(row, text=name, foreground=FG).pack(side="left", padx=(6, 0))

            ttk.Label(row, text=str(ds), foreground=FG_DIM).pack(side="left", padx=(10, 0))

        self.ds_count_var.set(f"{len(datasets)} dataset(s) found")

        # Keep scroll position sane
        try:
            self.ds_canvas.yview_moveto(0.0)
        except Exception:
            pass

    def _select_all_datasets(self) -> None:
        for var in self._dataset_vars.values():
            var.set(True)

    def _select_none_datasets(self) -> None:
        for var in self._dataset_vars.values():
            var.set(False)

    def _selected_dataset_paths(self) -> List[Path]:
        out: List[Path] = []
        for p, var in self._dataset_vars.items():
            try:
                if bool(var.get()):
                    out.append(p.resolve())
            except Exception:
                pass
        return out

    # ------------------------
    # Browsers
    # ------------------------

    def _browse_mix_output(self) -> None:
        p = filedialog.asksaveasfilename(
            title="Select output CSV path",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            initialfile="training_mix.csv",
        )
        if not p:
            return
        self.mix_output_var.set(str(Path(p).expanduser().resolve()))

    def _browse_train_script(self) -> None:
        p = filedialog.askopenfilename(
            title="Select train script",
            filetypes=[("Python", "*.py"), ("All files", "*.*")],
            initialdir=str(_project_root() / "src"),
        )
        if not p:
            return
        self.train_script_var.set(str(Path(p).expanduser().resolve()))

    # ------------------------
    # Logging / worker control
    # ------------------------

    def _log(self, s: str) -> None:
        self._log_q.put(s)

    def _drain_log_queue(self) -> None:
        try:
            while True:
                s = self._log_q.get_nowait()
                self.log_text.insert("end", s + "\n")
                self.log_text.see("end")
        except queue.Empty:
            pass
        self.after(50, self._drain_log_queue)

    def _set_busy(self, busy: bool) -> None:
        if busy:
            self.build_mix_btn.configure(state="disabled")
            self.run_train_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
        else:
            self.build_mix_btn.configure(state="normal")
            self.run_train_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")

    def _request_stop(self) -> None:
        self._stop_requested = True
        self._log("Stop requested (best-effort).")

    def _start_worker(self, target) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showwarning("Busy", "Another job is already running.")
            return
        self._stop_requested = False
        self._set_busy(True)
        t = threading.Thread(target=target, daemon=True)
        self._worker = t
        t.start()

    # ------------------------
    # Mix build
    # ------------------------

    def _parse_int_optional(self, s: str) -> Optional[int]:
        s = (s or "").strip()
        if not s:
            return None
        return int(s)

    def _parse_float(self, s: str) -> float:
        return float((s or "").strip())

    def _start_build_mix(self) -> None:
        def work():
            try:
                dataset_roots = self._selected_dataset_paths()
                if not dataset_roots:
                    self._log("Error: no datasets selected.")
                    return

                max_per_class = self._parse_int_optional(self.max_per_class_var.get())
                train_ratio = self._parse_float(self.train_ratio_var.get())
                seed = int((self.mix_seed_var.get() or "42").strip())
                output_csv = Path(self.mix_output_var.get()).expanduser().resolve()

                tools_dir = _project_root() / "tools"
                sys.path.insert(0, str(tools_dir))
                try:
                    import build_training_mix as btm  # type: ignore
                finally:
                    try:
                        sys.path.remove(str(tools_dir))
                    except Exception:
                        pass

                self._log("Building training mix...")
                self._log("Datasets:")
                for p in dataset_roots:
                    self._log(f"  - {p}")

                self._log(f"Max per class: {max_per_class if max_per_class is not None else 'no cap'}")
                self._log(f"Train ratio:   {train_ratio}")
                self._log(f"Seed:          {seed}")
                self._log(f"Output:        {output_csv}")

                btm.build_training_mix(
                    dataset_roots=dataset_roots,
                    max_per_class=max_per_class,
                    train_ratio=train_ratio,
                    output_csv=output_csv,
                    seed=seed,
                )
                self._log("Training mix build complete.")
            except Exception as e:
                self._log(f"Error building mix: {e}")
            finally:
                self._set_busy(False)

        self._start_worker(work)

    # ------------------------
    # Training run
    # ------------------------

    def _start_training(self) -> None:
        def work():
            proc: Optional[subprocess.Popen] = None
            try:
                mix_csv = Path(self.mix_output_var.get()).expanduser().resolve()
                if not mix_csv.is_file():
                    self._log(f"Error: mix CSV not found: {mix_csv}")
                    return

                train_script = Path(self.train_script_var.get()).expanduser().resolve()
                if not train_script.is_file():
                    self._log(f"Error: train script not found: {train_script}")
                    return

                cmd = [
                    _python_exec(),
                    "-u",
                    str(train_script),
                    "--mix-csv", str(mix_csv),
                    "--batch-size", (self.batch_size_var.get() or "64").strip(),
                    "--epochs", (self.epochs_var.get() or "30").strip(),
                    "--lr", (self.lr_var.get() or "1e-3").strip(),
                    "--num-workers", (self.num_workers_var.get() or "0").strip(),
                    "--seed", (self.train_seed_var.get() or "42").strip(),
                    "--checkpoint-name", (self.ckpt_name_var.get() or "note_cnn_best.pt").strip(),
                ]

                self._log("Starting training...")
                self._log(" ".join(cmd))

                env = dict(**{k: v for k, v in ({}).items()})
                env.update(dict(os.environ))
                env["PYTHONUNBUFFERED"] = "1"

                proc = subprocess.Popen(
                    cmd,
                    cwd=str(_project_root()),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    env=env,
                )

                self._log("Training process started; waiting for first output...")

                assert proc.stdout is not None
                for line in proc.stdout:
                    if self._stop_requested and proc.poll() is None:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                    self._log(line.rstrip("\n"))

                rc = proc.wait()
                self._log(f"Training finished (exit code {rc}).")

            except Exception as e:
                self._log(f"Error running training: {e}")
            finally:
                try:
                    if proc is not None and proc.poll() is None:
                        proc.terminate()
                except Exception:
                    pass
                self._set_busy(False)

        self._start_worker(work)


def main() -> None:
    app = TrainGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
