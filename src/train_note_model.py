#!/usr/bin/env python

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import shutil
import sys
import platform

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from note_model import NoteCNN

'''
Example usage:

cd ~/code/fretListener

python src/train_note_model.py \
  --mix-csv recipes/training_mix_neck.csv \
  --batch-size 64 \
  --epochs 30

'''
# -----------------------------
# Device selection
# -----------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------
# Dataset
# -----------------------------

class NoteDatasetFromMix(Dataset):
    """
    Dataset for note+octave classification from a training_mix CSV.

    Labels are MIDI-based classes:
      - All unique MIDI values across the mix are collected
      - Each MIDI is mapped to a contiguous class index [0..N-1]

    Returns:
      spec: Float tensor [1, n_mels, num_frames]
      label: int class index (for CrossEntropy)
    """

    def __init__(
        self,
        rows: List[dict],
        midi_to_idx: Dict[int, int],
        sample_rate: int = 48000,
        attack_duration: float = 0.25,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 48,
        f_min: float = 55.0,
        f_max: float = 3000.0,
        num_frames: int = 32,
        device: Optional[torch.device] = None,
        augment: bool = False,
    ) -> None:
        super().__init__()

        self.rows = rows
        self.midi_to_idx = midi_to_idx
        self.sample_rate = sample_rate
        self.attack_duration = attack_duration
        self.attack_samples = int(round(sample_rate * attack_duration))
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.num_frames = num_frames
        self.augment = augment  # currently unused (augmentation disabled)

        # Audio transforms
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        if device is not None:
            self.mel_spec = self.mel_spec.to(device)
            self.db_transform = self.db_transform.to(device)
            self.transform_device = device
        else:
            self.transform_device = None

    def __len__(self) -> int:
        return len(self.rows)

    def _load_audio(self, path_str: str) -> torch.Tensor:
        """
        Load audio, convert to mono, resample to target SR, crop/pad to fixed attack window.
        """
        path = Path(path_str)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        wav, sr = torchaudio.load(path)  # [C, T], float32
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(
                wav, orig_freq=sr, new_freq=self.sample_rate
            )

        if wav.size(1) < self.attack_samples:
            pad = self.attack_samples - wav.size(1)
            wav = torch.nn.functional.pad(wav, (0, pad))
        elif wav.size(1) > self.attack_samples:
            wav = wav[:, : self.attack_samples]

        return wav

    def _augment_waveform(self, wav: torch.Tensor) -> torch.Tensor:
        """
        (Currently unused) Light augmentation that preserves pitch:
          - small gain jitter
          - low-level additive noise
          - tiny time shift within the attack window
        """
        # Small gain jitter: +/- ~1 dB
        gain_db = np.random.uniform(-1.0, 1.0)
        gain = 10.0 ** (gain_db / 20.0)
        wav = wav * float(gain)

        # Add low-level noise relative to signal max
        max_val = wav.abs().max().item()
        if max_val > 0:
            noise_std = 0.005 * max_val
            noise = torch.randn_like(wav) * noise_std
            wav = wav + noise

        # Tiny time shift: up to ~2 ms
        max_shift_sec = 0.002
        max_shift_samples = int(self.sample_rate * max_shift_sec)
        if max_shift_samples > 0:
            shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)
            if shift != 0:
                wav = torch.roll(wav, shifts=shift, dims=1)

        return wav

    def _waveform_to_logmel(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Convert [1, T] waveform to [1, n_mels, num_frames] log-mel tensor.
        """
        device = self.transform_device or wav.device
        wav = wav.to(device)

        spec = self.mel_spec(wav)         # [1, n_mels, T_spec]
        spec_db = self.db_transform(spec) # [1, n_mels, T_spec]

        # Per-sample normalization over time
        # mean/std over time dimension (dim=2), keep frequency structure
        mean = spec_db.mean(dim=2, keepdim=True)
        std = spec_db.std(dim=2, keepdim=True) + 1e-6
        spec_db = (spec_db - mean) / std

        _, n_mels, T = spec_db.shape
        if T < self.num_frames:
            pad = self.num_frames - T
            spec_db = torch.nn.functional.pad(spec_db, (0, pad))
        elif T > self.num_frames:
            spec_db = spec_db[:, :, : self.num_frames]

        return spec_db  # [1, n_mels, num_frames]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.rows[idx]
        abs_path = row["abs_path"]
        midi = row["midi"]
        label = self.midi_to_idx[midi]

        wav = self._load_audio(abs_path)

        # Augmentation disabled for now, even if self.augment is True
        # if self.augment:
        #     wav = self._augment_waveform(wav)

        spec = self._waveform_to_logmel(wav)

        return spec.cpu(), int(label)


# -----------------------------
# CSV loading and mapping
# -----------------------------

def load_mix_csv(csv_path: Path) -> Tuple[List[dict], List[dict]]:
    """
    Load training mix CSV and split into train/val row lists.

    Assumes CSV has:
      - abs_path
      - midi
      - split in {"train", "val"}
    """
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Training mix CSV not found: {csv_path}")

    train_rows: List[dict] = []
    val_rows: List[dict] = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.get("split")
            abs_path = row.get("abs_path")
            midi_str = row.get("midi")

            if split not in ("train", "val"):
                continue
            if not abs_path or midi_str is None:
                continue

            try:
                midi = int(midi_str)
            except ValueError:
                continue

            row["midi"] = midi
            row["abs_path"] = abs_path

            if split == "train":
                train_rows.append(row)
            else:
                val_rows.append(row)

    if not train_rows:
        raise RuntimeError(f"No train rows found in {csv_path}")
    if not val_rows:
        raise RuntimeError(f"No val rows found in {csv_path}")

    return train_rows, val_rows


def build_midi_class_mapping(
    train_rows: List[dict],
    val_rows: List[dict],
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Build MIDI -> class index map and reverse.

    Uses all MIDI values seen in train + val to define the set of classes.
    """
    midi_values = set()
    for row in train_rows + val_rows:
        midi_values.add(row["midi"])

    midi_list = sorted(midi_values)
    midi_to_idx: Dict[int, int] = {m: i for i, m in enumerate(midi_list)}
    idx_to_midi: Dict[int, int] = {i: m for i, m in enumerate(midi_list)}

    return midi_to_idx, idx_to_midi


# -----------------------------
# Training loop helpers
# -----------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for specs, labels in loader:
        specs = specs.to(device, dtype=torch.float32)  # [B, 1, n_mels, T]
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for specs, labels in loader:
            specs = specs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            logits = model(specs)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc


def compute_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> np.ndarray:
    """
    Compute confusion matrix over the given loader using the provided model.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()

    with torch.no_grad():
        for specs, labels in loader:
            specs = specs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            logits = model(specs)
            preds = logits.argmax(dim=1)

            y_true = labels.cpu().numpy()
            y_pred = preds.cpu().numpy()

            for t, p in zip(y_true, y_pred):
                cm[int(t), int(p)] += 1

    return cm


# -----------------------------
# Artifact utilities
# -----------------------------

def get_project_root() -> Path:
    """
    Assumes this script lives in <root>/src/train_note_model.py
    """
    return Path(__file__).resolve().parent.parent


def create_run_dir(model_type: str = "note") -> Path:
    """
    Create a unique artifacts subdirectory for this training run.

    Layout:
      <project_root>/artifacts/models/<model_type>_YYYYMMDD_HHMMSS[_NN]
    """
    root = get_project_root()
    artifacts_root = root / "artifacts" / "models"
    artifacts_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{model_type}_{ts}"
    run_dir = artifacts_root / base_name

    suffix = 1
    while run_dir.exists():
        run_dir = artifacts_root / f"{base_name}_{suffix:02d}"
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def plot_training_curves(
    history: Dict[str, List[float]],
    run_dir: Path,
) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curves.png")
    plt.close()

    # Accuracy curves
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(run_dir / "accuracy_curves.png")
    plt.close()


def plot_lr_curve(
    history: Dict[str, List[float]],
    run_dir: Path,
) -> None:
    if "lr" not in history or not history["lr"]:
        return
    epochs = range(1, len(history["lr"]) + 1)
    plt.figure()
    plt.plot(epochs, history["lr"], label="lr")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(run_dir / "lr_curve.png")
    plt.close()


def plot_confusion_matrix(
    confusion: np.ndarray,
    run_dir: Path,
    title: str = "Confusion Matrix (Validation)",
) -> None:
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted class index")
    plt.ylabel("True class index")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png")
    plt.close()


def plot_per_class_accuracy(
    confusion: np.ndarray,
    idx_to_midi: Dict[int, int],
    run_dir: Path,
) -> None:
    true_counts = confusion.sum(axis=1)
    correct = np.diag(confusion)
    acc = np.zeros_like(correct, dtype=np.float32)
    nonzero = true_counts > 0
    acc[nonzero] = correct[nonzero] / true_counts[nonzero]

    # Label with MIDI (stable across runs)
    labels = [str(idx_to_midi[i]) for i in range(len(acc))]

    plt.figure(figsize=(12, 4))
    plt.plot(range(len(acc)), acc, marker="o", linestyle="-", label="per-class acc")
    plt.xticks(ticks=range(len(acc)), labels=labels, rotation=90)
    plt.ylim(0.0, 1.0)
    plt.xlabel("MIDI")
    plt.ylabel("Accuracy")
    plt.title("Per-class Accuracy (Validation)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(run_dir / "per_class_accuracy.png")
    plt.close()


def export_onnx_model(
    model: nn.Module,
    run_dir: Path,
    num_mels: int = 48,
    num_frames: int = 32,
) -> None:
    """
    Export the model to an ONNX file for Netron visualization.
    """
    model.cpu().eval()
    dummy_input = torch.randn(1, 1, num_mels, num_frames, dtype=torch.float32)
    onnx_path = run_dir / "note_cnn.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["logmel"],
        output_names=["logits"],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes={
            "logmel": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )


def write_summary(
    run_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    num_classes: int,
    midi_to_idx: Dict[int, int],
    idx_to_midi: Dict[int, int],
    train_len: int,
    val_len: int,
    history: Dict[str, List[float]],
    best_epoch: int,
    best_val_acc: float,
    best_val_loss: float,
    confusion: np.ndarray,
    dataset_params: Dict[str, float],
    model_params: Dict[str, float],
    optimizer_params: Dict[str, float],
    scheduler_params: Dict[str, float],
) -> None:
    summary_path = run_dir / "summary.txt"
    with summary_path.open("w") as f:
        f.write("Note CNN Training Summary\n")
        f.write("=========================\n\n")
        f.write(f"Run directory: {run_dir}\n")
        f.write(f"Mix CSV:       {Path(args.mix_csv).resolve()}\n")
        f.write(f"Device:        {device.type}\n")
        f.write(f"Seed:          {args.seed}\n")
        f.write(f"Timestamp:     {datetime.now().isoformat(timespec='seconds')}\n\n")

        f.write("Repro / Environment:\n")
        f.write(f"  Python:      {sys.version.split()[0]}\n")
        f.write(f"  Platform:    {platform.platform()}\n")
        f.write(f"  Torch:       {torch.__version__}\n")
        f.write(f"  Torchaudio:  {torchaudio.__version__}\n\n")

        f.write("Command:\n")
        f.write("  " + " ".join(sys.argv) + "\n\n")

        f.write("Hyperparameters (CLI):\n")
        f.write(f"  Batch size:  {args.batch_size}\n")
        f.write(f"  Epochs:      {args.epochs}\n")
        f.write(f"  Learning rate (initial): {args.lr}\n")
        f.write(f"  Num workers: {args.num_workers}\n")
        f.write(f"  n_mels:      {args.n_mels}\n")
        f.write(f"  hop_length:  {args.hop_length}\n")
        f.write(f"  num_frames:  {args.num_frames}\n\n")

        f.write("Dataset params (fixed/defaults):\n")
        for k in sorted(dataset_params.keys()):
            f.write(f"  {k}: {dataset_params[k]}\n")
        f.write("\n")

        f.write("Model params:\n")
        for k in sorted(model_params.keys()):
            f.write(f"  {k}: {model_params[k]}\n")
        f.write("\n")

        f.write("Optimizer:\n")
        for k in sorted(optimizer_params.keys()):
            f.write(f"  {k}: {optimizer_params[k]}\n")
        f.write("\n")

        f.write("LR Scheduler:\n")
        for k in sorted(scheduler_params.keys()):
            f.write(f"  {k}: {scheduler_params[k]}\n")
        f.write("\n")

        f.write("Data:\n")
        f.write(f"  Train samples: {train_len}\n")
        f.write(f"  Val samples:   {val_len}\n")
        f.write(f"  Num classes (unique MIDI): {num_classes}\n\n")

        f.write("Classes (MIDI -> class index):\n")
        for midi, idx in sorted(midi_to_idx.items(), key=lambda kv: kv[1]):
            f.write(f"  class {idx:2d}: MIDI {midi}\n")
        f.write("\n")

        f.write("Training history:\n")
        for epoch in range(len(history["train_loss"])):
            tl = history["train_loss"][epoch]
            ta = history["train_acc"][epoch]
            vl = history["val_loss"][epoch]
            va = history["val_acc"][epoch]
            lr_str = ""
            if "lr" in history and len(history["lr"]) == len(history["train_loss"]):
                lr_str = f", lr={history['lr'][epoch]:.6g}"
            f.write(
                f"  Epoch {epoch+1:02d}: "
                f"train_loss={tl:.4f}, train_acc={ta:.4f}, "
                f"val_loss={vl:.4f}, val_acc={va:.4f}{lr_str}\n"
            )
        f.write("\n")

        f.write("Best epoch:\n")
        f.write(f"  Epoch:     {best_epoch+1}\n")
        f.write(f"  Val loss:  {best_val_loss:.4f}\n")
        f.write(f"  Val acc:   {best_val_acc:.4f}\n\n")

        f.write("Confusion matrix (rows=true, cols=pred):\n")
        f.write(str(confusion))
        f.write("\n\n")

        f.write("Figures written:\n")
        f.write("  - loss_curves.png\n")
        f.write("  - accuracy_curves.png\n")
        if "lr" in history and history["lr"]:
            f.write("  - lr_curve.png\n")
        f.write("  - confusion_matrix.png\n")
        f.write("  - per_class_accuracy.png\n")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train CNN for note+octave classification from training_mix CSV."
    )
    parser.add_argument(
        "--mix-csv",
        type=str,
        required=True,
        help="Path to training mix CSV (e.g., data/training_mix.csv).",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="note_cnn_best.pt",
        help="Filename for best checkpoint within run artifacts dir.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--num-frames", type=int, default=96)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print("Using device:", device)

    # Create artifacts run directory
    run_dir = create_run_dir(model_type="note")
    print("Artifacts directory:", run_dir)

    # Copy training mix CSV into run dir
    mix_csv = Path(args.mix_csv).expanduser().resolve()
    if not mix_csv.is_file():
        raise FileNotFoundError(f"Training mix CSV not found: {mix_csv}")
    shutil.copy2(mix_csv, run_dir / "training_mix.csv")

    # Load CSV and build mappings
    train_rows, val_rows = load_mix_csv(mix_csv)
    midi_to_idx, idx_to_midi = build_midi_class_mapping(train_rows, val_rows)
    num_classes = len(midi_to_idx)
    print(f"Unique MIDI notes: {num_classes}")
    print("MIDI classes:", sorted(midi_to_idx.keys()))

    # Fixed dataset params (defaults) written into summary/checkpoint for reproducibility
    sample_rate = 48000
    attack_duration = 0.25
    n_fft = 1024
    f_min = 55.0
    f_max = 3000.0

    # Datasets (augmentation currently disabled even for train)
    train_ds = NoteDatasetFromMix(
        train_rows,
        midi_to_idx,
        sample_rate=sample_rate,
        attack_duration=attack_duration,
        n_fft=n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        f_min=f_min,
        f_max=f_max,
        num_frames=args.num_frames,
        augment=False,
    )
    val_ds = NoteDatasetFromMix(
        val_rows,
        midi_to_idx,
        sample_rate=sample_rate,
        attack_duration=attack_duration,
        n_fft=n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        f_min=f_min,
        f_max=f_max,
        num_frames=args.num_frames,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # Model and training setup
    dropout_p = 0.3
    label_smoothing = 0.05

    model = NoteCNN(num_classes=num_classes, n_mels=args.n_mels, dropout_p=dropout_p).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # LR scheduler on validation accuracy
    scheduler_mode = "max"
    scheduler_factor = 0.1
    scheduler_patience = 5

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_mode,
        factor=scheduler_factor,
        patience=scheduler_patience,
        verbose=True,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = -1
    checkpoint_path = run_dir / args.checkpoint_name

    # Training loop
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1:02d}/{args.epochs:02d} | "
            f"Train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"Val loss {val_loss:.4f}, acc {val_acc:.4f}"
        )

        # Step LR scheduler on validation accuracy
        scheduler.step(val_acc)

        # Record LR (after scheduler step)
        try:
            history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        except Exception:
            history["lr"].append(float("nan"))

        # Track best by validation accuracy (tie-breaker: lower loss)
        is_better = (val_acc > best_val_acc) or (
            val_acc == best_val_acc and val_loss < best_val_loss
        )
        if is_better:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint = {
                "model_state": model.state_dict(),
                "num_classes": num_classes,
                "midi_to_idx": midi_to_idx,
                "idx_to_midi": idx_to_midi,
                "sample_rate": train_ds.sample_rate,
                "attack_duration": train_ds.attack_duration,
                "attack_samples": train_ds.attack_samples,
                "mel_params": {
                    "n_fft": train_ds.n_fft,
                    "hop_length": train_ds.hop_length,
                    "n_mels": train_ds.n_mels,
                    "f_min": train_ds.f_min,
                    "f_max": train_ds.f_max,
                    "num_frames": train_ds.num_frames,
                },
                "model_params": {
                    "model_class": "NoteCNN",
                    "dropout_p": dropout_p,
                },
                "training_params": {
                    "criterion": "CrossEntropyLoss",
                    "label_smoothing": label_smoothing,
                    "optimizer": "Adam",
                    "lr_initial": args.lr,
                    "scheduler": "ReduceLROnPlateau",
                    "scheduler_mode": scheduler_mode,
                    "scheduler_factor": scheduler_factor,
                    "scheduler_patience": scheduler_patience,
                },
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved new best checkpoint to {checkpoint_path}")

    # Reload best model for final evaluation and artifact generation
    if best_epoch >= 0 and checkpoint_path.is_file():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
    else:
        print("Warning: no best checkpoint found; using final epoch weights.")

    # Compute confusion matrix on validation set
    confusion = compute_confusion_matrix(
        model, val_loader, device, num_classes=num_classes
    )

    # Generate plots (paper-relevant, low overhead)
    plot_training_curves(history, run_dir)
    plot_lr_curve(history, run_dir)
    plot_confusion_matrix(confusion, run_dir)
    plot_per_class_accuracy(confusion, idx_to_midi, run_dir)

    # Export ONNX for Netron
    export_onnx_model(
        model,
        run_dir,
        num_mels=args.n_mels,
        num_frames=train_ds.num_frames,
    )

    dataset_params = {
        "sample_rate": float(train_ds.sample_rate),
        "attack_duration_sec": float(train_ds.attack_duration),
        "attack_samples": float(train_ds.attack_samples),
        "n_fft": float(train_ds.n_fft),
        "hop_length": float(train_ds.hop_length),
        "n_mels": float(train_ds.n_mels),
        "f_min": float(train_ds.f_min),
        "f_max": float(train_ds.f_max),
        "num_frames": float(train_ds.num_frames),
        "augment": float(0.0),
    }
    model_params = {
        "model_class": "NoteCNN",
        "dropout_p": float(dropout_p),
        "num_classes": float(num_classes),
    }
    optimizer_params = {
        "optimizer": "Adam",
        "lr_initial": float(args.lr),
    }
    scheduler_params = {
        "scheduler": "ReduceLROnPlateau",
        "mode": scheduler_mode,
        "factor": float(scheduler_factor),
        "patience": float(scheduler_patience),
        "step_metric": "val_acc",
    }

    # Write summary
    write_summary(
        run_dir=run_dir,
        args=args,
        device=device,
        num_classes=num_classes,
        midi_to_idx=midi_to_idx,
        idx_to_midi=idx_to_midi,
        train_len=len(train_ds),
        val_len=len(val_ds),
        history=history,
        best_epoch=best_epoch,
        best_val_acc=best_val_acc,
        best_val_loss=best_val_loss,
        confusion=confusion,
        dataset_params=dataset_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        scheduler_params=scheduler_params,
    )

    print("Artifacts written to:", run_dir)


if __name__ == "__main__":
    main()
