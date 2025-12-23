#!/usr/bin/env python3

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import shutil
import platform

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from string_model import StringCNN

"""
python src/train_string_model.py \
  --mix-csv recipes/training_mix.csv \
  --batch-size 64 \
  --epochs 30 \
  --lr 1e-3 \
  --num-workers 0 \
  --seed 42 \
  --checkpoint-name string_cnn_best.pt \
  --n-mels 64 \
  --hop-length 128 \
  --num-frames 96
"""
# -----------------------------
# Ambiguous position detection
# -----------------------------

# Positions where open strings share pitch with fretted notes on adjacent strings
AMBIGUOUS_POSITIONS = {
    (1, 0),  # Open E4
    (2, 0),  # Open B3
    (3, 0),  # Open G3
    (4, 0),  # Open D3
    (5, 0),  # Open A2
    (2, 5),  # B string fret 5 = E4
    (3, 4),  # G string fret 4 = B3
    (4, 5),  # D string fret 5 = G3
    (5, 5),  # A string fret 5 = D3
    (6, 5),  # E string fret 5 = A2
}


def count_ambiguous_samples(rows: List[dict]) -> int:
    """Count how many samples are from ambiguous positions."""
    count = 0
    for row in rows:
        try:
            string = int(row.get("string", 0))
            fret = int(row.get("fret", 0))
            if (string, fret) in AMBIGUOUS_POSITIONS:
                count += 1
        except (ValueError, TypeError):
            pass
    return count


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
# CSV loading
# -----------------------------

def load_mix_csv(csv_path: Path) -> Tuple[List[dict], List[dict]]:
    """
    Load training mix CSV and split into train/val row lists.

    Assumes CSV has at least:
      - abs_path
      - string
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

            if split not in ("train", "val"):
                continue
            if not abs_path:
                continue

            # keep as strings for now; dataset will normalize
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


# -----------------------------
# Dataset
# -----------------------------

def _detect_string_base(rows: List[dict]) -> int:
    """
    Detect whether the dataset uses 0-based (0..5) or 1-based (1..6) string labels.

    Returns:
      0 if labels are 0..5
      1 if labels are 1..6

    Raises if values do not match either scheme.
    """
    vals: List[int] = []
    for r in rows:
        v = r.get("string", None)
        if v is None:
            continue
        s = str(v).strip().lower()
        if s.startswith("s") and s[1:].isdigit():
            vals.append(int(s[1:]))
        elif s.isdigit():
            vals.append(int(s))
        else:
            raise ValueError(f"unrecognized string label: {v}")

    if not vals:
        raise ValueError("no string labels found to detect base")

    mn, mx = min(vals), max(vals)
    if mn == 0 and mx == 5:
        return 0
    if mn == 1 and mx == 6:
        return 1

    raise ValueError(
        f"inconsistent string labels (min={mn}, max={mx}); expected 0..5 or 1..6"
    )


def _parse_string_label(v: str, base: int) -> int:
    """
    Normalize 'string' field to 1..6 using detected base.

    Accepts:
      - "1".."6" (base=1)
      - "0".."5" (base=0; shifts to 1..6)
      - "s1".."s6" or "s0".."s5" depending on base
    """
    if v is None:
        raise ValueError("missing string label")

    s = str(v).strip().lower()
    if s.startswith("s") and s[1:].isdigit():
        n = int(s[1:])
    elif s.isdigit():
        n = int(s)
    else:
        raise ValueError(f"unrecognized string label: {v}")

    if base == 0:
        n = n + 1  # 0..5 -> 1..6

    if not (1 <= n <= 6):
        raise ValueError(f"string out of range after normalize: {n}")

    return n


class StringDatasetFromMix(Dataset):
    """
    Dataset for string classification from training_mix CSV.

    Returns:
      spec: Float tensor [1, n_mels, num_frames]
      label: int in [0..5] for CrossEntropy
    """

    def __init__(
        self,
        rows: List[dict],
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
        self.base = _detect_string_base(rows)

        self.sample_rate = sample_rate
        self.attack_duration = float(attack_duration)
        self.attack_samples = int(round(sample_rate * attack_duration))
        self.num_frames = int(num_frames)
        self.augment = bool(augment)  # currently unused

        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.f_min = float(f_min)
        self.f_max = float(f_max)

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
        path = Path(path_str)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        wav, sr = torchaudio.load(path)  # [C, T], float32
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)

        if wav.size(1) < self.attack_samples:
            pad = self.attack_samples - wav.size(1)
            wav = torch.nn.functional.pad(wav, (0, pad))
        elif wav.size(1) > self.attack_samples:
            wav = wav[:, : self.attack_samples]

        return wav

    def _waveform_to_logmel(self, wav: torch.Tensor) -> torch.Tensor:
        device = self.transform_device or wav.device
        wav = wav.to(device)

        spec = self.mel_spec(wav)          # [1, n_mels, T_spec]
        spec_db = self.db_transform(spec)  # [1, n_mels, T_spec]

        mean = spec_db.mean(dim=2, keepdim=True)
        std = spec_db.std(dim=2, keepdim=True) + 1e-6
        spec_db = (spec_db - mean) / std

        _, _, T = spec_db.shape
        if T < self.num_frames:
            pad = self.num_frames - T
            spec_db = torch.nn.functional.pad(spec_db, (0, pad))
        elif T > self.num_frames:
            spec_db = spec_db[:, :, : self.num_frames]

        return spec_db

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.rows[idx]
        abs_path = row["abs_path"]

        try:
            s = _parse_string_label(row.get("string", ""), base=self.base)
        except Exception as e:
            raise RuntimeError(f"Bad string label for row index {idx}: {e}")

        label = s - 1  # 0..5

        wav = self._load_audio(abs_path)
        spec = self._waveform_to_logmel(wav)

        return spec.cpu(), int(label)


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
        specs = specs.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += bs

    return total_loss / total_samples, total_correct / total_samples


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

            bs = labels.size(0)
            total_loss += loss.item() * bs
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += bs

    return total_loss / total_samples, total_correct / total_samples


def compute_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> np.ndarray:
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
    return Path(__file__).resolve().parent.parent


def create_run_dir(model_type: str = "string") -> Path:
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


def plot_training_curves(history: Dict[str, List[float]], run_dir: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

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

    if "lr" in history and len(history["lr"]) == len(history["train_loss"]):
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


def plot_confusion_matrix(cm: np.ndarray, run_dir: Path, title: str = "Confusion Matrix") -> None:
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png")
    plt.close()


def plot_per_class_accuracy(cm: np.ndarray, run_dir: Path, title: str = "Per-class Accuracy") -> None:
    # diag / row-sum
    row_sum = cm.sum(axis=1).astype(np.float64)
    diag = np.diag(cm).astype(np.float64)
    acc = np.zeros_like(diag)
    mask = row_sum > 0
    acc[mask] = diag[mask] / row_sum[mask]

    plt.figure()
    plt.plot(list(range(len(acc))), acc, marker="o")
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.xlabel("Class (string_idx: 0..5)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(run_dir / "per_class_accuracy.png")
    plt.close()


def export_onnx_model(model: nn.Module, run_dir: Path, num_mels: int, num_frames: int) -> None:
    model.cpu().eval()
    dummy_input = torch.randn(1, 1, num_mels, num_frames, dtype=torch.float32)
    onnx_path = run_dir / "string_cnn.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["logmel"],
        output_names=["logits"],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes={"logmel": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )


def _safe_get_version(mod) -> str:
    try:
        return str(mod.__version__)
    except Exception:
        return "unknown"


def write_summary(
    run_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    train_len: int,
    val_len: int,
    history: Dict[str, List[float]],
    best_epoch: int,
    best_val_acc: float,
    best_val_loss: float,
    confusion: np.ndarray,
    model: StringCNN,
    ds: StringDatasetFromMix,
    cmd_argv: List[str],
    figures: List[str],
    label_counts_train: np.ndarray,
    label_counts_val: np.ndarray,
    train_ambiguous: int = 0,
    val_ambiguous: int = 0,
) -> None:
    summary_path = run_dir / "summary.txt"
    with summary_path.open("w") as f:
        f.write("String CNN Training Summary\n")
        f.write("===========================\n\n")
        f.write(f"Run directory: {run_dir}\n")
        f.write(f"Mix CSV:       {Path(args.mix_csv).resolve()}\n")
        f.write(f"Device:        {device.type}\n")
        f.write(f"Seed:          {args.seed}\n")
        f.write(f"Timestamp:     {datetime.now().isoformat(timespec='seconds')}\n\n")

        f.write("Repro / Environment:\n")
        f.write(f"  Python:      {sys.version.split()[0]}\n")
        f.write(f"  Platform:    {platform.platform()}\n")
        f.write(f"  Torch:       {_safe_get_version(torch)}\n")
        f.write(f"  Torchaudio:  {_safe_get_version(torchaudio)}\n\n")

        f.write("Command:\n")
        f.write("  " + " ".join(cmd_argv) + "\n\n")

        f.write("Hyperparameters (CLI):\n")
        f.write(f"  Batch size:  {args.batch_size}\n")
        f.write(f"  Epochs:      {args.epochs}\n")
        f.write(f"  Learning rate (initial): {args.lr}\n")
        f.write(f"  Num workers: {args.num_workers}\n")
        f.write(f"  n_mels:      {args.n_mels}\n")
        f.write(f"  hop_length:  {args.hop_length}\n")
        f.write(f"  num_frames:  {args.num_frames}\n")
        f.write(f"  label_smoothing: {args.label_smoothing}\n\n")

        f.write("Dataset params (fixed/defaults):\n")
        f.write(f"  sample_rate: {ds.sample_rate}\n")
        f.write(f"  attack_duration_sec: {ds.attack_duration}\n")
        f.write(f"  attack_samples: {ds.attack_samples}\n")
        f.write(f"  n_fft: {ds.n_fft}\n")
        f.write(f"  hop_length: {ds.hop_length}\n")
        f.write(f"  n_mels: {ds.n_mels}\n")
        f.write(f"  f_min: {ds.f_min}\n")
        f.write(f"  f_max: {ds.f_max}\n")
        f.write(f"  num_frames: {ds.num_frames}\n")
        f.write(f"  augment: {int(ds.augment)}\n")
        f.write(f"  detected_label_base: {ds.base}  # 0=>0..5, 1=>1..6 in CSV\n\n")

        f.write("Model params:\n")
        f.write(f"  model_class: {model.__class__.__name__}\n")
        f.write(f"  num_strings: {model.num_strings}\n")
        f.write(f"  dropout_p: {model.dropout_p}\n\n")

        f.write("Optimizer:\n")
        f.write("  optimizer: Adam\n")
        f.write(f"  lr_initial: {args.lr}\n\n")

        f.write("LR Scheduler:\n")
        f.write("  scheduler: ReduceLROnPlateau\n")
        f.write("  mode: max\n")
        f.write(f"  factor: {args.lr_factor}\n")
        f.write(f"  patience: {args.lr_patience}\n")
        f.write("  step_metric: val_acc\n\n")

        f.write("Data:\n")
        f.write(f"  Train samples: {train_len}\n")
        f.write(f"  Val samples:   {val_len}\n")
        f.write("  Num classes (strings): 6\n")
        f.write(f"  Train ambiguous samples: {train_ambiguous} ({100*train_ambiguous/train_len:.1f}%)\n")
        f.write(f"  Val ambiguous samples:   {val_ambiguous} ({100*val_ambiguous/val_len:.1f}%)\n\n")

        f.write("Label counts (after normalization; labels are 0..5):\n")
        f.write(f"  Train: {label_counts_train.tolist()}\n")
        f.write(f"  Val:   {label_counts_val.tolist()}\n\n")

        f.write("Training history:\n")
        for epoch in range(len(history["train_loss"])):
            tl = history["train_loss"][epoch]
            ta = history["train_acc"][epoch]
            vl = history["val_loss"][epoch]
            va = history["val_acc"][epoch]
            lr = history["lr"][epoch] if "lr" in history and epoch < len(history["lr"]) else float("nan")
            f.write(
                f"  Epoch {epoch+1:02d}: "
                f"train_loss={tl:.4f}, train_acc={ta:.4f}, "
                f"val_loss={vl:.4f}, val_acc={va:.4f}, lr={lr:g}\n"
            )
        f.write("\n")

        f.write("Best epoch:\n")
        f.write(f"  Epoch:     {best_epoch+1}\n")
        f.write(f"  Val loss:  {best_val_loss:.4f}\n")
        f.write(f"  Val acc:   {best_val_acc:.4f}\n\n")

        f.write("Confusion matrix (rows=true, cols=pred):\n")
        f.write(str(confusion))
        f.write("\n\n")

        if figures:
            f.write("Figures written:\n")
            for fig in figures:
                f.write(f"  - {fig}\n")
            f.write("\n")


def _count_labels(ds: StringDatasetFromMix) -> np.ndarray:
    counts = np.zeros(6, dtype=np.int64)
    for i in range(len(ds)):
        _, y = ds[i]
        counts[int(y)] += 1
    return counts


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train CNN for string classification from training_mix CSV."
    )
    parser.add_argument("--mix-csv", type=str, required=True)
    parser.add_argument("--checkpoint-name", type=str, default="string_cnn_best.pt")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=0,
                       help="Early stopping patience (0 = disabled)")

    parser.add_argument("--n-mels", type=int, default=48)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=32)

    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)

    parser.add_argument("--lr-factor", type=float, default=0.1)
    parser.add_argument("--lr-patience", type=int, default=5)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print("Using device:", device)

    run_dir = create_run_dir(model_type="string")
    print("Artifacts directory:", run_dir)

    mix_csv = Path(args.mix_csv).expanduser().resolve()
    if not mix_csv.is_file():
        raise FileNotFoundError(f"Training mix CSV not found: {mix_csv}")
    shutil.copy2(mix_csv, run_dir / "training_mix.csv")

    train_rows, val_rows = load_mix_csv(mix_csv)

    # Log ambiguous sample counts
    train_ambiguous = count_ambiguous_samples(train_rows)
    val_ambiguous = count_ambiguous_samples(val_rows)
    print(f"Ambiguous samples: {train_ambiguous} train, {val_ambiguous} val "
          f"({100*train_ambiguous/len(train_rows):.1f}% / {100*val_ambiguous/len(val_rows):.1f}%)")

    train_ds = StringDatasetFromMix(
        train_rows,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        num_frames=args.num_frames,
        augment=False,
    )
    val_ds = StringDatasetFromMix(
        val_rows,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        num_frames=args.num_frames,
        augment=False,
    )

    # Sanity: label distributions (post-normalization)
    label_counts_train = _count_labels(train_ds)
    label_counts_val = _count_labels(val_ds)
    print("Train label counts (0..5):", label_counts_train.tolist())
    print("Val label counts (0..5):  ", label_counts_val.tolist())

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

    model = StringCNN(num_strings=6, n_mels=args.n_mels, dropout_p=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(args.lr_factor),
        patience=int(args.lr_patience),
        verbose=True,
    )

    history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    checkpoint_path = run_dir / args.checkpoint_name

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        print(
            f"Epoch {epoch+1:02d}/{args.epochs:02d} | "
            f"Train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"Val loss {val_loss:.4f}, acc {val_acc:.4f}"
        )

        scheduler.step(val_acc)

        is_better = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if is_better:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            checkpoint = {
                "model_state": model.state_dict(),
                "num_strings": 6,
                "idx_to_string": {i: i + 1 for i in range(6)},  # 0->1, ..., 5->6
                "sample_rate": train_ds.sample_rate,
                "attack_samples": train_ds.attack_samples,
                "mel_params": {
                    "n_fft": train_ds.n_fft,
                    "hop_length": train_ds.hop_length,
                    "n_mels": train_ds.n_mels,
                    "f_min": train_ds.f_min,
                    "f_max": train_ds.f_max,
                    "num_frames": train_ds.num_frames,
                    "attack_duration": train_ds.attack_duration,
                    "sample_rate": train_ds.sample_rate,
                },
                "model_params": {
                    "dropout_p": model.dropout_p,
                    "model_class": model.__class__.__name__,
                },
                "optimizer_params": {
                    "optimizer": "Adam",
                    "lr_initial": args.lr,
                },
                "scheduler_params": {
                    "scheduler": "ReduceLROnPlateau",
                    "mode": "max",
                    "factor": args.lr_factor,
                    "patience": args.lr_patience,
                    "step_metric": "val_acc",
                },
                "training_params": {
                    "early_stopping_patience": args.patience,
                },
                "label_schema": {
                    "csv_base_detected": int(train_ds.base),  # 0 => CSV 0..5, 1 => CSV 1..6
                    "train_label_range": "0..5",
                    "val_label_range": "0..5",
                },
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved new best checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1} (patience={args.patience})")
            break

    if best_epoch >= 0 and checkpoint_path.is_file():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
    else:
        print("Warning: no best checkpoint found; using final epoch weights.")

    confusion = compute_confusion_matrix(model, val_loader, device, num_classes=6)

    figures: List[str] = []
    plot_training_curves(history, run_dir)
    figures.extend(["loss_curves.png", "accuracy_curves.png", "lr_curve.png"])

    plot_confusion_matrix(confusion, run_dir, title="String Confusion Matrix")
    figures.append("confusion_matrix.png")

    plot_per_class_accuracy(confusion, run_dir, title="Per-string Accuracy")
    figures.append("per_class_accuracy.png")

    export_onnx_model(model, run_dir, num_mels=args.n_mels, num_frames=train_ds.num_frames)
    figures.append("string_cnn.onnx")

    write_summary(
        run_dir=run_dir,
        args=args,
        device=device,
        train_len=len(train_ds),
        val_len=len(val_ds),
        history=history,
        best_epoch=best_epoch,
        best_val_acc=best_val_acc,
        best_val_loss=best_val_loss,
        confusion=confusion,
        model=model,
        ds=train_ds,
        cmd_argv=[str(Path(__file__).resolve())] + sys.argv[1:],
        figures=figures,
        label_counts_train=label_counts_train,
        label_counts_val=label_counts_val,
        train_ambiguous=train_ambiguous,
        val_ambiguous=val_ambiguous,
    )

    print("Artifacts written to:", run_dir)


if __name__ == "__main__":
    main()