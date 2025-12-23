#!/usr/bin/env python3
"""
train_disambiguator.py - Train string disambiguation model.

Trains a small CNN to disambiguate between two adjacent strings
for notes that can be played on either string.

Usage:
    # Single pair training
    python train_disambiguator.py \\
        --mix-csv recipes/disambig/disambig_a_d_mix.csv \\
        --pair A_D

    # Batch training (all pairs)
    python train_disambiguator.py \\
        --batch \\
        --mix-dir recipes/disambig/
"""

import argparse
import csv
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.disambiguator_model import DisambiguatorCNN

STRING_PAIRS = {
    'E_A': (6, 5, 'Low E', 'A'),
    'A_D': (5, 4, 'A', 'D'),
    'D_G': (4, 3, 'D', 'G'),
    'G_B': (3, 2, 'G', 'B'),
    'B_e': (2, 1, 'B', 'High e'),
}


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DisambiguatorDataset(Dataset):
    def __init__(
        self,
        rows: List[dict],
        sample_rate: int = 16000,
        attack_duration: float = 0.25,
        n_fft: int = 1024,
        hop_length: int = 128,
        n_mels: int = 64,
        f_min: float = 55.0,
        f_max: float = 3000.0,
        num_frames: int = 32,
        augment: bool = False,
    ):
        self.rows = rows
        self.sample_rate = sample_rate
        self.attack_duration = attack_duration
        self.attack_samples = int(sample_rate * attack_duration)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.num_frames = num_frames
        self.augment = augment
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=f_min, f_max=f_max,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
    
    def __len__(self):
        return len(self.rows)
    
    def _load_audio(self, path: str) -> np.ndarray:
        import soundfile as sf
        audio, sr = sf.read(path, dtype='float32')
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        if len(audio) > self.attack_samples:
            audio = audio[:self.attack_samples]
        elif len(audio) < self.attack_samples:
            audio = np.pad(audio, (0, self.attack_samples - len(audio)))
        return audio.astype(np.float32)
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        if not self.augment:
            return audio
        if random.random() < 0.5:
            gain_db = (random.random() - 0.5) * 6
            audio = audio * (10 ** (gain_db / 20))
        if random.random() < 0.3:
            shift = random.randint(-len(audio) // 10, len(audio) // 10)
            audio = np.roll(audio, shift)
        if random.random() < 0.3:
            audio = audio + np.random.randn(len(audio)).astype(np.float32) * 0.005
        return audio
    
    def _audio_to_mel(self, audio: np.ndarray) -> torch.Tensor:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95
        audio = audio - np.mean(audio)
        wav_t = torch.from_numpy(audio).float().unsqueeze(0)
        mel = self.mel_transform(wav_t)
        mel_db = self.db_transform(mel)
        mean = mel_db.mean(dim=2, keepdim=True)
        std = mel_db.std(dim=2, keepdim=True) + 1e-6
        mel_db = (mel_db - mean) / std
        if mel_db.size(2) < self.num_frames:
            mel_db = torch.nn.functional.pad(mel_db, (0, self.num_frames - mel_db.size(2)))
        elif mel_db.size(2) > self.num_frames:
            mel_db = mel_db[:, :, :self.num_frames]
        return mel_db
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        audio = self._load_audio(row['abs_path'])
        if self.augment:
            audio = self._augment_audio(audio)
        mel = self._audio_to_mel(audio)
        return {
            'mel': mel,
            'target': torch.tensor(int(row['target']), dtype=torch.long),
            'fret_a': torch.tensor(int(row['fret_a']), dtype=torch.long),
            'fret_b': torch.tensor(int(row['fret_b']), dtype=torch.long),
            'midi': int(row['midi']),
        }


def load_mix_csv(csv_path: Path) -> Tuple[List[dict], List[dict]]:
    train_rows, val_rows = [], []
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            (val_rows if row.get('split') == 'val' else train_rows).append(row)
    return train_rows, val_rows


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        mel = batch['mel'].to(device)
        target = batch['target'].to(device)
        fret_a, fret_b = batch['fret_a'].to(device), batch['fret_b'].to(device)
        optimizer.zero_grad()
        logits = model(mel, fret_a, fret_b)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * mel.size(0)
        correct += (logits.argmax(1) == target).sum().item()
        total += mel.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    midi_correct, midi_total = {}, {}
    with torch.no_grad():
        for batch in loader:
            mel = batch['mel'].to(device)
            target = batch['target'].to(device)
            fret_a, fret_b = batch['fret_a'].to(device), batch['fret_b'].to(device)
            logits = model(mel, fret_a, fret_b)
            loss = criterion(logits, target)
            total_loss += loss.item() * mel.size(0)
            pred = logits.argmax(1)
            correct += (pred == target).sum().item()
            total += mel.size(0)
            for i, midi in enumerate(batch['midi']):
                midi = int(midi)
                midi_correct.setdefault(midi, 0)
                midi_total.setdefault(midi, 0)
                midi_correct[midi] += (pred[i] == target[i]).item()
                midi_total[midi] += 1
    midi_acc = {m: midi_correct[m] / midi_total[m] for m in midi_total}
    return total_loss / total, correct / total, midi_acc


def create_run_dir(model_type: str = "disambig", pair: str = None) -> Path:
    artifacts_root = REPO_ROOT / "artifacts" / "models"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{model_type}_{pair}_{ts}" if pair else f"{model_type}_{ts}"
    run_dir = artifacts_root / base_name
    suffix = 1
    while run_dir.exists():
        run_dir = artifacts_root / f"{base_name}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def plot_training_curves(history: Dict, run_dir: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
    plt.savefig(run_dir / "loss_curves.png"); plt.close()
    
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["val_acc"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True)
    plt.savefig(run_dir / "accuracy_curves.png"); plt.close()
    
    if history.get("lr"):
        plt.figure()
        plt.plot(epochs, history["lr"])
        plt.xlabel("Epoch"); plt.ylabel("LR"); plt.grid(True)
        plt.savefig(run_dir / "lr_curve.png"); plt.close()


def plot_per_midi_accuracy(midi_acc: Dict[int, float], run_dir: Path, pair: str) -> None:
    notes = sorted(midi_acc.keys())
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    labels = [f"{names[m % 12]}{m // 12 - 1}" for m in notes]
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(notes)), [midi_acc[m] for m in notes], tick_label=labels)
    plt.ylim(0, 1.05); plt.xlabel("MIDI"); plt.ylabel("Accuracy")
    plt.title(f"Per-MIDI Accuracy ({pair})"); plt.grid(True, axis='y')
    plt.savefig(run_dir / "per_midi_accuracy.png"); plt.close()


def write_summary(run_dir, args, pair, device, train_len, val_len, history,
                  best_epoch, best_val_acc, best_val_loss, midi_acc) -> None:
    with open(run_dir / "summary.txt", "w") as f:
        f.write(f"{'='*60}\nDISAMBIGUATOR TRAINING SUMMARY - {pair}\n{'='*60}\n\n")
        f.write(f"Run directory: {run_dir}\nDevice: {device}\n\n")
        f.write(f"Dataset: {train_len} train, {val_len} val\n")
        f.write(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}, Patience: {args.patience}\n\n")
        f.write(f"Best epoch: {best_epoch + 1}, Val acc: {best_val_acc:.4f}, Val loss: {best_val_loss:.4f}\n\n")
        f.write("Per-MIDI accuracy:\n")
        names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for midi, acc in sorted(midi_acc.items()):
            f.write(f"  MIDI {midi} ({names[midi % 12]}{midi // 12 - 1}): {acc:.4f}\n")


def train_single_pair(mix_csv: Path, pair: str, args, run_dir: Path = None) -> Dict:
    device = get_device()
    print(f"Using device: {device}")
    
    if run_dir is None:
        run_dir = create_run_dir("disambig", pair)
    print(f"Artifacts directory: {run_dir}")
    
    shutil.copy2(mix_csv, run_dir / "training_mix.csv")
    train_rows, val_rows = load_mix_csv(mix_csv)
    print(f"Train: {len(train_rows)}, Val: {len(val_rows)}")
    
    train_ds = DisambiguatorDataset(train_rows, n_mels=args.n_mels, hop_length=args.hop_length,
                                     num_frames=args.num_frames, augment=args.augment)
    val_ds = DisambiguatorDataset(val_rows, n_mels=args.n_mels, hop_length=args.hop_length,
                                   num_frames=args.num_frames, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = DisambiguatorCNN(n_mels=args.n_mels, num_frets=13, dropout_p=args.dropout)
    dummy = torch.zeros(1, 1, args.n_mels, args.num_frames)
    dummy_fret = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        model(dummy, dummy_fret, dummy_fret)
    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                      patience=args.lr_patience, verbose=True)
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc, best_val_loss, best_epoch = 0.0, float("inf"), -1
    patience_counter = 0
    best_midi_acc = {}
    ckpt_path = run_dir / args.checkpoint_name
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, midi_acc = evaluate(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        
        midi_str = " | ".join([f"M{m}:{a:.2f}" for m, a in sorted(midi_acc.items())])
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | "
              f"Train {train_loss:.4f}/{train_acc:.4f} | Val {val_loss:.4f}/{val_acc:.4f}")
        print(f"         {midi_str}")
        
        scheduler.step(val_acc)
        
        is_better = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if is_better:
            best_val_acc, best_val_loss, best_epoch = val_acc, val_loss, epoch
            best_midi_acc = midi_acc.copy()
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(), "pair": pair,
                "val_acc": val_acc, "val_loss": val_loss, "epoch": epoch + 1,
                "midi_acc": midi_acc, "sample_rate": train_ds.sample_rate,
                "attack_samples": train_ds.attack_samples,
                "mel_params": {"n_fft": train_ds.n_fft, "hop_length": train_ds.hop_length,
                               "n_mels": train_ds.n_mels, "f_min": train_ds.f_min,
                               "f_max": train_ds.f_max, "num_frames": train_ds.num_frames},
                "model_params": {"model_class": "DisambiguatorCNN", "num_frets": 13,
                                 "embed_dim": 8, "dropout_p": args.dropout},
            }, ckpt_path)
            print(f"         *** New best: {val_acc:.4f} ***")
        else:
            patience_counter += 1
        
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1} (patience={args.patience})")
            break
    
    plot_training_curves(history, run_dir)
    plot_per_midi_accuracy(best_midi_acc, run_dir, pair)
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    write_summary(run_dir, args, pair, device, len(train_ds), len(val_ds),
                  history, best_epoch, best_val_acc, best_val_loss, best_midi_acc)
    
    print(f"\nBest val accuracy: {best_val_acc:.4f} at epoch {best_epoch + 1}")
    print(f"Artifacts: {run_dir}")
    return {"pair": pair, "best_val_acc": best_val_acc, "best_epoch": best_epoch + 1, "run_dir": run_dir}


def train_batch(args) -> None:
    mix_dir = Path(args.mix_dir)
    if not mix_dir.exists():
        raise FileNotFoundError(f"Mix directory not found: {mix_dir}")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = REPO_ROOT / "artifacts" / "models" / f"disambig_batch_{ts}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"BATCH TRAINING: All disambiguator pairs")
    print(f"Output: {batch_dir}")
    print("=" * 60)
    
    results = {}
    for pair_name in STRING_PAIRS.keys():
        mix_csv = mix_dir / f"disambig_{pair_name.lower()}_mix.csv"
        if not mix_csv.exists():
            print(f"\nSkipping {pair_name}: {mix_csv} not found")
            results[pair_name] = {"status": "skipped"}
            continue
        
        print(f"\n{'='*60}\nTraining {pair_name}...\n{'='*60}")
        pair_dir = batch_dir / pair_name
        pair_dir.mkdir(exist_ok=True)
        
        try:
            result = train_single_pair(mix_csv, pair_name, args, run_dir=pair_dir)
            results[pair_name] = {"status": "success", **result}
        except Exception as e:
            print(f"ERROR: {e}")
            results[pair_name] = {"status": "error", "error": str(e)}
    
    print(f"\n{'='*60}\nBATCH SUMMARY\n{'='*60}")
    for pair, r in results.items():
        if r["status"] == "success":
            print(f"  {pair}: {r['best_val_acc']:.4f} (epoch {r['best_epoch']})")
        else:
            print(f"  {pair}: {r['status']}")
    
    with open(batch_dir / "batch_summary.json", "w") as f:
        json.dump({"timestamp": ts, "results": {k: {kk: str(vv) if isinstance(vv, Path) else vv 
                  for kk, vv in v.items()} for k, v in results.items()}}, f, indent=2)
    print(f"\nResults: {batch_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train string disambiguator")
    parser.add_argument("--mix-csv", type=str, help="Mix CSV (single mode)")
    parser.add_argument("--pair", type=str, choices=list(STRING_PAIRS.keys()))
    parser.add_argument("--batch", action="store_true", help="Train all pairs")
    parser.add_argument("--mix-dir", type=str, help="Mix directory (batch mode)")
    parser.add_argument("--checkpoint-name", type=str, default="disambig_best.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.batch:
        if not args.mix_dir:
            parser.error("--mix-dir required with --batch")
        train_batch(args)
    else:
        if not args.mix_csv or not args.pair:
            parser.error("--mix-csv and --pair required")
        mix_csv = Path(args.mix_csv).expanduser().resolve()
        if not mix_csv.is_file():
            raise FileNotFoundError(f"Mix CSV not found: {mix_csv}")
        train_single_pair(mix_csv, args.pair, args)


if __name__ == "__main__":
    main()
