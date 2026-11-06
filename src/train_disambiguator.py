#!/usr/bin/env python3
"""
train_disambiguator.py - Train string disambiguation models.

Supports two modes:
- Triplet (3-way): TripletDisambiguatorCNN, 3-class output (recommended)
- Pair (2-way): DisambiguatorCNN, 2-class output (legacy)

Usage:
    # Train all triplets (recommended)
    python train_disambiguator.py --batch --type triplet --mix-dir recipes/disambig/
    
    # Train all pairs (legacy)
    python train_disambiguator.py --batch --type pair --mix-dir recipes/disambig/
"""

import argparse
import csv
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio

import matplotlib
matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from disambiguator_model import TripletDisambiguatorCNN, DisambiguatorCNN

STRING_TRIPLETS = {'E_A_D': (6, 5, 4), 'A_D_G': (5, 4, 3), 'D_G_B': (4, 3, 2), 'G_B_e': (3, 2, 1)}
STRING_PAIRS = {'E_A': (6, 5), 'A_D': (5, 4), 'D_G': (4, 3), 'G_B': (3, 2), 'B_e': (2, 1)}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DisambigDataset(Dataset):
    """Dataset for both triplet and pair disambiguation.
    
    Uses same audio parameters as note model for consistency:
    - Native 48kHz sample rate (no resampling)
    - Same mel spectrogram parameters that work for note detection
    """
    
    def __init__(self, rows, num_classes=3, sample_rate=48000, attack_duration=0.25, 
                 n_fft=1024, hop_length=256, n_mels=48, f_min=55.0, f_max=3000.0, 
                 num_frames=32, augment=False):
        self.rows = rows
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.attack_samples = int(sample_rate * attack_duration)
        self.n_fft, self.hop_length, self.n_mels = n_fft, hop_length, n_mels
        self.f_min, self.f_max, self.num_frames = f_min, f_max, num_frames
        self.augment = augment
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=f_min, f_max=f_max)
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
    
    def __len__(self):
        return len(self.rows)
    
    def _load_audio(self, path):
        import soundfile as sf
        audio, sr = sf.read(path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        if len(audio) > self.attack_samples:
            audio = audio[:self.attack_samples]
        elif len(audio) < self.attack_samples:
            audio = np.pad(audio, (0, self.attack_samples - len(audio)))
        return audio.astype(np.float32)
    
    def _augment_audio(self, audio):
        if not self.augment:
            return audio
        if random.random() < 0.5:
            audio = audio * (10 ** ((random.random() - 0.5) * 6 / 20))
        if random.random() < 0.3:
            audio = np.roll(audio, random.randint(-len(audio)//10, len(audio)//10))
        if random.random() < 0.3:
            audio = audio + np.random.randn(len(audio)).astype(np.float32) * 0.005
        if random.random() < 0.3:
            audio = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=(random.random()-0.5)*0.2)
            audio = audio.astype(np.float32)
        return audio
    
    def _augment_spec(self, spec):
        if not self.augment:
            return spec
        _, n_mels, n_frames = spec.shape
        if random.random() < 0.3:
            w = random.randint(1, min(8, n_mels//4))
            s = random.randint(0, n_mels - w)
            spec[:, s:s+w, :] = 0
        if random.random() < 0.3:
            w = random.randint(1, min(5, n_frames//4))
            s = random.randint(0, n_frames - w)
            spec[:, :, s:s+w] = 0
        return spec
    
    def _audio_to_mel(self, audio):
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95
        audio = audio - np.mean(audio)
        mel = self.mel_transform(torch.from_numpy(audio).float().unsqueeze(0))
        mel_db = self.db_transform(mel)
        mel_db = (mel_db - mel_db.mean(dim=2, keepdim=True)) / (mel_db.std(dim=2, keepdim=True) + 1e-6)
        if mel_db.size(2) < self.num_frames:
            mel_db = torch.nn.functional.pad(mel_db, (0, self.num_frames - mel_db.size(2)))
        elif mel_db.size(2) > self.num_frames:
            mel_db = mel_db[:, :, :self.num_frames]
        return mel_db
    
    def __getitem__(self, idx):
        row = self.rows[idx]
        audio = self._load_audio(row['abs_path'])
        if self.augment:
            audio = self._augment_audio(audio)
        mel = self._audio_to_mel(audio)
        if self.augment:
            mel = self._augment_spec(mel)
        
        result = {
            'mel': mel, 
            'target': torch.tensor(int(row['target']), dtype=torch.long), 
            'midi': int(row['midi'])
        }
        
        # For pair models, also include fret info
        if self.num_classes == 2:
            result['fret_a'] = torch.tensor(int(row['fret_a']), dtype=torch.long)
            result['fret_b'] = torch.tensor(int(row['fret_b']), dtype=torch.long)
        
        return result


def load_mix_csv(csv_path):
    train_rows, val_rows = [], []
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            (val_rows if row.get('split') == 'val' else train_rows).append(row)
    return train_rows, val_rows


def train_one_epoch(model, loader, criterion, optimizer, device, model_type):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        mel, target = batch['mel'].to(device), batch['target'].to(device)
        optimizer.zero_grad()
        
        if model_type == 'triplet':
            logits = model(mel)
        else:  # pair
            fret_a, fret_b = batch['fret_a'].to(device), batch['fret_b'].to(device)
            logits = model(mel, fret_a, fret_b)
        
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * mel.size(0)
        correct += (logits.argmax(1) == target).sum().item()
        total += mel.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device, model_type):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    midi_correct, midi_total = {}, {}
    with torch.no_grad():
        for batch in loader:
            mel, target = batch['mel'].to(device), batch['target'].to(device)
            
            if model_type == 'triplet':
                logits = model(mel)
            else:  # pair
                fret_a, fret_b = batch['fret_a'].to(device), batch['fret_b'].to(device)
                logits = model(mel, fret_a, fret_b)
            
            pred = logits.argmax(1)
            total_loss += criterion(logits, target).item() * mel.size(0)
            correct += (pred == target).sum().item()
            total += mel.size(0)
            for i, midi in enumerate(batch['midi']):
                midi = int(midi)
                midi_correct[midi] = midi_correct.get(midi, 0) + (pred[i] == target[i]).item()
                midi_total[midi] = midi_total.get(midi, 0) + 1
    return total_loss/total, correct/total, {m: midi_correct[m]/midi_total[m] for m in midi_total}


def train_single(mix_csv, name, model_type, args, run_dir=None):
    """Train a single disambiguator (triplet or pair)."""
    device = get_device()
    print(f"Device: {device}, Type: {model_type}")
    
    num_classes = 3 if model_type == 'triplet' else 2
    
    if run_dir is None:
        run_dir = REPO_ROOT / "artifacts" / "models" / f"disambig_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts: {run_dir}")
    
    shutil.copy2(mix_csv, run_dir / "training_mix.csv")
    train_rows, val_rows = load_mix_csv(mix_csv)
    print(f"Train: {len(train_rows)}, Val: {len(val_rows)}")
    
    train_ds = DisambigDataset(train_rows, num_classes=num_classes, n_mels=args.n_mels, 
                                hop_length=args.hop_length, num_frames=args.num_frames, augment=args.augment)
    val_ds = DisambigDataset(val_rows, num_classes=num_classes, n_mels=args.n_mels,
                              hop_length=args.hop_length, num_frames=args.num_frames, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create model based on type
    if model_type == 'triplet':
        model = TripletDisambiguatorCNN(n_mels=args.n_mels, dropout_p=args.dropout).to(device)
        strings = STRING_TRIPLETS.get(name, (0, 0, 0))
    else:
        model = DisambiguatorCNN(n_mels=args.n_mels, dropout_p=args.dropout).to(device)
        strings = STRING_PAIRS.get(name, (0, 0))
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience)
    
    best_val_acc, best_epoch, patience_counter = 0.0, -1, 0
    ckpt_path = run_dir / "disambig_best.pt"
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, model_type)
        val_loss, val_acc, midi_acc = evaluate(model, val_loader, criterion, device, model_type)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs} | Train {train_loss:.4f}/{train_acc:.4f} | Val {val_loss:.4f}/{val_acc:.4f}")
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "model_type": model_type,  # 'triplet' or 'pair'
                "name": name,
                "val_acc": val_acc, "epoch": epoch + 1, "midi_acc": midi_acc,
                "sample_rate": train_ds.sample_rate, "attack_samples": train_ds.attack_samples,
                "mel_params": {"n_fft": train_ds.n_fft, "hop_length": train_ds.hop_length,
                               "n_mels": train_ds.n_mels, "f_min": train_ds.f_min,
                               "f_max": train_ds.f_max, "num_frames": train_ds.num_frames},
                "model_params": {"model_class": "TripletDisambiguatorCNN" if model_type == 'triplet' else "DisambiguatorCNN"},
                "strings": strings,
            }, ckpt_path)
            print(f"  *** New best: {val_acc:.4f} ***")
        else:
            patience_counter += 1
        
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return {"name": name, "type": model_type, "best_val_acc": best_val_acc, "best_epoch": best_epoch, "run_dir": str(run_dir)}


def main():
    parser = argparse.ArgumentParser(description="Train disambiguator (triplet or pair)")
    parser.add_argument('--mix-csv', type=str, help='Training mix CSV (single mode)')
    parser.add_argument('--name', type=str, help='Model name (e.g., E_A_D or A_D)')
    parser.add_argument('--type', type=str, choices=['triplet', 'pair'], default='triplet',
                       help='Model type: triplet (3-way) or pair (2-way)')
    parser.add_argument('--batch', action='store_true', help='Train all models of given type')
    parser.add_argument('--mix-dir', type=str, help='Directory with mix CSVs (batch mode)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lr-patience', type=int, default=5)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--n-mels', type=int, default=48)  # Match note model
    parser.add_argument('--hop-length', type=int, default=256)  # Match note model
    parser.add_argument('--num-frames', type=int, default=32)
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--no-augment', dest='augment', action='store_false')
    args = parser.parse_args()
    
    model_type = args.type
    
    if args.batch:
        if not args.mix_dir:
            parser.error("--mix-dir required with --batch")
        
        mix_dir = Path(args.mix_dir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = REPO_ROOT / "artifacts" / "models" / f"disambig_{model_type}_batch_{ts}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Choose which models to train based on type
        if model_type == 'triplet':
            model_names = list(STRING_TRIPLETS.keys())
        else:
            model_names = list(STRING_PAIRS.keys())
        
        results = {}
        for name in model_names:
            mix_csv = mix_dir / f"disambig_{name.lower()}_mix.csv"
            if not mix_csv.exists():
                print(f"Skipping {name}: {mix_csv} not found")
                continue
            print(f"\n{'='*60}\nTraining {name} ({model_type})\n{'='*60}")
            run_dir = batch_dir / name
            try:
                results[name] = train_single(mix_csv, name, model_type, args, run_dir)
                results[name]["status"] = "success"
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        with open(batch_dir / "batch_summary.json", "w") as f:
            json.dump({"timestamp": ts, "type": model_type, "results": results}, f, indent=2)
        
        print(f"\n{'='*60}\nBATCH SUMMARY ({model_type})\n{'='*60}")
        for name, r in results.items():
            if r.get("status") == "success":
                print(f"  {name}: {r['best_val_acc']:.4f} @ epoch {r['best_epoch']+1}")
            else:
                print(f"  {name}: FAILED - {r.get('error', 'unknown')}")
        print(f"\nOutputs: {batch_dir}")
    else:
        if not args.mix_csv or not args.name:
            parser.error("--mix-csv and --name required (or use --batch)")
        train_single(Path(args.mix_csv), args.name, model_type, args)


if __name__ == '__main__':
    main()
