import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torchaudio
import numpy as np
from attack_model import AttackPositionNet

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def parse_class_name(name):
    s, f = name.split("_f")
    string = int(s[1:])
    fret = int(f)
    pos = (string - 1) * 8 + fret
    return pos

class AttackDataset(Dataset):
    def __init__(self, root, max_per_class=40):
        self.items = []
        self.root = root

        for d in sorted(root.iterdir()):
            if d.is_dir():
                pos = parse_class_name(d.name)
                wavs = sorted(d.glob("*.wav"))[:max_per_class]
                for w in wavs:
                    self.items.append((w, pos))

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=48000,
            n_fft=1024,
            hop_length=256,
            n_mels=64,
            f_min=55,
            f_max=2000
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, pos = self.items[idx]
        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0, keepdim=True)

        if wav.size(1) < 9120:
            wav = torch.nn.functional.pad(wav, (0, 9120 - wav.size(1)))
        else:
            wav = wav[:, :9120]

        spec = self.db(self.mel(wav))
        spec = spec[:, :, :32]
        return spec, pos

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("attack_position_48class.pt"))
    args = parser.parse_args()

    dataset = AttackDataset(args.dataset_root)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=64, sampler=RandomSampler(train_ds), num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, num_workers=0)

    device = get_device()
    print("Using device:", device)

    model = AttackPositionNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0

    for epoch in range(1, 21):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        print(f"Epoch {epoch} | Val Acc {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({"model_state": model.state_dict()}, args.checkpoint)
            print("Saved new best checkpoint")

if __name__ == "__main__":
    main()
