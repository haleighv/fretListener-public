# src/debug_dataloader.py

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from guitar_note_dataset import GuitarNoteDataset


def main():
    # Basic device selection (no model yet)
    mps_available = torch.backends.mps.is_available()
    device = torch.device("mps" if mps_available else "cpu")
    print(f"MPS available: {mps_available}, using device: {device}")

    train_ds = GuitarNoteDataset(split="train")
    val_ds = GuitarNoteDataset(split="val")
    test_ds = GuitarNoteDataset(split="test")

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    print(f"Test samples:  {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=0,  # keep 0 for debugging
        pin_memory=False,
    )

    # Inspect first batch
    batch = next(iter(train_loader))
    features, labels = batch

    # features: [B, 1, n_mels, T]
    print(f"features.shape: {features.shape}")
    print(f"note_id.shape:  {labels['note_id'].shape}")
    print(f"string.shape:   {labels['string'].shape}")
    print(f"fret.shape:     {labels['fret'].shape}")
    print(f"midi.shape:     {labels['midi'].shape}")
    print(f"is_ambiguous.shape: {labels['is_ambiguous'].shape}")

    # Optional: move one batch to device as a smoke test
    features = features.to(device)
    labels_device = {k: v.to(device) for k, v in labels.items()}
    print("Moved one batch to device successfully.")


if __name__ == "__main__":
    main()
