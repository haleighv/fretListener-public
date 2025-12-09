# src/train_pilot0.py

import csv
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from guitar_note_dataset import GuitarNoteDataset
from multihead_cnn import GuitarMultiHeadCNN


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion_note: nn.Module,
    criterion_string: nn.Module,
    criterion_fret: nn.Module,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_examples = 0

    correct_note = 0
    correct_string = 0
    correct_fret = 0
    correct_all = 0

    for features, labels in loader:
        features = features.to(device)  # [B, 1, n_mels, T]
        note_targets = labels["note_id"].to(device)   # [B]
        string_targets = labels["string"].to(device)  # [B]
        fret_targets = labels["fret"].to(device)      # [B]

        optimizer.zero_grad()

        outputs = model(features)
        logits_note = outputs["note"]
        logits_string = outputs["string"]
        logits_fret = outputs["fret"]

        loss_note = criterion_note(logits_note, note_targets)
        loss_string = criterion_string(logits_string, string_targets)
        loss_fret = criterion_fret(logits_fret, fret_targets)

        loss = loss_note + loss_string + loss_fret
        loss.backward()
        optimizer.step()

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        # Accuracies
        preds_note = logits_note.argmax(dim=1)
        preds_string = logits_string.argmax(dim=1)
        preds_fret = logits_fret.argmax(dim=1)

        correct_note += (preds_note == note_targets).sum().item()
        correct_string += (preds_string == string_targets).sum().item()
        correct_fret += (preds_fret == fret_targets).sum().item()

        all_correct = (
            (preds_note == note_targets)
            & (preds_string == string_targets)
            & (preds_fret == fret_targets)
        )
        correct_all += all_correct.sum().item()

    avg_loss = total_loss / total_examples if total_examples > 0 else 0.0
    acc_note = correct_note / total_examples if total_examples > 0 else 0.0
    acc_string = correct_string / total_examples if total_examples > 0 else 0.0
    acc_fret = correct_fret / total_examples if total_examples > 0 else 0.0
    acc_all = correct_all / total_examples if total_examples > 0 else 0.0

    return {
        "loss": avg_loss,
        "acc_note": acc_note,
        "acc_string": acc_string,
        "acc_fret": acc_fret,
        "acc_all": acc_all,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion_note: nn.Module,
    criterion_string: nn.Module,
    criterion_fret: nn.Module,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0

    correct_note = 0
    correct_string = 0
    correct_fret = 0
    correct_all = 0

    for features, labels in loader:
        features = features.to(device)
        note_targets = labels["note_id"].to(device)
        string_targets = labels["string"].to(device)
        fret_targets = labels["fret"].to(device)

        outputs = model(features)
        logits_note = outputs["note"]
        logits_string = outputs["string"]
        logits_fret = outputs["fret"]

        loss_note = criterion_note(logits_note, note_targets)
        loss_string = criterion_string(logits_string, string_targets)
        loss_fret = criterion_fret(logits_fret, fret_targets)

        loss = loss_note + loss_string + loss_fret

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        preds_note = logits_note.argmax(dim=1)
        preds_string = logits_string.argmax(dim=1)
        preds_fret = logits_fret.argmax(dim=1)

        correct_note += (preds_note == note_targets).sum().item()
        correct_string += (preds_string == string_targets).sum().item()
        correct_fret += (preds_fret == fret_targets).sum().item()

        all_correct = (
            (preds_note == note_targets)
            & (preds_string == string_targets)
            & (preds_fret == fret_targets)
        )
        correct_all += all_correct.sum().item()

    avg_loss = total_loss / total_examples if total_examples > 0 else 0.0
    acc_note = correct_note / total_examples if total_examples > 0 else 0.0
    acc_string = correct_string / total_examples if total_examples > 0 else 0.0
    acc_fret = correct_fret / total_examples if total_examples > 0 else 0.0
    acc_all = correct_all / total_examples if total_examples > 0 else 0.0

    return {
        "loss": avg_loss,
        "acc_note": acc_note,
        "acc_string": acc_string,
        "acc_fret": acc_fret,
        "acc_all": acc_all,
    }


def main():
    device = select_device()
    print(f"Using device: {device}")

    # Datasets / loaders
    train_ds = GuitarNoteDataset(split="train")
    val_ds = GuitarNoteDataset(split="val")
    test_ds = GuitarNoteDataset(split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Model
    model = GuitarMultiHeadCNN(
        n_mels=64,
        num_notes=12,
        num_strings=6,
        num_frets=6,
        hidden_dim=128,
    ).to(device)

    # Losses and optimizer
    criterion_note = nn.CrossEntropyLoss()
    criterion_string = nn.CrossEntropyLoss()
    criterion_fret = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    num_epochs = 20

    # Per-epoch history for plotting later
    history = []

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion_note,
            criterion_string,
            criterion_fret,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion_note,
            criterion_string,
            criterion_fret,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc_note={train_metrics['acc_note']:.3f} "
            f"train_acc_str={train_metrics['acc_string']:.3f} "
            f"train_acc_fret={train_metrics['acc_fret']:.3f} "
            f"train_acc_all={train_metrics['acc_all']:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc_note={val_metrics['acc_note']:.3f} "
            f"val_acc_str={val_metrics['acc_string']:.3f} "
            f"val_acc_fret={val_metrics['acc_fret']:.3f} "
            f"val_acc_all={val_metrics['acc_all']:.3f}"
        )

        # Log to history
        history.append({
            "epoch": epoch,
            "split": "train",
            "loss": train_metrics["loss"],
            "acc_note": train_metrics["acc_note"],
            "acc_string": train_metrics["acc_string"],
            "acc_fret": train_metrics["acc_fret"],
            "acc_all": train_metrics["acc_all"],
        })
        history.append({
            "epoch": epoch,
            "split": "val",
            "loss": val_metrics["loss"],
            "acc_note": val_metrics["acc_note"],
            "acc_string": val_metrics["acc_string"],
            "acc_fret": val_metrics["acc_fret"],
            "acc_all": val_metrics["acc_all"],
        })

    # Save training/validation history next to this script
    script_dir = Path(__file__).resolve().parent
    history_path = script_dir / "pilot0_training_history.csv"
    fieldnames = ["epoch", "split", "loss", "acc_note", "acc_string", "acc_fret", "acc_all"]
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)
    print(f"Saved training history to {history_path}")

    # Final test evaluation
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        criterion_note,
        criterion_string,
        criterion_fret,
    )
    print(
        "Test "
        f"loss={test_metrics['loss']:.4f} "
        f"acc_note={test_metrics['acc_note']:.3f} "
        f"acc_str={test_metrics['acc_string']:.3f} "
        f"acc_fret={test_metrics['acc_fret']:.3f} "
        f"acc_all={test_metrics['acc_all']:.3f}"
    )

    # Save checkpoint next to this script
    ckpt_path = script_dir / "pilot0_multihead_cnn.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_epochs": num_epochs,
        },
        ckpt_path,
    )
    print(f"Saved model checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
