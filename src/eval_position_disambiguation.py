# src/eval_position_disambiguation.py

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from guitar_note_dataset import GuitarNoteDataset
from multihead_cnn import GuitarMultiHeadCNN


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def init_model(device: torch.device) -> torch.nn.Module:
    model = GuitarMultiHeadCNN(
        n_mels=64,
        num_notes=12,
        num_strings=6,
        num_frets=6,
        hidden_dim=128,
    )
    ckpt_path = Path("pilot0_multihead_cnn.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_position_disambiguation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    """
    Runs evaluation on the provided loader and prints:

      - Global confusion matrices for string and fret
      - Ambiguous-only confusion matrices
      - Per-ambiguous-MIDI accuracy and confusion counts
    """
    # Confusion matrices:
    # string: rows = true string (1..6), cols = pred string (1..6)
    # fret:   rows = true fret (0..5),  cols = pred fret (0..5)
    string_conf = torch.zeros((7, 7), dtype=torch.long)  # index by [true, pred], 1..6 used
    fret_conf = torch.zeros((6, 6), dtype=torch.long)    # index by [true, pred], 0..5

    string_conf_amb = torch.zeros((7, 7), dtype=torch.long)
    fret_conf_amb = torch.zeros((6, 6), dtype=torch.long)

    # Per-MIDI stats for ambiguous notes
    # midi -> (correct_count, total_count)
    midi_correct = {}
    midi_total = {}

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            string_true = labels["string"]          # [B]
            fret_true = labels["fret"]              # [B]
            midi_true = labels["midi"]              # [B]
            is_ambiguous = labels["is_ambiguous"]   # [B]

            outputs = model(features)
            logits_string = outputs["string"]       # [B, 6]
            logits_fret = outputs["fret"]           # [B, 6]

            string_pred = logits_string.argmax(dim=1).cpu() + 1  # 0..5 → 1..6
            fret_pred = logits_fret.argmax(dim=1).cpu()          # 0..5
            string_true_cpu = string_true.cpu()
            fret_true_cpu = fret_true.cpu()
            midi_true_cpu = midi_true.cpu()
            is_ambig_cpu = is_ambiguous.cpu()

            # Global confusion
            for t_s, p_s, t_f, p_f in zip(
                string_true_cpu, string_pred, fret_true_cpu, fret_pred
            ):
                string_conf[t_s, p_s] += 1
                fret_conf[t_f, p_f] += 1

            # Ambiguous-only
            for t_s, p_s, t_f, p_f, midi, amb in zip(
                string_true_cpu, string_pred, fret_true_cpu, fret_pred, midi_true_cpu, is_ambig_cpu
            ):
                if not amb:
                    continue

                string_conf_amb[t_s, p_s] += 1
                fret_conf_amb[t_f, p_f] += 1

                m = int(midi.item())
                midi_total[m] = midi_total.get(m, 0) + 1
                correct_pos = int(t_s.item() == p_s.item() and t_f.item() == p_f.item())
                midi_correct[m] = midi_correct.get(m, 0) + correct_pos

    # Print confusion matrices
    print("=== Global STRING confusion (rows=true, cols=pred, 1..6) ===")
    for t_s in range(1, 7):
        row = " ".join(f"{int(string_conf[t_s, p_s]):3d}" for p_s in range(1, 7))
        print(f"true {t_s}: {row}")

    print("\n=== Global FRET confusion (rows=true, cols=pred, 0..5) ===")
    for t_f in range(0, 6):
        row = " ".join(f"{int(fret_conf[t_f, p_f]):3d}" for p_f in range(0, 6))
        print(f"true {t_f}: {row}")

    print("\n=== AMBIGUOUS-ONLY STRING confusion (rows=true, cols=pred, 1..6) ===")
    for t_s in range(1, 7):
        row = " ".join(f"{int(string_conf_amb[t_s, p_s]):3d}" for p_s in range(1, 7))
        print(f"true {t_s}: {row}")

    print("\n=== AMBIGUOUS-ONLY FRET confusion (rows=true, cols=pred, 0..5) ===")
    for t_f in range(0, 6):
        row = " ".join(f"{int(fret_conf_amb[t_f, p_f]):3d}" for p_f in range(0, 6))
        print(f"true {t_f}: {row}")

    # Ambiguous MIDI per-note stats
    print("\n=== Ambiguous MIDI per-note position accuracy ===")
    for midi, total in sorted(midi_total.items()):
        correct = midi_correct.get(midi, 0)
        acc = correct / total if total > 0 else 0.0
        print(f"MIDI {midi}: correct={correct}/{total} acc={acc:.3f}")


def main():
    device = select_device()
    print(f"Using device: {device}")

    # test split only
    test_ds = GuitarNoteDataset(split="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = init_model(device)
    evaluate_position_disambiguation(model, test_loader, device)


if __name__ == "__main__":
    main()
