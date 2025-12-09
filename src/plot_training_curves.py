# src/plot_training_curves.py

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[0]
HISTORY_CSV = ROOT / "pilot0_training_history.csv"


def load_history():
    epochs = []
    train_loss = []
    val_loss = []
    train_acc_all = []
    val_acc_all = []
    train_acc_note = []
    val_acc_note = []
    train_acc_string = []
    val_acc_string = []
    train_acc_fret = []
    val_acc_fret = []

    with HISTORY_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"])
            split = row["split"]
            loss = float(row["loss"])
            acc_note = float(row["acc_note"])
            acc_string = float(row["acc_string"])
            acc_fret = float(row["acc_fret"])
            acc_all = float(row["acc_all"])

            if split == "train":
                epochs.append(epoch)
                train_loss.append(loss)
                train_acc_all.append(acc_all)
                train_acc_note.append(acc_note)
                train_acc_string.append(acc_string)
                train_acc_fret.append(acc_fret)
            elif split == "val":
                val_loss.append(loss)
                val_acc_all.append(acc_all)
                val_acc_note.append(acc_note)
                val_acc_string.append(acc_string)
                val_acc_fret.append(acc_fret)

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc_all": train_acc_all,
        "val_acc_all": val_acc_all,
        "train_acc_note": train_acc_note,
        "val_acc_note": val_acc_note,
        "train_acc_string": train_acc_string,
        "val_acc_string": val_acc_string,
        "train_acc_fret": train_acc_fret,
        "val_acc_fret": val_acc_fret,
    }


def plot_curves():
    if not HISTORY_CSV.exists():
        raise FileNotFoundError(f"History CSV not found: {HISTORY_CSV}")

    hist = load_history()
    epochs = hist["epochs"]

    # 1) Loss
    plt.figure()
    plt.plot(epochs, hist["train_loss"], label="train")
    plt.plot(epochs, hist["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Pilot0: Loss vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "pilot0_loss.png")

    # 2) All-heads-correct accuracy
    plt.figure()
    plt.plot(epochs, hist["train_acc_all"], label="train")
    plt.plot(epochs, hist["val_acc_all"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Exact-match accuracy")
    plt.title("Pilot0: Exact-match Acc vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "pilot0_acc_all.png")

    # 3) Note accuracy
    plt.figure()
    plt.plot(epochs, hist["train_acc_note"], label="train")
    plt.plot(epochs, hist["val_acc_note"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Note accuracy")
    plt.title("Pilot0: Note Acc vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "pilot0_acc_note.png")

    # 4) String accuracy
    plt.figure()
    plt.plot(epochs, hist["train_acc_string"], label="train")
    plt.plot(epochs, hist["val_acc_string"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("String accuracy")
    plt.title("Pilot0: String Acc vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "pilot0_acc_string.png")

    # 5) Fret accuracy
    plt.figure()
    plt.plot(epochs, hist["train_acc_fret"], label="train")
    plt.plot(epochs, hist["val_acc_fret"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Fret accuracy")
    plt.title("Pilot0: Fret Acc vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "pilot0_acc_fret.png")

    print("Saved plots to:")
    print("  pilot0_loss.png")
    print("  pilot0_acc_all.png")
    print("  pilot0_acc_note.png")
    print("  pilot0_acc_string.png")
    print("  pilot0_acc_fret.png")


if __name__ == "__main__":
    plot_curves()
