# src/guitar_note_dataset.py

from pathlib import Path
import csv
from typing import Literal, List, Dict, Any

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "pilot_0" / "guitar_dataset"
FEATURES_DIR = DATA_ROOT / "features"
META_DIR = DATA_ROOT / "meta"
SPLITS_CSV = META_DIR / "pilot_0_splits.csv"


class GuitarNoteDataset(Dataset):
    """
    Dataset over precomputed feature tensors for the guitar pilot_0 data.

    Each sample returns:
        features: Tensor[1, n_mels, T]
        labels: {
            "note_id": int,
            "string": int,
            "fret": int,
            "midi": int,
            "is_ambiguous": bool,
        }
    """

    def __init__(self, split: Literal["train", "val", "test", "all"] = "train"):
        if not SPLITS_CSV.exists():
            raise FileNotFoundError(f"Splits CSV not found at {SPLITS_CSV}")

        self.split = split
        self.samples: List[Dict[str, Any]] = []

        with SPLITS_CSV.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                r_split = r["split"]
                if split != "all" and r_split != split:
                    continue
                self.samples.append(r)

        if not self.samples:
            raise RuntimeError(f"No samples found for split='{split}'")

        # Precompute feature paths and label info
        for r in self.samples:
            fname = r["filename"]
            pt_name = Path(fname).with_suffix(".pt").name
            r["feature_path"] = FEATURES_DIR / pt_name
            r["string"] = int(r["string"])
            r["fret"] = int(r["fret"])
            r["octave"] = int(r["octave"])
            r["take"] = int(r["take"])
            r["midi"] = int(r["midi"])
            r["is_ambiguous"] = r["is_ambiguous"].lower() == "true"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        r = self.samples[idx]
        pt_path: Path = r["feature_path"]

        if not pt_path.exists():
            raise FileNotFoundError(f"Feature tensor not found: {pt_path}")

        bundle = torch.load(pt_path)

        features = bundle["features"]  # Tensor[1, n_mels, T]
        # Use the precomputed note_id from bundle to keep it consistent
        note_id = int(bundle["note_id"])
        string = int(bundle["string"])
        fret = int(bundle["fret"])
        midi = int(bundle["midi"])
        is_ambiguous = bool(bundle["is_ambiguous"])

        labels = {
            "note_id": torch.tensor(note_id, dtype=torch.long),
            "string": torch.tensor(string, dtype=torch.long),
            "fret": torch.tensor(fret, dtype=torch.long),
            "midi": torch.tensor(midi, dtype=torch.long),
            "is_ambiguous": torch.tensor(is_ambiguous, dtype=torch.bool),
        }

        return features, labels
