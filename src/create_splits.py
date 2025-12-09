# src/create_splits.py

import csv
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META_DIR = ROOT / "data" / "pilot_0" / "guitar_dataset" / "meta"

FEATURE_META = META_DIR / "pilot_0_features_metadata.csv"
SPLITS_CSV = META_DIR / "pilot_0_splits.csv"

RANDOM_SEED = 42


def create_splits():
    if not FEATURE_META.exists():
        raise FileNotFoundError(f"Feature metadata not found: {FEATURE_META}")

    with FEATURE_META.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group rows by (string, fret)
    groups = defaultdict(list)
    for r in rows:
        s = int(r["string"])
        f = int(r["fret"])
        groups[(s, f)].append(r)

    random.seed(RANDOM_SEED)

    split_rows = []
    for (s, f), lst in groups.items():
        # Shuffle in-place
        random.shuffle(lst)
        n = len(lst)
        if n < 3:
            raise ValueError(f"Not enough samples for position (string={s}, fret={f}), got {n}")

        # For pilot_0: n should be 5
        # Use 3/1/1 split where possible
        n_train = min(3, n)
        n_val = max(1, (n - n_train) // 2)
        n_test = n - n_train - n_val

        # Assign splits
        for i, r in enumerate(lst):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"

            r_out = dict(r)
            r_out["split"] = split
            split_rows.append(r_out)

    fieldnames = list(rows[0].keys()) + ["split"]
    with SPLITS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in split_rows:
            writer.writerow(r)

    print(f"Wrote {len(split_rows)} rows with splits to {SPLITS_CSV}")


if __name__ == "__main__":
    create_splits()
