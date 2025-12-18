#!/usr/bin/env python3
"""
Build a training mix CSV optimized for STRING classification.

Balances samples across the 6 guitar strings (not fret positions),
with optional stratification by fret within each string for diversity.

Usage:
    python build_string_training_mix.py \
        --datasets ~/fretlistener/datasets/clean_tones \
        --max-per-string 500 \
        --train-ratio 0.8 \
        --output recipes/string_training_mix.csv \
        --seed 42
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


# -----------------------------
# Utilities
# -----------------------------

def load_metadata_csv(meta_csv: Path) -> List[dict]:
    with meta_csv.open("r", newline="") as f:
        return list(csv.DictReader(f))


def parse_string_value(val: str) -> Optional[int]:
    """
    Parse string field to int 1-6.
    Accepts: "1"-"6", "s1"-"s6", "S1"-"S6"
    """
    if val is None:
        return None
    s = str(val).strip().lower()
    if s.startswith("s") and s[1:].isdigit():
        n = int(s[1:])
    elif s.isdigit():
        n = int(s)
    else:
        return None

    # Handle 0-based datasets (0-5 -> 1-6)
    if n == 0:
        return 1
    if 1 <= n <= 6:
        return n
    if n == 5 and s.startswith("s"):
        # Could be 0-indexed s0-s5
        return 6
    return None


def normalize_row(row: dict, dataset_root: Path) -> Optional[dict]:
    """Normalize a row, ensuring required fields are present and valid."""
    r = dict(row)

    # Parse string (required for string classification)
    string_val = parse_string_value(r.get("string", ""))
    if string_val is None:
        return None
    r["string"] = string_val

    # Parse fret/pos (optional but useful for stratification)
    try:
        r["fret"] = int(r.get("fret", r.get("pos", -1)))
    except (ValueError, TypeError):
        r["fret"] = -1

    try:
        r["pos"] = int(r.get("pos", -1))
    except (ValueError, TypeError):
        r["pos"] = -1

    try:
        r["take"] = int(r.get("take", -1))
    except (ValueError, TypeError):
        r["take"] = -1

    # Resolve file path
    relpath = r.get("relpath") or r.get("filename")
    if not relpath:
        return None

    abs_path = (dataset_root / relpath).resolve()
    if not abs_path.exists():
        return None

    r["dataset_root"] = str(dataset_root)
    r["abs_path"] = str(abs_path)

    return r


def get_group_key(row: dict) -> str:
    """
    Provenance key to keep samples from the same recording together.
    Ensures train/val split doesn't leak data from the same take.
    """
    if row.get("source_relpath"):
        return f"src::{row['source_relpath']}"
    if row.get("dataset_root"):
        return f"ds::{row['dataset_root']}::take::{row.get('take', -1)}"
    return f"take::{row.get('take', -1)}"


# -----------------------------
# Core logic
# -----------------------------

def build_string_training_mix(
    dataset_roots: List[Path],
    max_per_string: Optional[int],
    train_ratio: float,
    output_csv: Path,
    seed: int,
    stratify_by_fret: bool = True,
) -> None:
    random.seed(seed)

    # -------- Load & normalize --------
    all_rows: List[dict] = []

    for root in dataset_roots:
        meta = root / "meta" / "metadata.csv"
        if not meta.exists():
            print(f"Skipping {root}: no metadata.csv found at {meta}")
            continue

        loaded = 0
        for row in load_metadata_csv(meta):
            norm = normalize_row(row, root)
            if norm:
                all_rows.append(norm)
                loaded += 1

        print(f"Loaded {loaded} samples from {root}")

    if not all_rows:
        raise SystemExit("No valid samples loaded.")

    print(f"Total samples loaded: {len(all_rows)}")

    # -------- Group by string --------
    by_string: Dict[int, List[dict]] = defaultdict(list)
    for r in all_rows:
        by_string[r["string"]].append(r)

    print("\nSamples per string (before balancing):")
    for s in sorted(by_string.keys()):
        print(f"  String {s}: {len(by_string[s])} samples")

    # -------- Per-string selection with fret stratification --------
    selected: List[dict] = []

    for string_num, rows in sorted(by_string.items()):
        rows_copy = rows[:]
        random.shuffle(rows_copy)

        if max_per_string and len(rows_copy) > max_per_string:
            if stratify_by_fret:
                # Group by fret within this string for diversity
                by_fret: Dict[int, List[dict]] = defaultdict(list)
                for r in rows_copy:
                    by_fret[r["fret"]].append(r)

                for fret_rows in by_fret.values():
                    random.shuffle(fret_rows)

                # Round-robin selection across frets
                balanced = []
                fret_keys = list(by_fret.keys())
                idx = {k: 0 for k in fret_keys}

                while len(balanced) < max_per_string:
                    progressed = False
                    for fret in fret_keys:
                        if idx[fret] < len(by_fret[fret]) and len(balanced) < max_per_string:
                            balanced.append(by_fret[fret][idx[fret]])
                            idx[fret] += 1
                            progressed = True
                    if not progressed:
                        break

                rows_copy = balanced
            else:
                # Simple random selection
                rows_copy = rows_copy[:max_per_string]

        selected.extend(rows_copy)
        print(f"String {string_num}: selected {len(rows_copy)} samples")

    print(f"\nTotal selected after balancing: {len(selected)}")

    # -------- Group by provenance for train/val split --------
    by_group: Dict[str, List[dict]] = defaultdict(list)
    for r in selected:
        by_group[get_group_key(r)].append(r)

    groups = list(by_group.items())
    random.shuffle(groups)

    total = sum(len(g[1]) for g in groups)
    target_train = int(total * train_ratio)

    # -------- Split groups (not individual samples) --------
    train_groups = set()
    val_groups = set()

    running = 0
    for key, rows in groups:
        if running < target_train:
            train_groups.add(key)
            running += len(rows)
        else:
            val_groups.add(key)

    # Ensure at least one group in val
    if not val_groups and groups:
        key, _ = groups[-1]
        train_groups.discard(key)
        val_groups.add(key)

    # -------- Assign splits --------
    final_rows = []
    for key, rows in by_group.items():
        split = "train" if key in train_groups else "val"
        for r in rows:
            r_copy = dict(r)
            r_copy["split"] = split
            final_rows.append(r_copy)

    # -------- Coverage check --------
    train_strings = defaultdict(int)
    val_strings = defaultdict(int)

    for r in final_rows:
        if r["split"] == "train":
            train_strings[r["string"]] += 1
        else:
            val_strings[r["string"]] += 1

    print("\nFinal distribution:")
    print("  String | Train | Val")
    print("  -------+-------+-----")
    for s in range(1, 7):
        print(f"       {s} | {train_strings[s]:5d} | {val_strings[s]:4d}")

    missing_train = [s for s in range(1, 7) if train_strings[s] == 0]
    missing_val = [s for s in range(1, 7) if val_strings[s] == 0]

    if missing_train:
        print(f"WARNING: strings missing in train: {missing_train}")
    if missing_val:
        print(f"WARNING: strings missing in val: {missing_val}")

    train_total = sum(train_strings.values())
    val_total = sum(val_strings.values())
    print(f"\nFinal split: {train_total} train / {val_total} val")

    # -------- Write CSV --------
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id", "dataset_name", "dataset_root", "relpath", "abs_path",
        "filename", "string", "fret", "pos", "midi", "note",
        "take", "sample_rate", "num_samples", "source_relpath",
        "split",
    ]

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in final_rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\nTraining mix written to {output_csv}")


# -----------------------------
# CLI
# -----------------------------

def main():
    p = argparse.ArgumentParser(
        description="Build training mix CSV optimized for string classification."
    )
    p.add_argument(
        "--datasets", nargs="+", required=True,
        help="Paths to dataset root directories (each should have meta/metadata.csv)"
    )
    p.add_argument(
        "--max-per-string", type=int, default=None,
        help="Maximum samples per string (for balancing). Default: no limit."
    )
    p.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="Fraction of data for training (default: 0.8)"
    )
    p.add_argument(
        "--output", type=str, default="recipes/string_training_mix.csv",
        help="Output CSV path"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    p.add_argument(
        "--no-stratify-fret", action="store_true",
        help="Disable fret stratification within each string"
    )

    args = p.parse_args()

    build_string_training_mix(
        dataset_roots=[Path(d).expanduser() for d in args.datasets],
        max_per_string=args.max_per_string,
        train_ratio=args.train_ratio,
        output_csv=Path(args.output).expanduser(),
        seed=args.seed,
        stratify_by_fret=not args.no_stratify_fret,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)