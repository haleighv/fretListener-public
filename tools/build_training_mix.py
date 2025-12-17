#!/usr/bin/env python

import argparse
import sys
from pathlib import Path
import csv
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# -----------------------------
# Utilities
# -----------------------------

def load_metadata_csv(meta_csv: Path) -> List[dict]:
    with meta_csv.open("r", newline="") as f:
        return list(csv.DictReader(f))


def normalize_row(row: dict, dataset_root: Path) -> Optional[dict]:
    r = dict(row)

    try:
        r["pos"] = int(r["pos"])
    except Exception:
        return None

    try:
        r["take"] = int(r.get("take", -1))
    except Exception:
        r["take"] = -1

    relpath = r.get("relpath") or r.get("filename")
    if not relpath:
        return None

    r["dataset_root"] = str(dataset_root)
    r["abs_path"] = str((dataset_root / relpath).resolve())

    return r


def get_group_key(row: dict) -> str:
    """
    Strongest available provenance key.
    Ensures all samples from the same recording stay together.
    """
    if row.get("source_relpath"):
        return f"src::{row['source_relpath']}"
    if row.get("dataset_root"):
        return f"ds::{row['dataset_root']}"
    return f"take::{row.get('take', -1)}"


# -----------------------------
# Core logic
# -----------------------------

def build_training_mix(
    dataset_roots: List[Path],
    max_per_class: Optional[int],
    train_ratio: float,
    output_csv: Path,
    seed: int,
) -> None:
    random.seed(seed)

    # -------- Load & normalize --------
    all_rows: List[dict] = []

    for root in dataset_roots:
        meta = root / "meta" / "metadata.csv"
        if not meta.exists():
            print(f"Skipping {root}: no metadata.csv")
            continue

        for row in load_metadata_csv(meta):
            norm = normalize_row(row, root)
            if norm:
                all_rows.append(norm)

    if not all_rows:
        raise SystemExit("No valid samples loaded.")

    print(f"Loaded {len(all_rows)} total samples")

    # -------- Group by class --------
    by_pos: Dict[int, List[dict]] = defaultdict(list)
    for r in all_rows:
        by_pos[r["pos"]].append(r)

    # -------- Per-class selection (with diversity) --------
    selected: List[dict] = []

    for pos, rows in sorted(by_pos.items()):
        rows_copy = rows[:]
        random.shuffle(rows_copy)

        if max_per_class and len(rows_copy) > max_per_class:
            by_group = defaultdict(list)
            for r in rows_copy:
                by_group[get_group_key(r)].append(r)

            for g in by_group.values():
                random.shuffle(g)

            balanced = []
            group_keys = list(by_group.keys())
            idx = {k: 0 for k in group_keys}

            while len(balanced) < max_per_class:
                progressed = False
                for k in group_keys:
                    if idx[k] < len(by_group[k]) and len(balanced) < max_per_class:
                        balanced.append(by_group[k][idx[k]])
                        idx[k] += 1
                        progressed = True
                if not progressed:
                    break

            rows_copy = balanced

        selected.extend(rows_copy)
        print(f"pos={pos}: using {len(rows_copy)} samples")

    print(f"Selected {len(selected)} samples after balancing")

    # -------- Group by provenance --------
    by_group: Dict[str, List[dict]] = defaultdict(list)
    for r in selected:
        by_group[get_group_key(r)].append(r)

    groups = list(by_group.items())
    random.shuffle(groups)

    group_sizes = [len(g[1]) for g in groups]
    total = sum(group_sizes)
    target_train = int(total * train_ratio)

    # -------- Split groups (not samples) --------
    train_groups = set()
    val_groups = set()

    running = 0
    for key, rows in groups:
        if running < target_train:
            train_groups.add(key)
            running += len(rows)
        else:
            val_groups.add(key)

    if not val_groups:
        # Force at least one group into val
        key, _ = groups[-1]
        train_groups.remove(key)
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
    train_classes = {r["pos"] for r in final_rows if r["split"] == "train"}
    val_classes = {r["pos"] for r in final_rows if r["split"] == "val"}

    missing_val = train_classes - val_classes
    if missing_val:
        print(f"WARNING: classes missing in val: {sorted(missing_val)}")

    print(
        f"Final split: "
        f"{sum(r['split']=='train' for r in final_rows)} train / "
        f"{sum(r['split']=='val' for r in final_rows)} val"
    )

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

    print(f"Training mix written to {output_csv}")


# -----------------------------
# CLI
# -----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--max-per-class", type=int)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--output", type=str, default="recipes/training_mix.csv")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    build_training_mix(
        dataset_roots=[Path(d).expanduser() for d in args.datasets],
        max_per_class=args.max_per_class,
        train_ratio=args.train_ratio,
        output_csv=Path(args.output).expanduser(),
        seed=args.seed,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
