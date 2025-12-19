#!/usr/bin/env python3
"""
build_string_training_mix.py - Build training mix for string classification

Balances by STRING (6 classes) instead of position.
Optionally upweights "ambiguous" positions where open strings share
the same pitch as fretted notes on adjacent strings:

  Open E4 (s1, f0) <-> B string fret 5 (s2, f5)   - same pitch
  Open B3 (s2, f0) <-> G string fret 4 (s3, f4)   - same pitch  
  Open G3 (s3, f0) <-> D string fret 5 (s4, f5)   - same pitch
  Open D3 (s4, f0) <-> A string fret 5 (s5, f5)   - same pitch
  Open A2 (s5, f0) <-> E string fret 5 (s6, f5)   - same pitch

Usage:
  python build_string_training_mix.py \
      --datasets data/strat_neck_119 data/rg550_neck_20 \
      --max-per-string 1500 \
      --ambiguous-multiplier 2.0 \
      --output recipes/string_training_mix.csv
"""

import argparse
import sys
from pathlib import Path
import csv
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


# Ambiguous (string, fret) pairs - these share pitch with adjacent string
# Format: (string, fret) pairs that produce same pitch as an open string
AMBIGUOUS_POSITIONS: Set[Tuple[int, int]] = {
    # Open strings (the "target" side of ambiguity)
    (1, 0),  # Open E4
    (2, 0),  # Open B3
    (3, 0),  # Open G3
    (4, 0),  # Open D3
    (5, 0),  # Open A2
    # Fretted equivalents (same pitch as open string above)
    (2, 5),  # B string fret 5 = E4
    (3, 4),  # G string fret 4 = B3
    (4, 5),  # D string fret 5 = G3
    (5, 5),  # A string fret 5 = D3
    (6, 5),  # E string fret 5 = A2
}


def load_metadata_csv(meta_csv: Path) -> List[dict]:
    with meta_csv.open("r", newline="") as f:
        return list(csv.DictReader(f))


def normalize_row(row: dict, dataset_root: Path) -> Optional[dict]:
    """Normalize a row, ensuring string field exists."""
    r = dict(row)

    # Must have string field
    try:
        r["string"] = int(r["string"])
    except (KeyError, ValueError):
        return None

    # Must have fret field
    try:
        r["fret"] = int(r["fret"])
    except (KeyError, ValueError):
        r["fret"] = 0

    try:
        r["pos"] = int(r.get("pos", 0))
    except Exception:
        r["pos"] = 0

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
    """Provenance key to keep samples from same recording together."""
    if row.get("source_relpath"):
        return f"src::{row['source_relpath']}"
    if row.get("dataset_root"):
        return f"ds::{row['dataset_root']}::{row.get('filename', '')}"
    return f"take::{row.get('take', -1)}"


def is_ambiguous(string: int, fret: int) -> bool:
    """Check if this position is one of the ambiguous open/fret pairs."""
    return (string, fret) in AMBIGUOUS_POSITIONS


def build_string_training_mix(
    dataset_roots: List[Path],
    max_per_string: Optional[int],
    train_ratio: float,
    output_csv: Path,
    seed: int,
    stratify_by_fret: bool = True,
    ambiguous_multiplier: float = 1.0,
) -> None:
    """
    Build training mix balanced by string with optional ambiguous position upweighting.
    
    Args:
        ambiguous_multiplier: How much to oversample ambiguous positions (1.0 = no change, 2.0 = 2x samples)
    """
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

    # -------- Group by string --------
    by_string: Dict[int, List[dict]] = defaultdict(list)
    for r in all_rows:
        by_string[r["string"]].append(r)

    print(f"Samples per string before balancing:")
    for s in sorted(by_string.keys()):
        print(f"  String {s}: {len(by_string[s])} samples")

    # -------- Per-string selection with fret stratification and ambiguous upweighting --------
    selected: List[dict] = []

    for string_num, rows in sorted(by_string.items()):
        rows_copy = rows[:]
        random.shuffle(rows_copy)

        target_count = max_per_string if max_per_string else len(rows_copy)

        if stratify_by_fret:
            # Group by fret within this string
            by_fret: Dict[int, List[dict]] = defaultdict(list)
            for r in rows_copy:
                by_fret[r["fret"]].append(r)

            for fret_rows in by_fret.values():
                random.shuffle(fret_rows)

            # Determine base samples per fret
            num_frets = len(by_fret)
            if num_frets == 0:
                continue

            # Calculate ambiguous frets for this string
            ambiguous_frets = {f for (s, f) in AMBIGUOUS_POSITIONS if s == string_num}
            
            # If we're upweighting ambiguous positions, adjust the selection
            if ambiguous_multiplier > 1.0 and ambiguous_frets:
                # Calculate how many extra samples we want for ambiguous frets
                non_ambiguous_frets = [f for f in by_fret.keys() if f not in ambiguous_frets]
                num_ambiguous = len(ambiguous_frets)
                num_non_ambiguous = len(non_ambiguous_frets)
                
                # Solve for base_per_fret such that:
                # num_non_ambiguous * base + num_ambiguous * base * multiplier = target_count
                # base * (num_non_ambiguous + num_ambiguous * multiplier) = target_count
                effective_frets = num_non_ambiguous + num_ambiguous * ambiguous_multiplier
                base_per_fret = target_count / effective_frets if effective_frets > 0 else target_count / num_frets
                
                balanced = []
                for fret, fret_rows in by_fret.items():
                    if fret in ambiguous_frets:
                        fret_target = int(base_per_fret * ambiguous_multiplier)
                    else:
                        fret_target = int(base_per_fret)
                    
                    # Take up to fret_target samples, cycling if needed for upweighting
                    if len(fret_rows) >= fret_target:
                        balanced.extend(fret_rows[:fret_target])
                    else:
                        # Cycle through available samples to reach target (for upweighting)
                        cycles = fret_target // len(fret_rows)
                        remainder = fret_target % len(fret_rows)
                        for _ in range(cycles):
                            balanced.extend(fret_rows)
                        balanced.extend(fret_rows[:remainder])
                
                rows_copy = balanced
            else:
                # Original round-robin across frets
                base_per_fret = target_count // num_frets

                balanced = []
                frets = list(by_fret.keys())
                idx = {f: 0 for f in frets}

                # Round-robin selection
                while len(balanced) < target_count:
                    progressed = False
                    for f in frets:
                        if idx[f] < len(by_fret[f]) and len(balanced) < target_count:
                            balanced.append(by_fret[f][idx[f]])
                            idx[f] += 1
                            progressed = True
                    if not progressed:
                        break

                rows_copy = balanced
        else:
            # No stratification - just limit total
            if max_per_string and len(rows_copy) > max_per_string:
                rows_copy = rows_copy[:max_per_string]

        selected.extend(rows_copy)
        
        # Count ambiguous samples selected for this string
        ambiguous_count = sum(1 for r in rows_copy if is_ambiguous(r["string"], r["fret"]))
        print(f"String {string_num}: selected {len(rows_copy)} samples ({ambiguous_count} ambiguous)")

    print(f"Selected {len(selected)} samples after balancing")

    # -------- Group by provenance for train/val split --------
    by_group: Dict[str, List[dict]] = defaultdict(list)
    for r in selected:
        by_group[get_group_key(r)].append(r)

    groups = list(by_group.items())
    random.shuffle(groups)

    total = sum(len(g[1]) for g in groups)
    target_train = int(total * train_ratio)

    # -------- Split groups (provenance-aware) --------
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

    # -------- Summary stats --------
    train_count = sum(1 for r in final_rows if r["split"] == "train")
    val_count = sum(1 for r in final_rows if r["split"] == "val")
    
    print(f"\nFinal split: {train_count} train / {val_count} val")
    
    # Per-string breakdown
    print("\nPer-string distribution:")
    for split in ["train", "val"]:
        print(f"  {split}:")
        string_counts = defaultdict(int)
        ambiguous_counts = defaultdict(int)
        for r in final_rows:
            if r["split"] == split:
                string_counts[r["string"]] += 1
                if is_ambiguous(r["string"], r["fret"]):
                    ambiguous_counts[r["string"]] += 1
        for s in sorted(string_counts.keys()):
            print(f"    String {s}: {string_counts[s]} ({ambiguous_counts[s]} ambiguous)")

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


def main():
    p = argparse.ArgumentParser(
        description="Build string classification training mix with ambiguous position upweighting"
    )
    p.add_argument("--datasets", nargs="+", required=True,
                   help="Dataset root directories")
    p.add_argument("--max-per-string", type=int, default=None,
                   help="Max samples per string (default: no limit)")
    p.add_argument("--train-ratio", type=float, default=0.8,
                   help="Train/val split ratio (default: 0.8)")
    p.add_argument("--output", type=str, default="recipes/string_training_mix.csv",
                   help="Output CSV path")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--no-stratify-fret", action="store_true",
                   help="Disable fret stratification within each string")
    p.add_argument("--ambiguous-multiplier", type=float, default=1.0,
                   help="Oversample ambiguous positions by this factor (e.g., 2.0 = 2x samples)")

    args = p.parse_args()

    build_string_training_mix(
        dataset_roots=[Path(d).expanduser() for d in args.datasets],
        max_per_string=args.max_per_string,
        train_ratio=args.train_ratio,
        output_csv=Path(args.output).expanduser(),
        seed=args.seed,
        stratify_by_fret=not args.no_stratify_fret,
        ambiguous_multiplier=args.ambiguous_multiplier,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)