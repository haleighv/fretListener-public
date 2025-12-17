#!/usr/bin/env python

import argparse
import sys
from pathlib import Path
import csv
import random

"""
Example usage:

Use all data from both datasets (no per-class cap):

    cd ~/code/fretListener

    python tools/build_training_mix.py \
      --datasets data/strat_neck_40 data/strat_neck_119 \
      --train-ratio 0.8 \
      --seed 42


    python tools/build_training_mix.py \
      --datasets data/strat_neck_40 data/strat_neck_119 data/rg550_neck_20 \
      --train-ratio 0.8 \
      --seed 42 \
      --output recipes/training_mix_full_mine.csv
"""


def prompt_path_list(prompt: str) -> list[Path]:
    print(prompt)
    print("Enter one dataset root per line (blank line to finish).")
    roots: list[Path] = []
    while True:
        value = input("> ").strip()
        if not value:
            break
        p = Path(value).expanduser()
        if not p.exists():
            print(f"  Path does not exist: {p}")
            continue
        roots.append(p)
    return roots


def prompt_int(prompt: str, default: int | None = None) -> int | None:
    """
    Prompt for an integer. If default is not None and user enters blank,
    return default. If default is None and user enters blank, return None.
    """
    while True:
        value = input(prompt).strip()
        if not value:
            # Allow "no cap" semantics when default is None
            return default
        try:
            return int(value)
        except ValueError:
            print("Please enter an integer.")


def prompt_float(prompt: str, default: float | None = None) -> float:
    while True:
        value = input(prompt).strip()
        if not value and default is not None:
            return default
        try:
            return float(value)
        except ValueError:
            print("Please enter a number.")


def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        value = input(prompt + suffix).strip().lower()
        if not value:
            return default
        if value in ("y", "yes"):
            return True
        if value in ("n", "no"):
            return False
        print("Please answer y or n.")


def load_metadata_csv(meta_csv: Path) -> list[dict]:
    rows: list[dict] = []
    with meta_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_training_mix(
    dataset_roots: list[Path],
    max_per_class: int | None,
    train_ratio: float,
    output_csv: Path,
    seed: int = 42,
) -> None:
    if not dataset_roots:
        raise SystemExit("No dataset roots provided.")

    random.seed(seed)

    # Load all samples, grouped by dataset
    datasets: dict[str, list[dict]] = {}  # dataset_root -> list of rows

    for root in dataset_roots:
        root = root.resolve()
        meta_csv = root / "meta" / "metadata.csv"
        if not meta_csv.is_file():
            print(f"Warning: metadata.csv not found for {root}; skipping.")
            continue

        print(f"Loading metadata from: {meta_csv}")
        rows = load_metadata_csv(meta_csv)
        if not rows:
            print(f"Warning: no rows in {meta_csv}; skipping.")
            continue

        dataset_rows: list[dict] = []
        for row in rows:
            row_copy = dict(row)

            # Normalize id (if present)
            if "id" in row_copy and row_copy["id"] != "":
                try:
                    row_copy["id"] = int(row_copy["id"])
                except Exception:
                    pass

            # Normalize pos (required)
            try:
                row_copy["pos"] = int(row_copy["pos"])
            except Exception:
                print(f"Warning: invalid pos value {row_copy.get('pos')}; skipping row.")
                continue

            # Normalize take (if present)
            if "take" in row_copy and row_copy["take"] != "":
                try:
                    row_copy["take"] = int(row_copy["take"])
                except Exception:
                    row_copy["take"] = -1
            else:
                row_copy["take"] = -1

            # Dataset root name, etc.
            row_copy["dataset_root"] = str(root)

            # abs_path to audio file
            relpath = row_copy.get("relpath") or row_copy.get("filename")
            if not relpath:
                print(f"Warning: missing relpath/filename for row: {row}")
                continue
            abs_path = (root / relpath).resolve()
            row_copy["abs_path"] = str(abs_path)

            dataset_rows.append(row_copy)

        if dataset_rows:
            datasets[str(root)] = dataset_rows

    if not datasets:
        raise SystemExit("No samples loaded from any dataset; nothing to do.")

    total_samples = sum(len(rows) for rows in datasets.values())
    print(f"Total samples loaded from all datasets: {total_samples}")

    # === LEAKAGE PREVENTION ===
    # Split at the DATASET level, not the sample level.
    # This ensures all samples from a recording session stay together,
    # preventing the model from learning session-specific characteristics.
    #
    # Strategy: Assign entire datasets to train or val, trying to approximate
    # the target train_ratio while ensuring val has at least one dataset.

    dataset_names = list(datasets.keys())
    random.shuffle(dataset_names)

    # Calculate cumulative sample counts to find best split point
    if len(dataset_names) == 1:
        # Only one dataset: fall back to within-dataset splitting by take/group
        # but warn the user about potential leakage
        print(
            "Warning: Only one dataset provided. Splitting within dataset by sample, "
            "which may cause leakage if samples share recording conditions."
        )
        train_datasets = dataset_names
        val_datasets = []
        single_dataset_mode = True
    else:
        single_dataset_mode = False
        # Find split point that best approximates train_ratio
        dataset_sizes = [len(datasets[name]) for name in dataset_names]
        cumsum = []
        running = 0
        for size in dataset_sizes:
            running += size
            cumsum.append(running)

        target_train_samples = int(total_samples * train_ratio)

        # Find best split: at least 1 dataset for val
        best_split_idx = 0
        best_diff = float("inf")
        for i in range(len(dataset_names) - 1):  # Leave at least 1 for val
            diff = abs(cumsum[i] - target_train_samples)
            if diff < best_diff:
                best_diff = diff
                best_split_idx = i

        train_datasets = dataset_names[: best_split_idx + 1]
        val_datasets = dataset_names[best_split_idx + 1 :]

        train_count = sum(len(datasets[d]) for d in train_datasets)
        val_count = sum(len(datasets[d]) for d in val_datasets)
        actual_ratio = train_count / total_samples if total_samples > 0 else 0

        print(f"\nDataset-level split to prevent leakage:")
        print(f"  Train datasets ({len(train_datasets)}):")
        for d in train_datasets:
            print(f"    - {Path(d).name}: {len(datasets[d])} samples")
        print(f"  Val datasets ({len(val_datasets)}):")
        for d in val_datasets:
            print(f"    - {Path(d).name}: {len(datasets[d])} samples")
        print(f"  Actual train ratio: {actual_ratio:.2%} (target: {train_ratio:.0%})")

    # Combine all samples with dataset info
    combined: list[dict] = []
    for dataset_root, rows in datasets.items():
        for row in rows:
            combined.append(row)

    # Group by position class (pos)
    by_pos: dict[int, list[dict]] = {}
    for row in combined:
        pos = row["pos"]
        by_pos.setdefault(pos, []).append(row)

    # Build balanced subset with optional per-class cap
    # When capping, try to maintain dataset diversity within each class
    selected: list[dict] = []
    for pos, rows in sorted(by_pos.items(), key=lambda kv: kv[0]):
        if max_per_class is not None and len(rows) > max_per_class:
            # Group by dataset first, then sample proportionally from each
            by_dataset: dict[str, list[dict]] = {}
            for r in rows:
                by_dataset.setdefault(r["dataset_root"], []).append(r)

            # Shuffle within each dataset
            for ds_rows in by_dataset.values():
                random.shuffle(ds_rows)

            # Proportionally sample from each dataset
            rows_copy: list[dict] = []
            dataset_keys = list(by_dataset.keys())
            random.shuffle(dataset_keys)

            # Round-robin selection to maintain diversity
            indices = {k: 0 for k in dataset_keys}
            while len(rows_copy) < max_per_class:
                added_any = False
                for k in dataset_keys:
                    if indices[k] < len(by_dataset[k]) and len(rows_copy) < max_per_class:
                        rows_copy.append(by_dataset[k][indices[k]])
                        indices[k] += 1
                        added_any = True
                if not added_any:
                    break
        else:
            rows_copy = rows[:]
            random.shuffle(rows_copy)

        selected.extend(rows_copy)
        datasets_in_class = len({r["dataset_root"] for r in rows_copy})
        print(
            f"Class pos={pos}: using {len(rows_copy)} / {len(rows)} samples "
            f"from {datasets_in_class} dataset(s)"
        )

    if not selected:
        raise SystemExit("No samples selected for training mix; nothing to write.")

    # Assign splits based on dataset membership
    final_rows: list[dict] = []

    if single_dataset_mode:
        # Fallback: split within the single dataset, but group by take if available
        by_pos_selected: dict[int, list[dict]] = {}
        for row in selected:
            by_pos_selected.setdefault(row["pos"], []).append(row)

        for pos, rows in sorted(by_pos_selected.items(), key=lambda kv: kv[0]):
            # Try to group by take to reduce leakage
            by_take: dict[int, list[dict]] = {}
            for r in rows:
                take = r.get("take", -1)
                by_take.setdefault(take, []).append(r)

            take_keys = list(by_take.keys())
            random.shuffle(take_keys)

            if len(take_keys) > 1:
                # Split by take groups
                n_takes = len(take_keys)
                n_train_takes = max(1, int(round(n_takes * train_ratio)))
                n_train_takes = min(n_train_takes, n_takes - 1)  # Keep at least 1 for val

                train_takes = set(take_keys[:n_train_takes])
                for r in rows:
                    r_copy = dict(r)
                    r_copy["split"] = "train" if r["take"] in train_takes else "val"
                    final_rows.append(r_copy)
            else:
                # No take info or single take: random split (with leakage warning already shown)
                random.shuffle(rows)
                n = len(rows)
                n_train = max(1, min(int(round(n * train_ratio)), n - 1)) if n > 1 else 1
                for i, r in enumerate(rows):
                    r_copy = dict(r)
                    r_copy["split"] = "train" if i < n_train else "val"
                    final_rows.append(r_copy)
    else:
        # Multi-dataset mode: assign based on dataset membership
        train_set = set(train_datasets)
        for row in selected:
            row_copy = dict(row)
            row_copy["split"] = "train" if row["dataset_root"] in train_set else "val"
            final_rows.append(row_copy)

    # Report final split statistics
    train_final = [r for r in final_rows if r["split"] == "train"]
    val_final = [r for r in final_rows if r["split"] == "val"]
    print(f"\nFinal training mix: {len(train_final)} train, {len(val_final)} val")

    # Check class coverage in both splits
    train_classes = set(r["pos"] for r in train_final)
    val_classes = set(r["pos"] for r in val_final)
    all_classes = train_classes | val_classes

    missing_in_train = all_classes - train_classes
    missing_in_val = all_classes - val_classes

    if missing_in_train:
        print(f"Warning: {len(missing_in_train)} classes missing from train split: {sorted(missing_in_train)}")
    if missing_in_val:
        print(f"Warning: {len(missing_in_val)} classes missing from val split: {sorted(missing_in_val)}")

    output_csv = output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        # From metadata
        "id",
        "dataset_name",
        "dataset_root",
        "relpath",
        "abs_path",
        "filename",
        "string",
        "fret",
        "pos",
        "midi",
        "note",
        "take",
        "sample_rate",
        "num_samples",
        "source_relpath",
        # New for mix
        "split",
    ]

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in final_rows:
            out_row = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(out_row)

    print(f"Training mix written to: {output_csv}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a custom training mix from one or more datasets "
            "that already have audio/ and meta/metadata.csv."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help=(
            "Dataset roots (e.g. data/strat_neck_40 data/strat_bridge_40). "
            "If omitted, script will prompt interactively."
        ),
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        help=(
            "Optional cap on samples per position class (pos). "
            "If omitted and --datasets is provided, all available samples "
            "for each class are used (no cap)."
        ),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (0.0-1.0). Default: 0.8.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="recipes/training_mix.csv",
        help="Path to output training mix CSV. Default: recipes/training_mix.csv",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling. Default: 42.",
    )
    args = parser.parse_args()

    # Dataset roots: CLI or interactive
    if args.datasets:
        dataset_roots = [Path(p).expanduser() for p in args.datasets]
    else:
        dataset_roots = prompt_path_list(
            "Enter dataset root directories that contain audio/ and meta/:"
        )

    if not dataset_roots:
        raise SystemExit("No dataset roots provided.")

    for r in dataset_roots:
        if not r.exists():
            raise SystemExit(f"Dataset root does not exist: {r}")

    # Max per class semantics:
    # - If CLI provided --max-per-class, use that.
    # - Else if CLI provided --datasets, default to "no cap" (None) and do NOT prompt.
    # - Else (fully interactive run), prompt the user.
    if args.max_per_class is not None:
        max_per_class = args.max_per_class
    else:
        if args.datasets:
            max_per_class = None  # no cap, non-interactive
        else:
            max_per_class = prompt_int(
                "Max samples per class (pos)? [blank for no cap]: ",
                default=None,
            )

    if args.train_ratio is not None:
        train_ratio = args.train_ratio
    else:
        train_ratio = prompt_float(
            "Train ratio (e.g. 0.8): ",
            default=0.8,
        )

    output_csv = Path(args.output).expanduser()

    print("Datasets to include:")
    for r in dataset_roots:
        print("  -", r)
    print(f"Max per class: {max_per_class if max_per_class is not None else 'no cap'}")
    print(f"Train ratio:   {train_ratio}")
    print(f"Output CSV:    {output_csv}")

    if not prompt_yes_no("Proceed with building training mix?", default=True):
        raise SystemExit("Aborted by user.")

    build_training_mix(
        dataset_roots=dataset_roots,
        max_per_class=max_per_class,
        train_ratio=train_ratio,
        output_csv=output_csv,
        seed=args.seed,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)