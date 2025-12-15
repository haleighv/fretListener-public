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

    combined: list[dict] = []

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

            combined.append(row_copy)

    if not combined:
        raise SystemExit("No samples loaded from any dataset; nothing to do.")

    print(f"Total samples loaded from all datasets: {len(combined)}")

    # Group by position class (pos)
    by_pos: dict[int, list[dict]] = {}
    for row in combined:
        pos = row["pos"]
        by_pos.setdefault(pos, []).append(row)

    # Build balanced subset with optional per-class cap
    selected: list[dict] = []
    for pos, rows in sorted(by_pos.items(), key=lambda kv: kv[0]):
        rows_copy = rows[:]
        random.shuffle(rows_copy)
        if max_per_class is not None:
            rows_copy = rows_copy[:max_per_class]
        selected.extend(rows_copy)
        print(
            f"Class pos={pos}: using {len(rows_copy)} / {len(rows)} samples "
            f"from {len({r.get('dataset_name', '') for r in rows})} dataset(s)"
        )

    if not selected:
        raise SystemExit("No samples selected for training mix; nothing to write.")

    # Train/val split per class to keep class balance in both
    final_rows: list[dict] = []
    by_pos_selected: dict[int, list[dict]] = {}
    for row in selected:
        by_pos_selected.setdefault(row["pos"], []).append(row)

    for pos, rows in sorted(by_pos_selected.items(), key=lambda kv: kv[0]):
        random.shuffle(rows)
        n = len(rows)
        if n == 1:
            n_train = 1
        else:
            n_train = int(round(n * train_ratio))
            n_train = max(1, min(n_train, n - 1))

        for i, row in enumerate(rows):
            split = "train" if i < n_train else "val"
            row_copy = dict(row)
            row_copy["split"] = split
            final_rows.append(row_copy)

    print(f"Final training mix size: {len(final_rows)} (train+val)")

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
