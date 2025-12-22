#!/usr/bin/env python3
"""
build_training_mix.py - Unified training mix builder for note and string models.

Usage:
  # Note classification (balance by position)
  python build_training_mix.py \\
      --mode note \\
      --datasets data/strat_119 data/tele_10 \\
      --max-per-class 500 \\
      --output recipes/note_mix.csv

  # String classification (balance by string with fret stratification)
  python build_training_mix.py \\
      --mode string \\
      --datasets data/strat_119 \\
      --max-per-string 1500 \\
      --stratify-by-fret \\
      --ambiguous-multiplier 2.0 \\
      --output recipes/string_mix.csv
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for common imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.dataset import (
    load_datasets,
    balance_by_class,
    balance_by_string_with_fret_stratification,
    write_training_mix_csv,
)
from common.guitar import is_ambiguous_position


def build_note_training_mix(
    dataset_roots: list,
    max_per_class: int = None,
    train_ratio: float = 0.8,
    output_csv: Path = None,
    seed: int = 42,
) -> None:
    """Build training mix for note/position classification."""
    print("=" * 60)
    print("Building NOTE training mix")
    print("=" * 60)
    
    # Load datasets
    all_rows = load_datasets(dataset_roots, require_string=False)
    if not all_rows:
        raise SystemExit("No valid samples loaded.")
    
    print(f"Loaded {len(all_rows)} total samples")
    
    # Balance by position class
    selected = balance_by_class(
        all_rows,
        class_key="pos",
        max_per_class=max_per_class,
        seed=seed
    )
    
    print(f"Selected {len(selected)} samples after balancing")
    
    # Write CSV with train/val split
    write_training_mix_csv(selected, output_csv, train_ratio=train_ratio, seed=seed)


def build_string_training_mix(
    dataset_roots: list,
    max_per_string: int = None,
    train_ratio: float = 0.8,
    output_csv: Path = None,
    seed: int = 42,
    stratify_by_fret: bool = True,
    ambiguous_multiplier: float = 1.0,
) -> None:
    """Build training mix for string classification."""
    print("=" * 60)
    print("Building STRING training mix")
    print("=" * 60)
    
    # Load datasets (require string field)
    all_rows = load_datasets(dataset_roots, require_string=True)
    if not all_rows:
        raise SystemExit("No valid samples loaded.")
    
    print(f"Loaded {len(all_rows)} total samples")
    
    # Balance by string with optional fret stratification
    selected = balance_by_string_with_fret_stratification(
        all_rows,
        max_per_string=max_per_string,
        stratify_by_fret=stratify_by_fret,
        ambiguous_multiplier=ambiguous_multiplier,
        seed=seed
    )
    
    print(f"Selected {len(selected)} samples after balancing")
    
    # Count ambiguous samples
    train_ambiguous = sum(1 for r in selected 
                         if is_ambiguous_position(r["string"], r["fret"]))
    print(f"Ambiguous samples: {train_ambiguous} ({100*train_ambiguous/len(selected):.1f}%)")
    
    # Write CSV with train/val split
    write_training_mix_csv(selected, output_csv, train_ratio=train_ratio, seed=seed)


def main():
    parser = argparse.ArgumentParser(
        description="Build training mix for note or string classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Note classification
  python build_training_mix.py --mode note --datasets data/strat_119 --output recipes/note_mix.csv

  # String classification with ambiguous upweighting
  python build_training_mix.py --mode string --datasets data/strat_119 --ambiguous-multiplier 2.0
"""
    )
    
    # Required arguments
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["note", "string"],
        help="Training mode: 'note' for position classification, 'string' for string classification"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset root directories"
    )
    
    # Common arguments
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train/val split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: recipes/{mode}_training_mix.csv)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    # Note mode arguments
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="[note mode] Max samples per position class"
    )
    
    # String mode arguments
    parser.add_argument(
        "--max-per-string",
        type=int,
        default=None,
        help="[string mode] Max samples per string"
    )
    parser.add_argument(
        "--stratify-by-fret",
        action="store_true",
        default=True,
        help="[string mode] Stratify samples by fret within each string (default: True)"
    )
    parser.add_argument(
        "--no-stratify-by-fret",
        action="store_true",
        help="[string mode] Disable fret stratification"
    )
    parser.add_argument(
        "--ambiguous-multiplier",
        type=float,
        default=1.0,
        help="[string mode] Oversample ambiguous positions by this factor (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output is None:
        if args.mode == "note":
            args.output = "recipes/note_training_mix.csv"
        else:
            args.output = "recipes/string_training_mix.csv"
    
    output_csv = Path(args.output).expanduser().resolve()
    dataset_roots = [Path(d).expanduser().resolve() for d in args.datasets]
    
    # Handle stratify flag
    stratify_by_fret = args.stratify_by_fret and not args.no_stratify_by_fret
    
    # Build appropriate mix
    if args.mode == "note":
        build_note_training_mix(
            dataset_roots=dataset_roots,
            max_per_class=args.max_per_class,
            train_ratio=args.train_ratio,
            output_csv=output_csv,
            seed=args.seed,
        )
    else:  # string mode
        build_string_training_mix(
            dataset_roots=dataset_roots,
            max_per_string=args.max_per_string,
            train_ratio=args.train_ratio,
            output_csv=output_csv,
            seed=args.seed,
            stratify_by_fret=stratify_by_fret,
            ambiguous_multiplier=args.ambiguous_multiplier,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
