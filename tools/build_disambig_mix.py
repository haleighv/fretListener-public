#!/usr/bin/env python3
"""
build_disambig_mix.py - Build training mix for string disambiguation models.

Creates training CSVs for disambiguating between two adjacent strings.
Only includes samples for notes that are ambiguous between the two strings.

Usage:
    # Build single pair mix
    python build_disambig_mix.py \\
        --pair A_D \\
        --datasets data/strat_neck_119 data/rg550_neck_20 \\
        --output recipes/disambig_a_d_mix.csv

    # Build ALL pairs at once (batch mode)
    python build_disambig_mix.py \\
        --all-pairs \\
        --datasets data/strat_neck_119 data/rg550_neck_20 \\
        --output-dir recipes/disambig/
        
    # List available datasets
    python build_disambig_mix.py --list-datasets
"""

import argparse
import csv
import random
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Add parent directory to path for common imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from common.dataset import load_datasets, get_group_key


# =============================================================================
# STRING PAIR DEFINITIONS
# =============================================================================

# String pair definitions: name -> (string_num_a, string_num_b)
# String numbers are 1-indexed: 1=high e, 6=low E
STRING_PAIRS = {
    'E_A': (6, 5),  # low E vs A
    'A_D': (5, 4),  # A vs D
    'D_G': (4, 3),  # D vs G
    'G_B': (3, 2),  # G vs B
    'B_e': (2, 1),  # B vs high e
}


# =============================================================================
# AMBIGUITY DETECTION
# =============================================================================

def get_ambiguous_midi_notes(string_a: int, string_b: int, max_fret: int = 12) -> Dict[int, Dict]:
    """
    Get MIDI notes that are ambiguous between two strings.
    
    Returns dict: midi -> {
        'fret_a': int,
        'fret_b': int,
    }
    """
    # Import here to avoid circular dependency
    open_midi = {
        1: 64,  # high e
        2: 59,  # B
        3: 55,  # G
        4: 50,  # D
        5: 45,  # A
        6: 40,  # low E
    }
    
    open_a = open_midi[string_a]
    open_b = open_midi[string_b]
    
    # MIDI range for each string (within max_fret)
    range_a = set(range(open_a, open_a + max_fret + 1))
    range_b = set(range(open_b, open_b + max_fret + 1))
    
    # Ambiguous = notes playable on both strings
    ambiguous = range_a & range_b
    
    result = {}
    for midi in sorted(ambiguous):
        fret_a = midi - open_a
        fret_b = midi - open_b
        result[midi] = {
            'fret_a': fret_a,
            'fret_b': fret_b,
        }
    
    return result


def midi_to_note_name(midi: int) -> str:
    """Convert MIDI note number to note name."""
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return names[midi % 12] + str(midi // 12 - 1)


# =============================================================================
# DATASET DISCOVERY
# =============================================================================

def find_datasets(data_dir: Path) -> List[Path]:
    """Find all dataset directories under data_dir."""
    datasets = []
    if not data_dir.exists():
        return datasets
    
    for item in sorted(data_dir.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            # Check if it has a metadata CSV
            meta_csv = item / "metadata.csv"
            if meta_csv.exists():
                datasets.append(item)
    
    return datasets


def list_datasets(data_dir: Path) -> None:
    """List available datasets with sample counts."""
    datasets = find_datasets(data_dir)
    
    if not datasets:
        print(f"No datasets found in {data_dir}")
        return
    
    print(f"Available datasets in {data_dir}:\n")
    
    for ds in datasets:
        meta_csv = ds / "metadata.csv"
        try:
            with open(meta_csv, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
            # Count by string
            string_counts = defaultdict(int)
            for row in rows:
                try:
                    string_counts[int(row['string'])] += 1
                except:
                    pass
            
            string_str = ", ".join([f"S{s}:{c}" for s, c in sorted(string_counts.items())])
            print(f"  {ds.name}: {len(rows)} samples ({string_str})")
            
        except Exception as e:
            print(f"  {ds.name}: Error reading metadata - {e}")
    
    print(f"\nTotal: {len(datasets)} datasets")


# =============================================================================
# MIX BUILDING
# =============================================================================

def build_disambig_mix(
    dataset_roots: List[Path],
    string_a: int,
    string_b: int,
    output_csv: Path,
    max_fret: int = 12,
    val_split: float = 0.15,
    seed: int = 42,
) -> Dict:
    """
    Build disambiguation training mix from dataset directories.
    
    Args:
        dataset_roots: List of dataset root paths
        string_a: First string number (1-indexed)
        string_b: Second string number (1-indexed)
        output_csv: Output path for disambiguation mix
        max_fret: Maximum fret to include
        val_split: Fraction for validation
        seed: Random seed
    
    Returns:
        Stats dict
    """
    random.seed(seed)
    
    # Get ambiguous MIDI notes for this pair
    ambiguous = get_ambiguous_midi_notes(string_a, string_b, max_fret)
    
    print(f"String pair: S{string_a} vs S{string_b}")
    print(f"Ambiguous MIDI notes: {sorted(ambiguous.keys())}")
    for midi, info in sorted(ambiguous.items()):
        name = midi_to_note_name(midi)
        print(f"  MIDI {midi} ({name}): S{string_a}F{info['fret_a']} vs S{string_b}F{info['fret_b']}")
    
    # Load datasets using common module
    print(f"\nLoading datasets...")
    all_rows = load_datasets(dataset_roots, require_string=True)
    
    if not all_rows:
        raise SystemExit("No valid samples loaded.")
    
    print(f"Loaded {len(all_rows)} total samples")
    
    # Filter to ambiguous MIDI notes on our target strings
    rows_by_midi_string = defaultdict(list)
    
    for row in all_rows:
        try:
            midi = int(row.get('midi', 0))
            string = int(row.get('string', 0))
        except (ValueError, TypeError):
            continue
        
        # Only keep rows for ambiguous MIDI notes on our target strings
        if midi in ambiguous and string in (string_a, string_b):
            rows_by_midi_string[(midi, string)].append(row)
    
    # Balance: for each ambiguous MIDI, take equal samples from each string
    balanced_rows = []
    stats = {
        'midi_counts': {},
        'string_a_total': 0,
        'string_b_total': 0,
    }
    
    for midi in sorted(ambiguous.keys()):
        rows_a = rows_by_midi_string.get((midi, string_a), [])
        rows_b = rows_by_midi_string.get((midi, string_b), [])
        
        # Shuffle before selecting
        random.shuffle(rows_a)
        random.shuffle(rows_b)
        
        # Take minimum of the two to balance
        n = min(len(rows_a), len(rows_b))
        
        if n == 0:
            print(f"  WARNING: No samples for MIDI {midi} on one or both strings!")
            continue
        
        # Add to output
        selected_a = rows_a[:n]
        selected_b = rows_b[:n]
        
        balanced_rows.extend(selected_a)
        balanced_rows.extend(selected_b)
        
        stats['midi_counts'][midi] = {
            'available_a': len(rows_a),
            'available_b': len(rows_b),
            'used': n,
        }
        stats['string_a_total'] += n
        stats['string_b_total'] += n
        
        print(f"  MIDI {midi}: {n} samples per string (available: {len(rows_a)}/{len(rows_b)})")
    
    if not balanced_rows:
        raise SystemExit("No samples found for this string pair!")
    
    # Shuffle and split train/val
    random.shuffle(balanced_rows)
    
    n_val = int(len(balanced_rows) * val_split)
    n_train = len(balanced_rows) - n_val
    
    # Assign splits
    for i, row in enumerate(balanced_rows):
        row['split'] = 'val' if i < n_val else 'train'
    
    # Add disambiguation-specific columns
    for row in balanced_rows:
        midi = int(row['midi'])
        string = int(row['string'])
        info = ambiguous[midi]
        
        # Target: 0 = string_a, 1 = string_b
        row['target'] = 0 if string == string_a else 1
        row['fret_a'] = info['fret_a']
        row['fret_b'] = info['fret_b']
        row['string_a'] = string_a
        row['string_b'] = string_b
    
    # Write output
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        'abs_path', 'dataset_root', 'filename', 'string', 'fret', 'midi', 'note',
        'take', 'sample_rate', 'split', 'target', 'fret_a', 'fret_b', 'string_a', 'string_b'
    ]
    
    print(f"\nWriting {output_csv}...")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(balanced_rows)
    
    # Summary
    stats['total'] = len(balanced_rows)
    stats['train'] = n_train
    stats['val'] = n_val
    
    print(f"\nSummary:")
    print(f"  Total samples: {stats['total']}")
    print(f"  Train: {stats['train']}")
    print(f"  Val: {stats['val']}")
    print(f"  String {string_a} samples: {stats['string_a_total']}")
    print(f"  String {string_b} samples: {stats['string_b_total']}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Build disambiguation training mix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
String pairs:
  E_A  - Low E (string 6) vs A (string 5)
  A_D  - A (string 5) vs D (string 4)
  D_G  - D (string 4) vs G (string 3)
  G_B  - G (string 3) vs B (string 2)
  B_e  - B (string 2) vs high e (string 1)

Examples:
  # List available datasets
  python build_disambig_mix.py --list-datasets
  
  # Build single pair mix
  python build_disambig_mix.py --pair A_D \\
      --datasets data/strat_neck_119 data/rg550_neck_20 \\
      --output recipes/disambig_a_d_mix.csv
      
  # Build ALL pairs at once
  python build_disambig_mix.py --all-pairs \\
      --datasets data/strat_neck_119 data/rg550_neck_20 \\
      --output-dir recipes/disambig/
"""
    )
    
    parser.add_argument('--list-datasets', action='store_true',
                       help='List available datasets and exit')
    parser.add_argument('--pair', type=str, choices=list(STRING_PAIRS.keys()),
                       help='String pair to build mix for (single pair mode)')
    parser.add_argument('--all-pairs', action='store_true',
                       help='Build mixes for ALL string pairs (batch mode)')
    parser.add_argument('--datasets', type=str, nargs='+',
                       help='Dataset directories to include')
    parser.add_argument('--output', type=str,
                       help='Output CSV path (single pair mode)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for CSVs (batch mode with --all-pairs)')
    parser.add_argument('--max-fret', type=int, default=12,
                       help='Maximum fret to include (default: 12)')
    parser.add_argument('--val-split', type=float, default=0.15,
                       help='Validation split fraction (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Find data directory
    data_dir = REPO_ROOT / "data"
    
    if args.list_datasets:
        list_datasets(data_dir)
        return
    
    # Validate datasets arg
    if not args.datasets:
        parser.error("--datasets is required")
    
    # Resolve dataset paths
    dataset_roots = []
    for ds in args.datasets:
        ds_path = Path(ds)
        if not ds_path.is_absolute():
            ds_path = REPO_ROOT / ds
        if not ds_path.exists():
            print(f"WARNING: Dataset not found: {ds_path}")
            continue
        dataset_roots.append(ds_path)
    
    if not dataset_roots:
        parser.error("No valid datasets found")
    
    # Batch mode: build all pairs
    if args.all_pairs:
        if not args.output_dir:
            parser.error("--output-dir is required with --all-pairs")
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("BATCH MODE: Building mixes for ALL string pairs")
        print("=" * 60)
        
        results = {}
        for pair_name, (string_a, string_b) in STRING_PAIRS.items():
            print(f"\n{'='*60}")
            print(f"Building {pair_name} mix...")
            print("=" * 60)
            
            output_csv = output_dir / f"disambig_{pair_name.lower()}_mix.csv"
            
            try:
                stats = build_disambig_mix(
                    dataset_roots=dataset_roots,
                    string_a=string_a,
                    string_b=string_b,
                    output_csv=output_csv,
                    max_fret=args.max_fret,
                    val_split=args.val_split,
                    seed=args.seed,
                )
                results[pair_name] = {'status': 'success', 'stats': stats, 'path': output_csv}
            except Exception as e:
                print(f"  ERROR: {e}")
                results[pair_name] = {'status': 'error', 'error': str(e)}
        
        # Summary
        print("\n" + "=" * 60)
        print("BATCH SUMMARY")
        print("=" * 60)
        for pair_name, result in results.items():
            if result['status'] == 'success':
                stats = result['stats']
                print(f"  {pair_name}: {stats['total']} samples ({stats['train']} train, {stats['val']} val)")
            else:
                print(f"  {pair_name}: FAILED - {result['error']}")
        
        print(f"\nOutputs written to: {output_dir}")
        return
    
    # Single pair mode
    if not args.pair:
        parser.error("--pair is required (or use --all-pairs for batch mode)")
    if not args.output:
        parser.error("--output is required")
    
    string_a, string_b = STRING_PAIRS[args.pair]
    
    build_disambig_mix(
        dataset_roots=dataset_roots,
        string_a=string_a,
        string_b=string_b,
        output_csv=Path(args.output),
        max_fret=args.max_fret,
        val_split=args.val_split,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
