#!/usr/bin/env python3
"""
build_disambig_mix.py - Build training mixes for string disambiguation models.

Supports two modes:
- Triplet (3-way): Disambiguate between 3 adjacent strings (recommended)
- Pair (2-way): Disambiguate between 2 adjacent strings (legacy)

Usage:
    # Build ALL triplets (recommended)
    python build_disambig_mix.py --all-triplets --output-dir recipes/disambig/ --datasets data/...

    # Build ALL pairs (legacy)
    python build_disambig_mix.py --all-pairs --output-dir recipes/disambig/ --datasets data/...
"""

import argparse
import csv
import random
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from common.dataset import load_datasets

# String definitions
OPEN_MIDI = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}

STRING_TRIPLETS = {
    'E_A_D': (6, 5, 4),
    'A_D_G': (5, 4, 3),
    'D_G_B': (4, 3, 2),
    'G_B_e': (3, 2, 1),
}

STRING_PAIRS = {
    'E_A': (6, 5),
    'A_D': (5, 4),
    'D_G': (4, 3),
    'G_B': (3, 2),
    'B_e': (2, 1),
}


def midi_to_note_name(midi: int) -> str:
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return names[midi % 12] + str(midi // 12 - 1)


def get_triplet_ambiguities(s_a: int, s_b: int, s_c: int, max_fret: int = 12) -> Dict:
    """Get MIDI notes playable on all 3 strings."""
    range_a = set(range(OPEN_MIDI[s_a], OPEN_MIDI[s_a] + max_fret + 1))
    range_b = set(range(OPEN_MIDI[s_b], OPEN_MIDI[s_b] + max_fret + 1))
    range_c = set(range(OPEN_MIDI[s_c], OPEN_MIDI[s_c] + max_fret + 1))
    ambiguous = range_a & range_b & range_c
    return {midi: {'fret_a': midi - OPEN_MIDI[s_a], 'fret_b': midi - OPEN_MIDI[s_b], 
                   'fret_c': midi - OPEN_MIDI[s_c]} for midi in sorted(ambiguous)}


def get_pair_ambiguities(s_a: int, s_b: int, max_fret: int = 12) -> Dict:
    """Get MIDI notes playable on both strings."""
    range_a = set(range(OPEN_MIDI[s_a], OPEN_MIDI[s_a] + max_fret + 1))
    range_b = set(range(OPEN_MIDI[s_b], OPEN_MIDI[s_b] + max_fret + 1))
    ambiguous = range_a & range_b
    return {midi: {'fret_a': midi - OPEN_MIDI[s_a], 'fret_b': midi - OPEN_MIDI[s_b]} 
            for midi in sorted(ambiguous)}


def find_datasets(data_dir: Path) -> List[Path]:
    datasets = []
    if not data_dir.exists():
        return datasets
    for item in sorted(data_dir.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            if (item / "metadata.csv").exists():
                datasets.append(item)
    return datasets


def list_datasets(data_dir: Path) -> None:
    datasets = find_datasets(data_dir)
    if not datasets:
        print(f"No datasets found in {data_dir}")
        return
    print(f"Available datasets in {data_dir}:\n")
    for ds in datasets:
        try:
            with open(ds / "metadata.csv", 'r') as f:
                rows = list(csv.DictReader(f))
            string_counts = defaultdict(int)
            for row in rows:
                try:
                    string_counts[int(row['string'])] += 1
                except:
                    pass
            string_str = ", ".join([f"S{s}:{c}" for s, c in sorted(string_counts.items())])
            print(f"  {ds.name}: {len(rows)} samples ({string_str})")
        except Exception as e:
            print(f"  {ds.name}: Error - {e}")
    print(f"\nTotal: {len(datasets)} datasets")


def build_triplet_mix(dataset_roots, string_a, string_b, string_c, output_csv, 
                      max_fret=12, val_split=0.15, seed=42):
    """Build 3-way disambiguation mix."""
    random.seed(seed)
    ambiguous = get_triplet_ambiguities(string_a, string_b, string_c, max_fret)
    
    print(f"Triplet: S{string_a} vs S{string_b} vs S{string_c}")
    print(f"Ambiguous MIDI notes: {list(ambiguous.keys())}")
    
    all_rows = load_datasets(dataset_roots, require_string=True)
    if not all_rows:
        raise SystemExit("No samples loaded")
    print(f"Loaded {len(all_rows)} total samples")
    
    # Group by (midi, string)
    rows_by_ms = defaultdict(list)
    for row in all_rows:
        try:
            midi, string = int(row.get('midi', 0)), int(row.get('string', 0))
        except:
            continue
        if midi in ambiguous and string in (string_a, string_b, string_c):
            rows_by_ms[(midi, string)].append(row)
    
    # Balance across all 3 strings per MIDI
    balanced = []
    for midi in sorted(ambiguous.keys()):
        rows_a = rows_by_ms.get((midi, string_a), [])
        rows_b = rows_by_ms.get((midi, string_b), [])
        rows_c = rows_by_ms.get((midi, string_c), [])
        n = min(len(rows_a), len(rows_b), len(rows_c))
        if n == 0:
            print(f"  WARNING: MIDI {midi} missing samples on some strings")
            continue
        random.shuffle(rows_a); random.shuffle(rows_b); random.shuffle(rows_c)
        balanced.extend(rows_a[:n] + rows_b[:n] + rows_c[:n])
        print(f"  MIDI {midi}: {n} per string (avail: {len(rows_a)}/{len(rows_b)}/{len(rows_c)})")
    
    if not balanced:
        raise SystemExit("No balanced samples")
    
    random.shuffle(balanced)
    n_val = int(len(balanced) * val_split)
    
    for i, row in enumerate(balanced):
        row['split'] = 'val' if i < n_val else 'train'
        midi, string = int(row['midi']), int(row['string'])
        info = ambiguous[midi]
        row['target'] = 0 if string == string_a else (1 if string == string_b else 2)
        row['fret_a'], row['fret_b'], row['fret_c'] = info['fret_a'], info['fret_b'], info['fret_c']
        row['string_a'], row['string_b'], row['string_c'] = string_a, string_b, string_c
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['abs_path', 'dataset_root', 'filename', 'string', 'fret', 'midi', 'note',
                  'take', 'sample_rate', 'split', 'target', 'fret_a', 'fret_b', 'fret_c',
                  'string_a', 'string_b', 'string_c']
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(balanced)
    
    print(f"\nWrote {len(balanced)} samples ({len(balanced)-n_val} train, {n_val} val)")
    return {'total': len(balanced), 'train': len(balanced)-n_val, 'val': n_val}


def build_pair_mix(dataset_roots, string_a, string_b, output_csv, 
                   max_fret=12, val_split=0.15, seed=42):
    """Build 2-way disambiguation mix."""
    random.seed(seed)
    ambiguous = get_pair_ambiguities(string_a, string_b, max_fret)
    
    print(f"Pair: S{string_a} vs S{string_b}")
    print(f"Ambiguous MIDI notes: {list(ambiguous.keys())}")
    
    all_rows = load_datasets(dataset_roots, require_string=True)
    if not all_rows:
        raise SystemExit("No samples loaded")
    print(f"Loaded {len(all_rows)} total samples")
    
    rows_by_ms = defaultdict(list)
    for row in all_rows:
        try:
            midi, string = int(row.get('midi', 0)), int(row.get('string', 0))
        except:
            continue
        if midi in ambiguous and string in (string_a, string_b):
            rows_by_ms[(midi, string)].append(row)
    
    balanced = []
    for midi in sorted(ambiguous.keys()):
        rows_a = rows_by_ms.get((midi, string_a), [])
        rows_b = rows_by_ms.get((midi, string_b), [])
        n = min(len(rows_a), len(rows_b))
        if n == 0:
            print(f"  WARNING: MIDI {midi} missing samples")
            continue
        random.shuffle(rows_a); random.shuffle(rows_b)
        balanced.extend(rows_a[:n] + rows_b[:n])
        print(f"  MIDI {midi}: {n} per string (avail: {len(rows_a)}/{len(rows_b)})")
    
    if not balanced:
        raise SystemExit("No balanced samples")
    
    random.shuffle(balanced)
    n_val = int(len(balanced) * val_split)
    
    for i, row in enumerate(balanced):
        row['split'] = 'val' if i < n_val else 'train'
        midi, string = int(row['midi']), int(row['string'])
        info = ambiguous[midi]
        row['target'] = 0 if string == string_a else 1
        row['fret_a'], row['fret_b'] = info['fret_a'], info['fret_b']
        row['string_a'], row['string_b'] = string_a, string_b
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['abs_path', 'dataset_root', 'filename', 'string', 'fret', 'midi', 'note',
                  'take', 'sample_rate', 'split', 'target', 'fret_a', 'fret_b', 'string_a', 'string_b']
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(balanced)
    
    print(f"\nWrote {len(balanced)} samples ({len(balanced)-n_val} train, {n_val} val)")
    return {'total': len(balanced), 'train': len(balanced)-n_val, 'val': n_val}


def main():
    parser = argparse.ArgumentParser(description="Build disambiguation training mixes")
    parser.add_argument('--list-datasets', action='store_true')
    
    # Triplet mode
    parser.add_argument('--triplet', choices=list(STRING_TRIPLETS.keys()))
    parser.add_argument('--all-triplets', action='store_true')
    
    # Pair mode
    parser.add_argument('--pair', choices=list(STRING_PAIRS.keys()))
    parser.add_argument('--all-pairs', action='store_true')
    
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--output', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--max-fret', type=int, default=12)
    parser.add_argument('--val-split', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if args.list_datasets:
        list_datasets(REPO_ROOT / "data")
        return
    
    if not args.datasets:
        parser.error("--datasets is required")
    
    dataset_roots = []
    for ds in args.datasets:
        ds_path = Path(ds) if Path(ds).is_absolute() else REPO_ROOT / ds
        if ds_path.exists():
            dataset_roots.append(ds_path)
        else:
            print(f"WARNING: {ds_path} not found")
    
    if not dataset_roots:
        parser.error("No valid datasets")
    
    # Batch triplet mode
    if args.all_triplets:
        if not args.output_dir:
            parser.error("--output-dir required with --all-triplets")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print("="*60 + "\nBuilding ALL triplet mixes\n" + "="*60)
        for name, (sa, sb, sc) in STRING_TRIPLETS.items():
            print(f"\n--- {name} ---")
            try:
                build_triplet_mix(dataset_roots, sa, sb, sc, out_dir / f"disambig_{name.lower()}_mix.csv",
                                  args.max_fret, args.val_split, args.seed)
            except Exception as e:
                print(f"ERROR: {e}")
        return
    
    # Batch pair mode
    if args.all_pairs:
        if not args.output_dir:
            parser.error("--output-dir required with --all-pairs")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print("="*60 + "\nBuilding ALL pair mixes\n" + "="*60)
        for name, (sa, sb) in STRING_PAIRS.items():
            print(f"\n--- {name} ---")
            try:
                build_pair_mix(dataset_roots, sa, sb, out_dir / f"disambig_{name.lower()}_mix.csv",
                               args.max_fret, args.val_split, args.seed)
            except Exception as e:
                print(f"ERROR: {e}")
        return
    
    # Single triplet
    if args.triplet:
        if not args.output:
            parser.error("--output required with --triplet")
        sa, sb, sc = STRING_TRIPLETS[args.triplet]
        build_triplet_mix(dataset_roots, sa, sb, sc, Path(args.output), args.max_fret, args.val_split, args.seed)
        return
    
    # Single pair
    if args.pair:
        if not args.output:
            parser.error("--output required with --pair")
        sa, sb = STRING_PAIRS[args.pair]
        build_pair_mix(dataset_roots, sa, sb, Path(args.output), args.max_fret, args.val_split, args.seed)
        return
    
    parser.error("Specify --triplet, --all-triplets, --pair, or --all-pairs")


if __name__ == '__main__':
    main()
