#!/usr/bin/env python3
"""
common/dataset.py - Dataset loading and training mix utilities.

Handles:
- Loading metadata CSVs from dataset directories
- Normalizing rows for training
- Provenance-aware train/val splitting
- Building training mixes for note and string models
"""

import csv
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .guitar import is_ambiguous_position


# =============================================================================
# CSV LOADING
# =============================================================================

def load_metadata_csv(meta_csv: Path) -> List[dict]:
    """Load a metadata CSV file and return list of row dicts."""
    with meta_csv.open("r", newline="") as f:
        return list(csv.DictReader(f))


def normalize_row(row: dict, dataset_root: Path, require_string: bool = False) -> Optional[dict]:
    """
    Normalize a metadata row for training.
    
    Args:
        row: Raw row dict from CSV
        dataset_root: Path to dataset root directory
        require_string: If True, require 'string' field (for string classification)
    
    Returns:
        Normalized row dict, or None if invalid
    """
    r = dict(row)
    
    # Parse position (required for note classification)
    try:
        r["pos"] = int(r.get("pos", 0))
    except (ValueError, TypeError):
        r["pos"] = 0
    
    # Parse string (required for string classification)
    try:
        r["string"] = int(r["string"])
    except (KeyError, ValueError, TypeError):
        if require_string:
            return None
        r["string"] = 0
    
    # Parse fret
    try:
        r["fret"] = int(r.get("fret", 0))
    except (ValueError, TypeError):
        r["fret"] = 0
    
    # Parse take
    try:
        r["take"] = int(r.get("take", -1))
    except (ValueError, TypeError):
        r["take"] = -1
    
    # Get file path
    relpath = r.get("relpath") or r.get("filename")
    if not relpath:
        return None
    
    r["dataset_root"] = str(dataset_root)
    r["abs_path"] = str((dataset_root / relpath).resolve())
    
    return r


def get_group_key(row: dict) -> str:
    """
    Get provenance key for a row.
    
    Used to ensure samples from the same recording session stay together
    during train/val splitting.
    """
    if row.get("source_relpath"):
        return f"src::{row['source_relpath']}"
    if row.get("dataset_root"):
        return f"ds::{row['dataset_root']}::{row.get('filename', '')}"
    return f"take::{row.get('take', -1)}"


# =============================================================================
# TRAIN/VAL SPLITTING
# =============================================================================

def split_by_provenance(
    rows: List[dict],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[dict], List[dict]]:
    """
    Split rows into train/val sets, keeping provenance groups together.
    
    Args:
        rows: List of row dicts
        train_ratio: Fraction of data for training
        seed: Random seed
    
    Returns:
        (train_rows, val_rows)
    """
    random.seed(seed)
    
    # Group by provenance
    by_group: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_group[get_group_key(r)].append(r)
    
    groups = list(by_group.items())
    random.shuffle(groups)
    
    total = sum(len(g[1]) for g in groups)
    target_train = int(total * train_ratio)
    
    train_groups = set()
    val_groups = set()
    
    running = 0
    for key, group_rows in groups:
        if running < target_train:
            train_groups.add(key)
            running += len(group_rows)
        else:
            val_groups.add(key)
    
    # Ensure at least one group in val
    if not val_groups and groups:
        key, _ = groups[-1]
        train_groups.discard(key)
        val_groups.add(key)
    
    train_rows = []
    val_rows = []
    
    for key, group_rows in by_group.items():
        if key in train_groups:
            train_rows.extend(group_rows)
        else:
            val_rows.extend(group_rows)
    
    return train_rows, val_rows


# =============================================================================
# TRAINING MIX BUILDING
# =============================================================================

def load_datasets(
    dataset_roots: List[Path],
    require_string: bool = False
) -> List[dict]:
    """
    Load and normalize rows from multiple dataset directories.
    
    Args:
        dataset_roots: List of dataset root paths
        require_string: If True, require 'string' field
    
    Returns:
        List of normalized row dicts
    """
    all_rows = []
    
    for root in dataset_roots:
        meta = root / "meta" / "metadata.csv"
        if not meta.exists():
            print(f"Skipping {root}: no metadata.csv")
            continue
        
        for row in load_metadata_csv(meta):
            norm = normalize_row(row, root, require_string=require_string)
            if norm:
                all_rows.append(norm)
    
    return all_rows


def balance_by_class(
    rows: List[dict],
    class_key: str,
    max_per_class: Optional[int] = None,
    seed: int = 42
) -> List[dict]:
    """
    Balance samples across classes with diversity preservation.
    
    Args:
        rows: List of row dicts
        class_key: Key to use for class grouping ('pos' or 'string')
        max_per_class: Maximum samples per class (None = no limit)
        seed: Random seed
    
    Returns:
        Balanced list of rows
    """
    random.seed(seed)
    
    # Group by class
    by_class: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        by_class[r[class_key]].append(r)
    
    selected = []
    
    for class_val, class_rows in sorted(by_class.items()):
        rows_copy = class_rows[:]
        random.shuffle(rows_copy)
        
        if max_per_class and len(rows_copy) > max_per_class:
            # Balance across provenance groups
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
        print(f"{class_key}={class_val}: using {len(rows_copy)} samples")
    
    return selected


def balance_by_string_with_fret_stratification(
    rows: List[dict],
    max_per_string: Optional[int] = None,
    stratify_by_fret: bool = True,
    ambiguous_multiplier: float = 1.0,
    seed: int = 42
) -> List[dict]:
    """
    Balance samples by string with optional fret stratification and ambiguous upweighting.
    
    Args:
        rows: List of row dicts
        max_per_string: Maximum samples per string (None = no limit)
        stratify_by_fret: If True, balance across frets within each string
        ambiguous_multiplier: Oversample ambiguous positions by this factor
        seed: Random seed
    
    Returns:
        Balanced list of rows
    """
    random.seed(seed)
    
    # Group by string
    by_string: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        by_string[r["string"]].append(r)
    
    print("Samples per string before balancing:")
    for s in sorted(by_string.keys()):
        print(f"  String {s}: {len(by_string[s])} samples")
    
    selected = []
    
    for string_num, string_rows in sorted(by_string.items()):
        rows_copy = string_rows[:]
        random.shuffle(rows_copy)
        
        target_count = max_per_string if max_per_string else len(rows_copy)
        
        if stratify_by_fret:
            # Group by fret within this string
            by_fret: Dict[int, List[dict]] = defaultdict(list)
            for r in rows_copy:
                by_fret[r["fret"]].append(r)
            
            for fret_rows in by_fret.values():
                random.shuffle(fret_rows)
            
            num_frets = len(by_fret)
            if num_frets == 0:
                continue
            
            # Calculate ambiguous frets for this string
            ambiguous_frets = {f for (s, f) in 
                [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
                 (2, 5), (3, 4), (4, 5), (5, 5), (6, 5)]
                if s == string_num}
            
            if ambiguous_multiplier > 1.0 and ambiguous_frets:
                # Upweight ambiguous positions
                non_ambiguous_frets = [f for f in by_fret.keys() if f not in ambiguous_frets]
                num_ambiguous = len([f for f in by_fret.keys() if f in ambiguous_frets])
                num_non_ambiguous = len(non_ambiguous_frets)
                
                effective_frets = num_non_ambiguous + num_ambiguous * ambiguous_multiplier
                base_per_fret = target_count / effective_frets if effective_frets > 0 else target_count / num_frets
                
                balanced = []
                for fret, fret_rows in by_fret.items():
                    if fret in ambiguous_frets:
                        fret_target = int(base_per_fret * ambiguous_multiplier)
                    else:
                        fret_target = int(base_per_fret)
                    
                    # Take samples, cycling if needed for upweighting
                    if len(fret_rows) >= fret_target:
                        balanced.extend(fret_rows[:fret_target])
                    else:
                        cycles = fret_target // len(fret_rows)
                        remainder = fret_target % len(fret_rows)
                        for _ in range(cycles):
                            balanced.extend(fret_rows)
                        balanced.extend(fret_rows[:remainder])
                
                rows_copy = balanced
            else:
                # Round-robin across frets
                balanced = []
                frets = list(by_fret.keys())
                idx = {f: 0 for f in frets}
                
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
            if max_per_string and len(rows_copy) > max_per_string:
                rows_copy = rows_copy[:max_per_string]
        
        selected.extend(rows_copy)
        
        ambiguous_count = sum(1 for r in rows_copy 
                             if is_ambiguous_position(r["string"], r["fret"]))
        print(f"String {string_num}: selected {len(rows_copy)} samples ({ambiguous_count} ambiguous)")
    
    return selected


def write_training_mix_csv(
    rows: List[dict],
    output_csv: Path,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[int, int]:
    """
    Write training mix CSV with train/val split.
    
    Args:
        rows: List of row dicts
        output_csv: Output CSV path
        train_ratio: Fraction for training
        seed: Random seed
    
    Returns:
        (train_count, val_count)
    """
    # Split by provenance
    random.seed(seed)
    
    by_group: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_group[get_group_key(r)].append(r)
    
    groups = list(by_group.items())
    random.shuffle(groups)
    
    total = sum(len(g[1]) for g in groups)
    target_train = int(total * train_ratio)
    
    train_groups = set()
    val_groups = set()
    
    running = 0
    for key, group_rows in groups:
        if running < target_train:
            train_groups.add(key)
            running += len(group_rows)
        else:
            val_groups.add(key)
    
    if not val_groups and groups:
        key, _ = groups[-1]
        train_groups.discard(key)
        val_groups.add(key)
    
    # Assign splits
    final_rows = []
    for key, group_rows in by_group.items():
        split = "train" if key in train_groups else "val"
        for r in group_rows:
            r_copy = dict(r)
            r_copy["split"] = split
            final_rows.append(r_copy)
    
    train_count = sum(1 for r in final_rows if r["split"] == "train")
    val_count = sum(1 for r in final_rows if r["split"] == "val")
    
    print(f"Final split: {train_count} train / {val_count} val")
    
    # Write CSV
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
    
    return train_count, val_count
