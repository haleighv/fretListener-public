#!/usr/bin/env python

import numpy as np
import pandas as pd
from pathlib import Path

# ============================
# CONFIG
# ============================

DATA_ROOT = Path("pilot1_two_strings")
META_CSV_IN = DATA_ROOT / "metadata.csv"
META_CSV_OUT = DATA_ROOT / "metadata_balanced.csv"

RANDOM_SEED = 42


# ============================
# MAIN
# ============================

def main():
    if not META_CSV_IN.exists():
        raise FileNotFoundError(f"Input metadata not found: {META_CSV_IN}")

    df = pd.read_csv(META_CSV_IN)
    print(f"Loaded {len(df)} rows from {META_CSV_IN}")

    # Group by (string_idx, fret)
    grp = df.groupby(["string_idx", "fret"])
    counts = grp.size().reset_index(name="count")
    print("\nPer-position counts (before balancing):")
    print(counts.sort_values(["string_idx", "fret"]).to_string(index=False))

    min_count = counts["count"].min()
    print(f"\nMinimum samples per (string_idx, fret): {min_count}")

    rng = np.random.RandomState(RANDOM_SEED)
    keep_indices = []

    # For each position, randomly keep exactly min_count rows
    for (s_idx, fret), group in grp:
        idxs = group.index.to_numpy()
        if len(idxs) > min_count:
            chosen = rng.choice(idxs, size=min_count, replace=False)
        else:
            chosen = idxs
        keep_indices.append(chosen)

    keep_indices = np.concatenate(keep_indices)

    df_bal = df.loc[keep_indices].copy()
    # Shuffle rows for good measure
    df_bal = df_bal.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    df_bal.to_csv(META_CSV_OUT, index=False)
    print(f"\nWrote balanced metadata with {len(df_bal)} rows to {META_CSV_OUT}")

    # Show new counts
    grp_bal = df_bal.groupby(["string_idx", "fret"])
    counts_bal = grp_bal.size().reset_index(name="count")
    print("\nPer-position counts (after balancing):")
    print(counts_bal.sort_values(["string_idx", "fret"]).to_string(index=False))


if __name__ == "__main__":
    main()
