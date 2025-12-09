# src/report_balance.py

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META_DIR = ROOT / "data" / "pilot_0" / "guitar_dataset" / "meta"
FEATURE_META = META_DIR / "pilot_0_features_metadata.csv"


def load_feature_metadata():
    if not FEATURE_META.exists():
        raise FileNotFoundError(f"Feature metadata not found: {FEATURE_META}")
    with FEATURE_META.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for r in rows:
        r["string"] = int(r["string"])
        r["fret"] = int(r["fret"])
        r["octave"] = int(r["octave"])
        r["midi"] = int(r["midi"])
        r["is_ambiguous"] = r["is_ambiguous"].lower() == "true"
    return rows


def report_balance():
    rows = load_feature_metadata()

    pos_counter = Counter()
    note_counter = Counter()
    ambiguous_counter = Counter()

    for r in rows:
        pos = (r["string"], r["fret"])
        pos_counter[pos] += 1
        note_counter[(r["note"], r["octave"])] += 1
        ambiguous_counter[r["is_ambiguous"]] += 1

    print("=== Position counts (string, fret) ===")
    for (s, f), c in sorted(pos_counter.items()):
        print(f"string={s} fret={f}: {c}")

    print("\n=== Note counts (note, octave) ===")
    for (note, octv), c in sorted(note_counter.items()):
        print(f"{note}{octv}: {c}")

    print("\n=== Ambiguity counts ===")
    for key, c in ambiguous_counter.items():
        print(f"is_ambiguous={key}: {c}")

    total = len(rows)
    print(f"\nTotal samples: {total}")


if __name__ == "__main__":
    report_balance()
