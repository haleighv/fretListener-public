# src/build_feature_metadata.py

import csv
from collections import defaultdict
from pathlib import Path

from labels import note_octave_to_midi, assert_string_fret_range

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "data" / "pilot_0" / "guitar_dataset" / "audio"
META_DIR = ROOT / "data" / "pilot_0" / "guitar_dataset" / "meta"

MASTER_META = META_DIR / "master_metadata.csv"
FEATURE_META = META_DIR / "pilot_0_features_metadata.csv"

MAX_FRET = 5


def build_feature_metadata():
    if not MASTER_META.exists():
        raise FileNotFoundError(f"master_metadata.csv not found at {MASTER_META}")

    rows = []
    midi_counts = defaultdict(int)

    with MASTER_META.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            filename = raw["filename"]
            string = int(raw["string"])
            fret = int(raw["fret"])
            note = raw["note"]
            octave = int(raw["octave"])
            take = int(raw["take"])
            session = raw["session"]
            guitar_model = raw.get("guitar_model", "")
            pickup_type = raw.get("pickup_type", "")
            pickup_selected = raw.get("pickup_selected", "")
            timestamp = raw.get("timestamp", "")

            # Enforce pilot constraints
            assert_string_fret_range(string, fret, max_fret=MAX_FRET)

            audio_path = AUDIO_DIR / filename
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file from metadata not found: {audio_path}")

            midi = note_octave_to_midi(note, octave)

            row = {
                "filepath": str(audio_path.relative_to(ROOT)),  # relative path
                "filename": filename,
                "string": string,
                "fret": fret,
                "note": note,
                "octave": octave,
                "take": take,
                "session": session,
                "guitar_model": guitar_model,
                "pickup_type": pickup_type,
                "pickup_selected": pickup_selected,
                "timestamp": timestamp,
                "midi": midi,
                "is_ambiguous": False,  # to be filled later
            }
            rows.append(row)
            midi_counts[midi] += 1

    # Mark ambiguous MIDI values: same pitch from multiple positions
    ambiguous_midis = {m for m, c in midi_counts.items() if c > 1}
    for r in rows:
        r["is_ambiguous"] = r["midi"] in ambiguous_midis

    # Write feature metadata
    FEATURE_META.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filepath",
        "filename",
        "string",
        "fret",
        "note",
        "octave",
        "take",
        "session",
        "guitar_model",
        "pickup_type",
        "pickup_selected",
        "timestamp",
        "midi",
        "is_ambiguous",
    ]

    with FEATURE_META.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {FEATURE_META}")


if __name__ == "__main__":
    build_feature_metadata()
