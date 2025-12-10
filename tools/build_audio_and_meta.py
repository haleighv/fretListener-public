#!/usr/bin/env python

import argparse
import sys
from pathlib import Path
import shutil
import csv
import re

import soundfile as sf


# ---------- helpers ----------

STANDARD_TUNING_MIDI = {
    6: 40,  # E2
    5: 45,  # A2
    4: 50,  # D3
    3: 55,  # G3
    2: 59,  # B3
    1: 64,  # E4
}


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]


TAKE_REGEX = re.compile(r"_t(\d+)_", re.IGNORECASE)


def parse_class_name(name: str) -> tuple[int, int, int]:
    """
    name: e.g. 's1_f0', 's6_f7'
    returns: (string, fret, pos)
      string: 1..6
      fret:   0..7
      pos:    (string-1)*8 + fret  -> 0..47
    """
    if not (name.startswith("s") and "_f" in name):
        raise ValueError(f"Unexpected class directory name: {name}")
    s_part, f_part = name.split("_f", 1)
    string = int(s_part[1:])
    fret = int(f_part)
    pos = (string - 1) * 8 + fret
    return string, fret, pos


def midi_from_string_fret(string: int, fret: int) -> int:
    base = STANDARD_TUNING_MIDI.get(string)
    if base is None:
        raise ValueError(f"Invalid string number {string}")
    return base + fret


def midi_to_note_name(midi: int) -> str:
    name = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return f"{name}{octave}"


def extract_take_from_filename(filename: str) -> int:
    """
    Extracts take number from filenames like:
      s1_f0_t01_sessB.wav  -> 1
    Returns -1 if not found.
    """
    m = TAKE_REGEX.search(filename)
    if not m:
        return -1
    return int(m.group(1))


def prompt_path(prompt: str) -> Path:
    while True:
        value = input(prompt).strip()
        p = Path(value).expanduser()
        if p.exists():
            return p
        print(f"Path does not exist: {p}")


def prompt_int(prompt: str, default: int | None = None) -> int:
    while True:
        value = input(prompt).strip()
        if not value and default is not None:
            return default
        try:
            return int(value)
        except ValueError:
            print("Please enter an integer.")


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


# ---------- main builder ----------

def build_audio_and_meta(
    organized_dir: Path,
    samples_per_class: int,
    force: bool = False,
) -> None:
    organized_dir = organized_dir.resolve()
    if not organized_dir.is_dir():
        raise SystemExit(f"Organized directory not found: {organized_dir}")

    dataset_root = organized_dir.parent
    audio_dir = dataset_root / "audio"
    meta_dir = dataset_root / "meta"
    meta_csv = meta_dir / "metadata.csv"

    print(f"Dataset root: {dataset_root}")
    print(f"Organized dir: {organized_dir}")
    print(f"Audio dir:     {audio_dir}")
    print(f"Meta dir:      {meta_dir}")
    print(f"Meta csv:      {meta_csv}")
    print(f"Samples/class: {samples_per_class}")

    if audio_dir.exists() and any(audio_dir.iterdir()):
        if not force:
            ok = prompt_yes_no(
                f"Audio directory {audio_dir} already exists and is not empty. Overwrite?",
                default=False,
            )
            if not ok:
                raise SystemExit("Aborted by user.")
        shutil.rmtree(audio_dir)

    if meta_dir.exists() and any(meta_dir.iterdir()):
        if not force:
            ok = prompt_yes_no(
                f"Meta directory {meta_dir} already exists and is not empty. Overwrite?",
                default=False,
            )
            if not ok:
                raise SystemExit("Aborted by user.")
        shutil.rmtree(meta_dir)

    audio_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    file_id = 0

    class_dirs = sorted(
        [p for p in organized_dir.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )

    if not class_dirs:
        raise SystemExit(f"No class directories found under {organized_dir}")

    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        string, fret, pos = parse_class_name(cls_name)

        wavs = sorted(cls_dir.glob("*.wav"))
        if not wavs:
            print(f"Warning: no .wav files in {cls_dir}")
            continue

        selected = wavs[:samples_per_class]
        print(
            f"Class {cls_name}: using {len(selected)} / {len(wavs)} files "
            f"(string={string}, fret={fret}, pos={pos})"
        )

        for w in selected:
            data, sr = sf.read(str(w), dtype="float32", always_2d=False)

            if isinstance(data, tuple):
                num_samples = data[0]
            else:
                num_samples = data.shape[0] if hasattr(data, "shape") else len(data)

            midi = midi_from_string_fret(string, fret)
            note = midi_to_note_name(midi)
            take = extract_take_from_filename(w.name)

            target_name = f"{cls_name}__{w.name}"
            target_path = audio_dir / target_name
            shutil.copy2(w, target_path)

            rows.append(
                {
                    "id": file_id,
                    "dataset_name": dataset_root.name,
                    "relpath": str(target_path.relative_to(dataset_root)),
                    "filename": target_name,
                    "string": string,
                    "fret": fret,
                    "pos": pos,
                    "midi": midi,
                    "note": note,
                    "take": take,
                    "sample_rate": sr,
                    "num_samples": int(num_samples),
                    "source_relpath": str(w.relative_to(dataset_root)),
                }
            )
            file_id += 1

    if not rows:
        raise SystemExit("No audio files processed; nothing to write.")

    fieldnames = [
        "id",
        "dataset_name",
        "relpath",
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
    ]

    with meta_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote metadata for {len(rows)} files to {meta_csv}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Build flat audio/ and meta/metadata.csv from organized/ dataset."
    )
    parser.add_argument(
        "--organized-dir",
        type=str,
        help="Path to organized/ directory (e.g. data/strat_neck_40/organized)",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        help="Max number of .wav files to take per class directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing audio/ and meta/ without prompting.",
    )
    args = parser.parse_args()

    if args.organized_dir:
        organized_dir = Path(args.organized_dir).expanduser()
    else:
        organized_dir = prompt_path(
            "Enter path to organized/ directory "
            "(e.g. data/strat_neck_40/organized): "
        )

    if args.samples_per_class is not None:
        samples_per_class = args.samples_per_class
    else:
        samples_per_class = prompt_int(
            "How many samples per class should be used? [default 40]: ",
            default=40,
        )

    build_audio_and_meta(
        organized_dir=organized_dir,
        samples_per_class=samples_per_class,
        force=args.force,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)
