# src/labels.py

NOTE_TO_INDEX = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}

INDEX_TO_NOTE = {v: k for k, v in NOTE_TO_INDEX.items()}


def note_octave_to_midi(note: str, octave: int) -> int:
    """
    Convert note name + octave to MIDI number.

    Convention:
        C4 = 60
        midi = 12 * (octave + 1) + NOTE_TO_INDEX[note]
    """
    if note not in NOTE_TO_INDEX:
        raise ValueError(f"Unknown note name: {note}")
    return 12 * (octave + 1) + NOTE_TO_INDEX[note]


def assert_string_fret_range(string: int, fret: int, max_fret: int = 5) -> None:
    """
    Pilot constraint:
      string in [1..6]
      fret in [0..max_fret]
    """
    if not (1 <= string <= 6):
        raise ValueError(f"Invalid string: {string}")
    if not (0 <= fret <= max_fret):
        raise ValueError(f"Invalid fret: {fret}")
