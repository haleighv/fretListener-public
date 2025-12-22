#!/usr/bin/env python3
"""
common/guitar.py - Guitar theory constants and utilities.

String convention (matches existing data_collection.py and training data):
  - String 1 = high E (E4, MIDI 64)
  - String 6 = low E (E2, MIDI 40)

This is the guitarist's convention where string 1 is the thinnest/highest.
"""

from typing import Tuple, Set

# =============================================================================
# CONSTANTS
# =============================================================================

# Standard tuning MIDI values, indexed by string number (1-6)
# String 1 = high E (E4), String 6 = low E (E2)
STANDARD_TUNING_MIDI = {
    1: 64,  # E4 (high E)
    2: 59,  # B3
    3: 55,  # G3
    4: 50,  # D3
    5: 45,  # A2
    6: 40,  # E2 (low E)
}

# Standard tuning as list indexed 0-5 for internal display purposes
# Index 0 = low E (bottom of fretboard), Index 5 = high E (top of fretboard)
STANDARD_TUNING_LIST = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4

# Note names for MIDI conversion
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# String names for display (indexed 0-5, bottom to top of fretboard)
STRING_NAMES_BY_INDEX = ["E2", "A2", "D3", "G3", "B3", "E4"]

# String names top-to-bottom for fretboard display (high E at top)
STRING_NAMES_TOP_TO_BOTTOM = ["E4", "B3", "G3", "D3", "A2", "E2"]

# Ambiguous positions: open strings and their fretted equivalents (same pitch)
# Format: (string_num, fret) using 1-6 string convention
AMBIGUOUS_POSITIONS: Set[Tuple[int, int]] = {
    # Open strings
    (1, 0),  # Open E4
    (2, 0),  # Open B3
    (3, 0),  # Open G3
    (4, 0),  # Open D3
    (5, 0),  # Open A2
    # Fretted equivalents (same pitch as open string above)
    (2, 5),  # B string fret 5 = E4
    (3, 4),  # G string fret 4 = B3
    (4, 5),  # D string fret 5 = G3
    (5, 5),  # A string fret 5 = D3
    (6, 5),  # E string fret 5 = A2
}


# =============================================================================
# MIDI / NOTE CONVERSION
# =============================================================================

def midi_to_note_name(midi: int) -> str:
    """
    Convert MIDI note number to note name with octave.
    
    Examples:
        midi_to_note_name(40) -> "E2"
        midi_to_note_name(64) -> "E4"
        midi_to_note_name(60) -> "C4"
    """
    note = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return f"{note}{octave}"


def note_name_to_midi(name: str) -> int:
    """
    Convert note name with octave to MIDI number.
    
    Examples:
        note_name_to_midi("E2") -> 40
        note_name_to_midi("C#4") -> 61
    """
    name = name.strip()
    if len(name) < 2:
        raise ValueError(f"Invalid note name: {name}")
    
    if name[1] == '#':
        note_part = name[:2]
        octave_part = name[2:]
    else:
        note_part = name[0]
        octave_part = name[1:]
    
    try:
        note_idx = NOTE_NAMES.index(note_part.upper())
    except ValueError:
        raise ValueError(f"Invalid note: {note_part}")
    
    try:
        octave = int(octave_part)
    except ValueError:
        raise ValueError(f"Invalid octave: {octave_part}")
    
    return (octave + 1) * 12 + note_idx


def midi_from_string_fret(string_num: int, fret: int) -> int:
    """
    Get MIDI note number from string number (1-6) and fret.
    
    Args:
        string_num: 1-6 (1=high E, 6=low E)
        fret: 0-24
    
    Returns:
        MIDI note number
    
    Examples:
        midi_from_string_fret(6, 0) -> 40  # Open low E
        midi_from_string_fret(1, 0) -> 64  # Open high E
        midi_from_string_fret(6, 5) -> 45  # A2 on low E string
    """
    base = STANDARD_TUNING_MIDI.get(string_num)
    if base is None:
        raise ValueError(f"Invalid string number: {string_num} (must be 1-6)")
    return base + fret


def string_num_to_name(string_num: int) -> str:
    """
    Convert string number (1-6) to open string note name.
    
    Examples:
        string_num_to_name(6) -> "E2"
        string_num_to_name(1) -> "E4"
    """
    midi = STANDARD_TUNING_MIDI.get(string_num)
    if midi is None:
        return f"String {string_num}"
    return midi_to_note_name(midi)


def string_idx_to_name(string_idx: int) -> str:
    """
    Convert internal string index (0-5) to open string note name.
    Index 0 = low E (E2), Index 5 = high E (E4)
    
    Examples:
        string_idx_to_name(0) -> "E2"
        string_idx_to_name(5) -> "E4"
    """
    if 0 <= string_idx < 6:
        return STRING_NAMES_BY_INDEX[string_idx]
    return f"String {string_idx}"


def string_idx_to_num(string_idx: int) -> int:
    """
    Convert internal string index (0-5) to string number (1-6).
    Index 0 = String 6 (low E), Index 5 = String 1 (high E)
    """
    return 6 - string_idx


def string_num_to_idx(string_num: int) -> int:
    """
    Convert string number (1-6) to internal string index (0-5).
    String 6 = Index 0 (low E), String 1 = Index 5 (high E)
    """
    return 6 - string_num


# =============================================================================
# POSITION HELPERS
# =============================================================================

def is_ambiguous_position(string_num: int, fret: int) -> bool:
    """
    Check if this position is one of the ambiguous open/fret pairs
    where the same pitch can be played on adjacent strings.
    """
    return (string_num, fret) in AMBIGUOUS_POSITIONS


def count_ambiguous_samples(rows: list) -> int:
    """
    Count how many samples in a list of row dicts are from ambiguous positions.
    Expects rows to have 'string' and 'fret' keys.
    """
    count = 0
    for row in rows:
        try:
            string = int(row.get("string", 0))
            fret = int(row.get("fret", 0))
            if is_ambiguous_position(string, fret):
                count += 1
        except (ValueError, TypeError):
            pass
    return count


def compute_position(string_num: int, fret: int, frets_per_string: int = 13) -> int:
    """
    Compute position index for note classification.
    
    Args:
        string_num: 1-6 (1=high E, 6=low E)
        fret: 0-N
        frets_per_string: Number of frets per string in dataset (default 13 for 0-12)
    
    Returns:
        Position index: (string_num - 1) * frets_per_string + fret
    """
    return (string_num - 1) * frets_per_string + fret


def position_to_string_fret(pos: int, frets_per_string: int = 13) -> Tuple[int, int]:
    """
    Convert position index back to string number and fret.
    
    Returns:
        (string_num, fret) where string_num is 1-6
    """
    string_num = pos // frets_per_string + 1
    fret = pos % frets_per_string
    return string_num, fret
