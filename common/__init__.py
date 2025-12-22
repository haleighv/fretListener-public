#!/usr/bin/env python3
"""
common/ - Shared utilities for fretListener project.

Module Organization:
- audio.py: Core audio utilities (shared by everything)
- audio_training.py: Training/data collection preprocessing
- audio_inference.py: Inference audio processing
- guitar.py: Guitar theory constants and conversions
- dataset.py: Dataset loading and training mix utilities
- fretboard.py: Tkinter fretboard widget
"""

# =============================================================================
# Core audio utilities
# =============================================================================
from .audio import (
    # Constants
    DEFAULT_SAMPLE_RATE,
    # Functions
    remove_dc_offset,
    compute_rms,
    compute_peak,
    save_wav_int16,
    load_wav_mono,
)

# =============================================================================
# Training audio preprocessing
# =============================================================================
from .audio_training import (
    # Constants
    DEFAULT_FADE_MS,
    DEFAULT_HPF_ALPHA,
    DEFAULT_ONSET_RMS_WINDOW,
    DEFAULT_ONSET_THRESHOLD,
    DEFAULT_MIN_GAP_SEC,
    DEFAULT_ANALYSIS_HOP,
    # Functions
    apply_highpass_filter,
    apply_raised_cosine_fades,
    preprocess_audio,
    detect_onsets,
)

# =============================================================================
# Inference audio processing (requires torch)
# =============================================================================
try:
    from .audio_inference import (
        # Functions
        peak_normalize,
        waveform_to_logmel_torch,
        waveform_to_logmel_numpy,
        create_mel_transforms,
        prepare_audio_for_inference,
        sweep_best_window,
    )
    INFERENCE_AVAILABLE = True
except ImportError:
    # torch not available - inference functions won't be available
    INFERENCE_AVAILABLE = False
    peak_normalize = None
    waveform_to_logmel_torch = None
    waveform_to_logmel_numpy = None
    create_mel_transforms = None
    prepare_audio_for_inference = None
    sweep_best_window = None

# =============================================================================
# Guitar theory
# =============================================================================
from .guitar import (
    # Constants
    STANDARD_TUNING_MIDI,
    STANDARD_TUNING_LIST,
    NOTE_NAMES,
    STRING_NAMES_BY_INDEX,
    STRING_NAMES_TOP_TO_BOTTOM,
    AMBIGUOUS_POSITIONS,
    # Functions
    midi_to_note_name,
    note_name_to_midi,
    midi_from_string_fret,
    string_num_to_name,
    string_idx_to_num,
    string_num_to_idx,
    is_ambiguous_position,
    count_ambiguous_samples,
    compute_position,
    position_to_string_fret,
)

# =============================================================================
# Dataset utilities
# =============================================================================
from .dataset import (
    load_metadata_csv,
    normalize_row,
    get_group_key,
    split_by_provenance,
    load_datasets,
    balance_by_class,
    balance_by_string_with_fret_stratification,
    write_training_mix_csv,
)

# =============================================================================
# Version
# =============================================================================
__version__ = "1.0.0"
