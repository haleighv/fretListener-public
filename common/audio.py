#!/usr/bin/env python3
"""
common/audio.py - Core audio utilities.

Shared utilities used by both training and inference code.
For training-specific preprocessing, see audio_training.py.
For inference-specific processing, see audio_inference.py.
"""

import numpy as np
from typing import Optional, Tuple
from pathlib import Path


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_SAMPLE_RATE = 48000


# =============================================================================
# BASIC SIGNAL PROCESSING
# =============================================================================

def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """Remove DC offset from audio signal."""
    return audio - np.mean(audio)


def compute_rms(audio: np.ndarray) -> float:
    """Compute RMS of audio signal."""
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio ** 2)))


def compute_peak(audio: np.ndarray) -> float:
    """Compute peak amplitude of audio signal."""
    if audio.size == 0:
        return 0.0
    return float(np.max(np.abs(audio)))


# =============================================================================
# FILE I/O
# =============================================================================

def save_wav_int16(path: Path, sr: int, audio_float32: np.ndarray) -> None:
    """
    Save audio as 16-bit WAV file.
    
    Args:
        path: Output file path
        sr: Sample rate
        audio_float32: Audio data as float32 array (-1.0 to 1.0)
    """
    from scipy.io import wavfile
    
    x = np.clip(audio_float32.astype(np.float32), -1.0, 1.0)
    x_i16 = np.int16(x * 32767)
    wavfile.write(str(path), sr, x_i16)


def load_wav_mono(path: Path, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Load WAV file as mono float32 array.
    
    Args:
        path: Input file path
        target_sr: Target sample rate (None = use file's native rate)
    
    Returns:
        (audio_array, sample_rate)
    """
    import soundfile as sf
    
    audio, sr = sf.read(str(path), dtype='float32')
    
    # Convert to mono if stereo
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    
    # Resample if needed
    if target_sr is not None and sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    return audio.astype(np.float32), sr
