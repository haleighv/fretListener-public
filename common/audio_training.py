#!/usr/bin/env python3
"""
common/audio_training.py - Audio preprocessing for training data.

CRITICAL: All processing here must match data_collection.py exactly to ensure
training data consistency. Do not modify without updating all consumers.

Processing chain (in order):
1. DC offset removal: audio - mean(audio)
2. High-pass filter: alpha=0.995 (~20Hz cutoff at 48kHz)
3. Raised cosine fades: 3ms fade-in/fade-out

This module is used by:
- data_collection.py (creating training data)
- build_training_mix.py (verifying data)

NOT used during inference - see audio_inference.py for that.
"""

import numpy as np
from typing import List

from .audio import remove_dc_offset, compute_rms, DEFAULT_SAMPLE_RATE


# =============================================================================
# CONSTANTS (matching data_collection.py)
# =============================================================================

DEFAULT_FADE_MS = 3.0
DEFAULT_HPF_ALPHA = 0.995  # ~20Hz cutoff at 48kHz

# Onset detection defaults
DEFAULT_ONSET_RMS_WINDOW = 2048
DEFAULT_ONSET_THRESHOLD = 0.010
DEFAULT_MIN_GAP_SEC = 0.35
DEFAULT_ANALYSIS_HOP = 1024


# =============================================================================
# TRAINING PREPROCESSING CHAIN
# =============================================================================

def apply_highpass_filter(audio: np.ndarray, alpha: float = DEFAULT_HPF_ALPHA) -> np.ndarray:
    """
    Apply simple first-order high-pass filter to remove rumble.
    
    Args:
        audio: Input audio array
        alpha: Filter coefficient (0.995 gives ~20Hz cutoff at 48kHz)
    
    Returns:
        Filtered audio array
    """
    filtered = np.zeros_like(audio)
    filtered[0] = audio[0]
    for i in range(1, len(audio)):
        filtered[i] = alpha * (filtered[i-1] + audio[i] - audio[i-1])
    return filtered


def apply_raised_cosine_fades(
    audio: np.ndarray, 
    sr: int, 
    fade_ms: float = DEFAULT_FADE_MS
) -> np.ndarray:
    """
    Apply raised cosine fade-in and fade-out to audio.
    
    Args:
        audio: Input audio array
        sr: Sample rate
        fade_ms: Fade duration in milliseconds
    
    Returns:
        Audio with fades applied
    """
    fade_samples = int(sr * fade_ms / 1000)
    fade_samples = min(fade_samples, len(audio) // 4)  # Don't fade more than 25%
    
    if fade_samples < 2:
        return audio
    
    audio = audio.copy()
    
    # Raised cosine fades (smoother than linear)
    fade_in = 0.5 * (1 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))
    fade_out = 0.5 * (1 + np.cos(np.pi * np.arange(fade_samples) / fade_samples))
    
    fade_in = fade_in.astype(np.float32)
    fade_out = fade_out.astype(np.float32)
    
    # Apply fades
    audio[:fade_samples] *= fade_in
    audio[-fade_samples:] *= fade_out
    
    return audio


def preprocess_audio(
    audio: np.ndarray, 
    sr: int,
    fade_ms: float = DEFAULT_FADE_MS,
    hpf_alpha: float = DEFAULT_HPF_ALPHA
) -> np.ndarray:
    """
    Apply full preprocessing chain matching data_collection.py.
    
    Processing order:
    1. DC offset removal
    2. High-pass filter (~20Hz)
    3. Raised cosine fades (3ms)
    
    Args:
        audio: Input audio array (float32)
        sr: Sample rate
        fade_ms: Fade duration in milliseconds
        hpf_alpha: High-pass filter coefficient
    
    Returns:
        Preprocessed audio array
    """
    # 1. Remove DC offset
    audio = remove_dc_offset(audio)
    
    # 2. Apply high-pass filter
    audio = apply_highpass_filter(audio, alpha=hpf_alpha)
    
    # 3. Apply raised cosine fades
    audio = apply_raised_cosine_fades(audio, sr, fade_ms=fade_ms)
    
    return audio


# =============================================================================
# ONSET DETECTION (for data collection)
# =============================================================================

def detect_onsets(
    audio: np.ndarray,
    sr: int = DEFAULT_SAMPLE_RATE,
    rms_window: int = DEFAULT_ONSET_RMS_WINDOW,
    threshold: float = DEFAULT_ONSET_THRESHOLD,
    min_gap_sec: float = DEFAULT_MIN_GAP_SEC,
    analysis_hop: int = DEFAULT_ANALYSIS_HOP
) -> List[int]:
    """
    Detect note onsets using RMS with rising-edge detection.
    
    Args:
        audio: Input audio array
        sr: Sample rate
        rms_window: Window size for RMS calculation
        threshold: RMS threshold for onset detection
        min_gap_sec: Minimum gap between onsets in seconds
        analysis_hop: Hop size for analysis
    
    Returns:
        List of sample indices where onsets occur
    """
    min_gap_samples = int(sr * min_gap_sec)
    total_samples = len(audio)
    rms_vals = []
    onset_indices = []
    
    for i in range(0, total_samples - rms_window, analysis_hop):
        frame = audio[i:i + rms_window]
        rms = compute_rms(frame)
        rms_vals.append(rms)
        
        if rms > threshold:
            if len(rms_vals) < 2:
                onset_indices.append(i)
            else:
                prev_rms = rms_vals[-2]
                if rms > prev_rms * 1.5:
                    onset_indices.append(i)
    
    # Hard inter-onset suppression
    clean_onsets = []
    last = -10**12
    for idx in onset_indices:
        if idx - last >= min_gap_samples:
            clean_onsets.append(idx)
            last = idx
    
    return clean_onsets
