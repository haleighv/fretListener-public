#!/usr/bin/env python3
"""
common/audio_inference.py - Audio processing for inference.

This module contains audio processing functions used during real-time
inference in fretListener.py, fretTrainer.py, and similar tools.

The inference path is intentionally different from training preprocessing:
- Training: Full chain (DC removal → HPF → fades) via audio_training.py
- Inference: Peak normalize → DC removal in mel conversion

This module is used by:
- fretListener.py (real-time inference)
- fretTrainer.py (practice tool with inference)
- infer_note_from_wav.py (batch inference)
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any

from .audio import remove_dc_offset

# Optional torch import for environments without it
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Type hints that work with or without torch
if TORCH_AVAILABLE:
    TorchDevice = torch.device
    TorchTensor = torch.Tensor
else:
    TorchDevice = Any
    TorchTensor = Any


# =============================================================================
# PEAK NORMALIZATION
# =============================================================================

def peak_normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Peak normalize audio to target amplitude.
    
    This normalization is applied before inference to match the
    amplitude characteristics the model was trained on.
    
    Args:
        audio: Input audio array
        target_peak: Target peak amplitude (default 0.95)
    
    Returns:
        Peak-normalized audio array
    """
    audio = audio.astype(np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = (audio / peak) * target_peak
    return audio


# =============================================================================
# MEL SPECTROGRAM CONVERSION (PyTorch)
# =============================================================================

def waveform_to_logmel_torch(
    wav_1d: np.ndarray,
    mel_transform,
    db_transform,
    num_frames: int,
    device: TorchDevice
) -> TorchTensor:
    """
    Convert waveform to log-mel spectrogram using PyTorch transforms.
    
    Processing:
    1. Remove DC offset
    2. Convert to tensor
    3. Apply mel spectrogram transform (on CPU for MPS compatibility)
    4. Convert to dB scale
    5. Per-frame normalization
    6. Pad/truncate to fixed frame count
    7. Move to target device
    
    Args:
        wav_1d: Input waveform (1D numpy array, should be peak-normalized)
        mel_transform: torchaudio.transforms.MelSpectrogram instance
        db_transform: torchaudio.transforms.AmplitudeToDB instance  
        num_frames: Fixed number of output frames
        device: Target device for output tensor
    
    Returns:
        Log-mel spectrogram tensor [1, 1, n_mels, num_frames]
    """
    # Remove DC offset
    wav = wav_1d.astype(np.float32)
    wav = wav - float(np.mean(wav))
    
    # Convert to tensor (keep on CPU - FFT not supported on MPS)
    wav_t = torch.from_numpy(wav).float().unsqueeze(0)
    
    # Apply mel spectrogram and dB conversion (on CPU)
    spec = mel_transform(wav_t)
    spec_db = db_transform(spec)
    
    # Per-frame normalization
    mean = spec_db.mean(dim=2, keepdim=True)
    std = spec_db.std(dim=2, keepdim=True) + 1e-6
    spec_db = (spec_db - mean) / std
    
    # Pad or truncate to fixed frame count
    _, _, T = spec_db.shape
    if T < num_frames:
        pad = num_frames - T
        spec_db = torch.nn.functional.pad(spec_db, (0, pad))
    elif T > num_frames:
        spec_db = spec_db[:, :, :num_frames]
    
    # Add batch dimension and move to device
    return spec_db.unsqueeze(0).to(device)


def waveform_to_logmel_numpy(
    wav: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 128,
    n_mels: int = 64,
    f_min: float = 55.0,
    f_max: float = 3000.0,
    num_frames: int = 96
) -> np.ndarray:
    """
    Convert waveform to log-mel spectrogram using numpy/librosa.
    
    Alternative to PyTorch version for environments without torchaudio.
    
    Args:
        wav: Input waveform (1D array, should be peak-normalized)
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        n_mels: Number of mel bins
        f_min: Minimum frequency
        f_max: Maximum frequency
        num_frames: Fixed number of output frames
    
    Returns:
        Log-mel spectrogram array [n_mels, num_frames]
    """
    import librosa
    
    # Remove DC offset
    wav = wav.astype(np.float32)
    wav = wav - float(np.mean(wav))
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max
    )
    
    # Convert to dB
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Per-frame normalization
    mean = mel_db.mean(axis=1, keepdims=True)
    std = mel_db.std(axis=1, keepdims=True) + 1e-6
    mel_db = (mel_db - mean) / std
    
    # Pad or truncate to fixed number of frames
    if mel_db.shape[1] < num_frames:
        pad_width = num_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    elif mel_db.shape[1] > num_frames:
        mel_db = mel_db[:, :num_frames]
    
    return mel_db.astype(np.float32)


# =============================================================================
# MEL TRANSFORM SETUP
# =============================================================================

def create_mel_transforms(
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 128,
    n_mels: int = 64,
    f_min: float = 55.0,
    f_max: float = 3000.0,
) -> Tuple[Any, Any]:
    """
    Create mel spectrogram and dB transforms for inference.
    
    Transforms are kept on CPU since FFT is not supported on MPS.
    
    Args:
        sample_rate: Audio sample rate
        n_fft: FFT size
        hop_length: Hop length between frames
        n_mels: Number of mel frequency bins
        f_min: Minimum frequency for mel filterbank
        f_max: Maximum frequency for mel filterbank
    
    Returns:
        (mel_transform, db_transform) tuple
    """
    import torchaudio
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
    )  # Keep on CPU
    
    db_transform = torchaudio.transforms.AmplitudeToDB()  # Keep on CPU
    
    return mel_transform, db_transform


# =============================================================================
# INFERENCE HELPERS
# =============================================================================

def prepare_audio_for_inference(
    audio: np.ndarray,
    target_length: Optional[int] = None
) -> np.ndarray:
    """
    Prepare audio for model inference.
    
    Applies peak normalization and optional padding/truncation.
    
    Args:
        audio: Input audio array
        target_length: Target length in samples (None = no change)
    
    Returns:
        Prepared audio array
    """
    # Peak normalize
    audio = peak_normalize(audio)
    
    # Pad or truncate if target length specified
    if target_length is not None:
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        elif len(audio) > target_length:
            audio = audio[:target_length]
    
    return audio


def sweep_best_window(
    captured: np.ndarray,
    attack_samples: int,
    sweep_step_samples: int,
    predict_fn,
    topk: int = 5
) -> Tuple[int, np.ndarray, list]:
    """
    Sweep through captured audio to find best inference window.
    
    Args:
        captured: Captured audio array
        attack_samples: Window size for inference
        sweep_step_samples: Step size for sweep
        predict_fn: Function that takes window and returns topk results
        topk: Number of top predictions to track
    
    Returns:
        (best_start, best_window, best_topk) tuple
    """
    # Pad if too short
    if len(captured) < attack_samples:
        padded = np.zeros(attack_samples, dtype=np.float32)
        padded[:len(captured)] = captured.astype(np.float32)
        captured = padded
    
    best_start = 0
    best_window = captured[:attack_samples].copy()
    best_prob = -1.0
    best_topk = []
    
    last_start = max(0, len(captured) - attack_samples)
    step = max(1, int(sweep_step_samples))
    
    for start in range(0, last_start + 1, step):
        window = captured[start:start + attack_samples]
        topk_results = predict_fn(window, topk)
        
        if topk_results:
            # Assume first element of first result is (something, name, prob)
            prob = topk_results[0][2] if len(topk_results[0]) >= 3 else topk_results[0][-1]
            if prob > best_prob:
                best_prob = prob
                best_start = start
                best_window = window.copy()
                best_topk = topk_results
    
    return best_start, best_window, best_topk
