import csv
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


class AttackDatasetFromMix(Dataset):
    """
    Dataset for 48-class (string,fret) attack classification.

    Consumes a training mix CSV produced by tools/build_training_mix.py and
    performs all audio preprocessing required for the attack model:

      - Loads audio from abs_path
      - Converts to mono
      - Resamples to target sample rate if needed
      - Crops/pads to fixed attack_duration
      - Computes log-mel spectrogram with configured parameters
      - Crops/pads time axis to fixed num_frames

    Returns:
      spec: Float tensor [1, n_mels, num_frames]
      label: int pos in [0, 47]
    """

    def __init__(
        self,
        csv_path: Path | str,
        split: str = "train",
        sample_rate: int = 48000,
        attack_duration: float = 0.19,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 64,
        f_min: float = 55.0,
        f_max: float = 2000.0,
        num_frames: int = 32,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            csv_path: Path to training mix CSV.
            split: "train" or "val".
            sample_rate: Target sample rate for training.
            attack_duration: Attack window in seconds.
            n_fft, hop_length, n_mels, f_min, f_max: Mel spectrogram params.
            num_frames: Time frames to crop/pad to.
            device: Optional device for pre-loaded transforms, otherwise CPU.
        """
        super().__init__()
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.split = split
        self.sample_rate = sample_rate
        self.attack_samples = int(round(sample_rate * attack_duration))
        self.num_frames = num_frames

        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split}")

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.rows: List[dict] = []
        with self.csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split") != split:
                    continue
                abs_path = row.get("abs_path")
                pos_str = row.get("pos")
                if not abs_path or pos_str is None:
                    continue
                try:
                    pos = int(pos_str)
                except ValueError:
                    continue
                row["pos"] = pos
                row["abs_path"] = abs_path
                self.rows.append(row)

        if not self.rows:
            raise RuntimeError(
                f"No rows found for split='{split}' in {self.csv_path}"
            )

        # Audio transforms
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        # Optional device for transforms
        if device is not None:
            self.mel_spec = self.mel_spec.to(device)
            self.db_transform = self.db_transform.to(device)
            self.transform_device = device
        else:
            self.transform_device = None

    def __len__(self) -> int:
        return len(self.rows)

    def _load_audio(self, path_str: str) -> torch.Tensor:
        """
        Loads audio from path, returns mono waveform at target sample rate.
        """
        path = Path(path_str)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        wav, sr = torchaudio.load(path)
        # To mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(
                wav, orig_freq=sr, new_freq=self.sample_rate
            )

        # Ensure fixed attack window
        if wav.size(1) < self.attack_samples:
            pad = self.attack_samples - wav.size(1)
            wav = torch.nn.functional.pad(wav, (0, pad))
        elif wav.size(1) > self.attack_samples:
            wav = wav[:, : self.attack_samples]

        return wav

    def _waveform_to_logmel(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Converts [1, T] waveform to [1, n_mels, num_frames] log-mel tensor.
        """
        device = self.transform_device or wav.device
        wav = wav.to(device)

        spec = self.mel_spec(wav)           # [1, n_mels, T_spec]
        spec_db = self.db_transform(spec)   # [1, n_mels, T_spec]
        _, n_mels, T = spec_db.shape

        # Pad or crop in time to num_frames
        if T < self.num_frames:
            pad = self.num_frames - T
            spec_db = torch.nn.functional.pad(spec_db, (0, pad))
        elif T > self.num_frames:
            spec_db = spec_db[:, :, : self.num_frames]

        return spec_db

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.rows[idx]
        abs_path = row["abs_path"]
        pos = row["pos"]

        wav = self._load_audio(abs_path)
        spec = self._waveform_to_logmel(wav)

        # Return CPU tensors by default (training script can move to device)
        return spec.cpu(), int(pos)
