# fretListener – Note Classification Stage

This document captures the current state of the **note+octave classification** pipeline for the `fretListener` project.

The goal at this stage is to train a robust, low‑latency CNN that predicts the **MIDI note (note+octave)** from individual guitar note recordings, without modeling string or fret explicitly.

---

## Repository Layout (Relevant Pieces)

Assumed project root:

```text
~/code/fretListener
├── data
│   ├── strat_neck_40
│   │   ├── organized/   # raw organized takes (sX_fY directories)
│   │   ├── audio/       # flat audio files (built by tool)
│   │   └── meta/        # metadata CSVs (built by tool)
│   └── training_mix_neck.csv  # training/validation mix over one or more datasets
│
├── src
│   ├── note_model.py        # CNN definition for note+octave classification
│   └── train_note_model.py  # training pipeline + artifact tracking
│
└── tools
    ├── build_dataset_from_organized.py  # (previous step; builds audio/meta)
    └── build_training_mix.py            # training mix builder
```

This README focuses on the **note model** and **training pipeline**.

---

## Data Flow Overview

1. **Raw takes** live under an `organized/` folder per dataset, for example:

   ```text
   data/strat_neck_40/organized/
     s1_f0/, s1_f1/, ..., s6_f7/
   ```

   Each `sX_fY` directory contains `.wav` files of single notes (different takes).

2. **Dataset builder tool** (`tools/build_dataset_from_organized.py`) flattens the organized data into:

   ```text
   data/strat_neck_40/
     audio/
       <all wavs for this dataset>
     meta/
       metadata.csv    # or similar; maps audio file -> string, fret, midi, etc.
   ```

3. **Training mix builder** (`tools/build_training_mix.py`) samples from one or more dataset+meta pairs and writes a **training mix CSV**, e.g.:

   ```text
   data/training_mix_neck.csv
   ```

   This CSV contains (at minimum):

   - `abs_path`: absolute path to the audio file
   - `dataset`: dataset name (e.g., `strat_neck_40`)
   - `split`: `train` or `val`
   - `string`, `fret`, `note`, `octave`, `midi`
   - `take`: take index per (string, fret) group

4. The **note model training script** (`src/train_note_model.py`) consumes the training mix CSV to train a CNN that predicts **MIDI note** only (note+octave).

---

## Model: `src/note_model.py`

### Purpose

`NoteCNN` is a small 2D CNN that operates on **log‑mel spectrograms** of the note attack and predicts a class index corresponding to a MIDI note.

### Input/Output

- **Input**: `logmel` of shape `[B, 1, n_mels, T]`
  - Default: `n_mels = 48`, `T = 32` frames.
- **Output**: logits of shape `[B, num_classes]`
  - Each class corresponds to one **MIDI note value** present in the training mix.

### Architecture (Current)

- Conv block 1: `Conv2d(1 → 32, k=3, padding=1)` → `BatchNorm` → `ReLU` → `MaxPool2d(2)`
- Conv block 2: `Conv2d(32 → 64, k=3, padding=1)` → `BatchNorm` → `ReLU` → `MaxPool2d(2)`
- Conv block 3: `Conv2d(64 → 128, k=3, padding=1)` → `BatchNorm` → `ReLU`
- Global average pooling over frequency × time → `[B, 128]`
- Fully connected head:
  - `fc1: 128 → 128` with `ReLU`
  - `Dropout(p=0.3)`
  - `fc2: 128 → num_classes`

A helper `forward_with_embedding` is provided if embeddings are needed later (e.g., for fusion models).

---

## Training Script: `src/train_note_model.py`

### High‑Level Behavior

- Loads a **training mix CSV** (e.g., `data/training_mix_neck.csv`).
- Builds a MIDI‑to‑class mapping across both train and validation rows.
- Constructs a `NoteDatasetFromMix` for:
  - Train split (with optional augmentation)
  - Validation split (no augmentation)
- Trains the `NoteCNN` with:
  - Cross‑entropy loss + label smoothing
  - Adam optimizer
  - LR scheduler based on validation accuracy
- Tracks metrics and writes artifacts into a **timestamped run directory** under `artifacts/`.

### Dataset: `NoteDatasetFromMix`

Key parameters (defaults):

- `sample_rate = 48000`
- `attack_duration = 0.25` seconds
- `n_fft = 1024`
- `hop_length = 256`
- `n_mels = 48`
- `f_min = 55.0 Hz`
- `f_max = 3000.0 Hz`
- `num_frames = 32`

Steps per sample:

1. **Waveform loading**:
   - Load `.wav` via `torchaudio.load`.
   - Convert to mono (channel mean) if multi‑channel.
   - Resample to `sample_rate` if necessary.
   - Crop or pad to `attack_duration` (fixed number of samples).

2. **Optional augmentation (train only)**:
   - Small gain jitter (±~1 dB).
   - Low‑level additive noise (relative to signal max).
   - Tiny time shift (up to ~2 ms) via `torch.roll`.

3. **Feature extraction**:
   - `MelSpectrogram` with parameters above.
   - Convert to dB using `AmplitudeToDB`.
   - Per‑sample normalization: subtract mean and divide by std over **time** for each mel band.
   - Crop or pad in time to `num_frames`.

4. Returns:
   - `spec: [1, n_mels, num_frames]` (log‑mel tensor)
   - `label: int` (class index derived from MIDI via `midi_to_idx`).

### Training Details

- **Loss**: `nn.CrossEntropyLoss(label_smoothing=0.05)`
- **Optimizer**: `Adam` with configurable learning rate (default `1e-3`)
- **Scheduler**: `ReduceLROnPlateau` on **validation accuracy**:
  - `mode="max"`
  - `factor=0.1`
  - `patience=5`
- **Epoch Loop**:
  - Per epoch:
    - Run one training epoch (loss + accuracy).
    - Evaluate on validation set (loss + accuracy).
    - Step LR scheduler with current val accuracy.
    - Track best model by **val accuracy** (tie‑broken with lower val loss).
    - Save best model checkpoint into the run directory.

---

## Artifacts and Run Directories

Each training run creates a unique directory under:

```text
artifacts/note_YYYYMMDD_HHMMSS[_NN]/
```

Contents include:

- `training_mix.csv` – a copy of the mix used for the run.
- `note_cnn_best.pt` – best model checkpoint (state dict + metadata):
  - `model_state`
  - `num_classes`
  - `midi_to_idx` / `idx_to_midi`
  - `sample_rate`
  - `attack_samples`
  - `mel_params` (FFT, hop, n_mels, f_min, f_max, num_frames)
- `loss_curves.png` – training and validation loss vs epoch.
- `accuracy_curves.png` – training and validation accuracy vs epoch.
- `summary.txt` – human‑readable training summary including:
  - Hyperparameters
  - Data sizes
  - Class mapping (MIDI ↔ class index)
  - Per‑epoch metrics
  - Best epoch (val loss/acc)
  - Validation confusion matrix
- `note_cnn.onnx` – ONNX export for Netron visualization (if export succeeds).

If ONNX export fails in the future (e.g., missing `onnx`), training still completes and logs the failure in the artifacts directory (depending on the export implementation version).

---

## Usage

### 1. Prepare Training Mix

Assuming you have already run the dataset and mix builders:

```bash
# Example: build training mix across one or more datasets
python tools/build_training_mix.py   --datasets data/strat_neck_40   --output data/training_mix_neck.csv   --samples-per-class 40   --val-ratio 0.2   --seed 42
```

(Exact CLI flags may differ depending on the final implementation of `build_training_mix.py`.)

### 2. Train Note Model

From the project root:

```bash
cd ~/code/fretListener

python src/train_note_model.py   --mix-csv data/training_mix_neck.csv   --batch-size 64   --epochs 30
```

The script will:

- Select the best available device (MPS → CUDA → CPU).
- Create a new artifacts directory under `artifacts/`.
- Train and validate for the specified number of epochs.
- Save the best checkpoint and plots.
- Emit a training summary and confusion matrix to `summary.txt`.

---

## Next Steps

With this note classifier in place, the next stages of the project are:

1. **Realtime note listener** – use the trained `note_cnn_best.pt` with a realtime audio pipeline to emit:
   - MIDI note
   - Note name
   - Confidence

2. **String/fret modeling** – extend from note‑only classification to full (string, fret) inference, using either:
   - Direct 48‑class models, or
   - Fusion architectures that combine note + additional features (e.g., attack characteristics, prior fretboard state).

This README describes the state of the system at the **note+octave classification** milestone.
