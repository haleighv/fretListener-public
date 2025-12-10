# FretListener Dataset Builders & Training Mix

Tools to standardize raw `organized/` guitar datasets into consistent `audio/` + `meta/` layouts and to create custom multi-dataset training mixes for model training.

All tools live under:

```bash
fretListener/tools/
```

---

## 1. Build `audio/` + `meta/metadata.csv` from `organized/`

This tool scans a dataset of the form:

```text
data/<dataset_name>/
  ├── organized/
  │   ├── s1_f0/
  │   ├── s1_f1/
  │   └── ...
  ├── audio/   (generated)
  └── meta/    (generated)
```

### What it does

- Flattens all `.wav` files under `organized/` into a single `audio/` directory.
- Generates `meta/metadata.csv` describing each file with:
  - `id` – stable integer ID within the dataset.
  - `dataset_name` – top-level dataset directory name (e.g. `strat_neck_40`).
  - `relpath` – path to the copied audio file relative to the dataset root.
  - `filename` – filename in `audio/`.
  - `string` – 1–6 (high E = 1, low E = 6).
  - `fret` – 0–7 for current pilots.
  - `pos` – 48-class index: `(string - 1) * 8 + fret`.
  - `midi` – MIDI note number derived from standard tuning and fret.
  - `note` – note name (e.g. `G#3`) from MIDI.
  - `take` – parsed from filename (e.g. `_t03_` → 3), `-1` if not found.
  - `sample_rate` – audio sample rate from file header.
  - `num_samples` – number of samples in the file.
  - `source_relpath` – original path relative to dataset root (for traceability).

This script is non-destructive to `organized/` and only creates/overwrites `audio/` and `meta/` under the dataset root.

### Usage (CLI)

```bash
cd ~/code/fretListener

python tools/build_audio_and_meta.py   --organized-dir data/strat_neck_40/organized   --samples-per-class 40
```

- `--organized-dir` – path to the `organized/` directory.
- `--samples-per-class` – max number of `.wav` files to take per class directory (e.g. 40).

### Interactive Mode

```bash
python tools/build_audio_and_meta.py
```

Prompts for:

- Path to `organized/` directory.
- Samples per class (defaults to 40 if blank).
- Overwrite confirmation if `audio/` or `meta/` already exist and are non-empty.

### Output Layout

```text
data/strat_neck_40/
  ├── organized/
  │   └── sX_fY/...
  ├── audio/
  │   ├── s1_f0__s1_f0_t01_sessB.wav
  │   ├── s1_f0__s1_f0_t02_sessB.wav
  │   └── ...
  └── meta/
      └── metadata.csv
```

`metadata.csv` columns:

```text
id
dataset_name
relpath
filename
string
fret
pos
midi
note
take
sample_rate
num_samples
source_relpath
```

---

## 2. Build a Custom Training Mix (Single or Multi-Dataset)

This tool combines one or more datasets that already contain `audio/` and `meta/metadata.csv` into a unified training mix CSV for model training.

### Capabilities

- Accepts one or many dataset roots (e.g. `data/strat_neck_40`, `data/strat_bridge_40`).
- Merges metadata across datasets.
- Groups samples by `pos` (48-class index).
- Optional per-class cap via `--max-per-class`.
- Per-class train/val split using `--train-ratio`.
- Deterministic shuffling via `--seed`.
- Adds absolute audio path (`abs_path`) for direct use in training.
- Preserves `id` and `take` from metadata.

### Usage (Single Dataset)

```bash
python tools/build_training_mix.py   --datasets data/strat_neck_40   --max-per-class 40   --train-ratio 0.8   --output data/training_mix_neck.csv
```

### Usage (Multiple Datasets)

```bash
python tools/build_training_mix.py   --datasets data/strat_neck_40 data/strat_bridge_40   --max-per-class 40   --train-ratio 0.8   --output data/training_mix_neck_bridge.csv
```

Arguments:

- `--datasets` – one or more dataset roots that contain `audio/` + `meta/metadata.csv`.
- `--max-per-class` – optional cap on samples per `pos` class (default: no cap).
- `--train-ratio` – fraction of samples per class assigned to the `train` split (default: 0.8).
- `--output` – path to output training mix CSV (default: `data/training_mix.csv`).
- `--seed` – random seed for shuffling (default: 42).

### Interactive Mode

```bash
python tools/build_training_mix.py
```

Prompts for:

- Dataset roots (one per line, blank line to finish).
- Max samples per class (blank for no cap).
- Train ratio.
- Output CSV path.
- Final confirmation before writing.

### Training Mix Output

The resulting CSV (e.g. `data/training_mix.csv`) has the following columns:

```text
id
dataset_name
dataset_root
relpath
abs_path
filename
string
fret
pos
midi
note
take
sample_rate
num_samples
source_relpath
split   (train | val)
```

This file is designed to be consumed directly by PyTorch-style dataset loaders.

---

## 3. Typical Workflow

1. Normalize each raw dataset:

```bash
python tools/build_audio_and_meta.py   --organized-dir data/strat_neck_40/organized   --samples-per-class 40

python tools/build_audio_and_meta.py   --organized-dir data/strat_bridge_40/organized   --samples-per-class 40
```

2. Build a unified training mix for model training:

```bash
python tools/build_training_mix.py   --datasets data/strat_neck_40 data/strat_bridge_40   --max-per-class 40   --train-ratio 0.8   --output data/training_mix.csv
```

3. Use `data/training_mix.csv` as the single source of truth for a training loader:
   - Read rows.
   - Filter by `split` (`train` / `val`).
   - Load audio from `abs_path`.
   - Use `string`, `fret`, or `pos` as labels.
   - Optionally stratify or filter by `dataset_name`, `take`, etc.

This structure decouples raw data collection from training configuration and makes it easy to remix multiple datasets in the future.
