# fretListener

**fretListener** is an end-to-end machine-learning system for guitar note recognition.

This repository contains tools for:
- Structured audio dataset collection from guitar strings/frets
- Training a convolutional neural network (CNN) on log-mel spectrograms
- Offline and realtime inference
- Rigorous batch evaluation with ground-truth validation

---

## Project Goals

- Accurately classify **guitar notes (pitch + octave)** from raw audio
- Maintain strict alignment between:
  - data collection
  - training preprocessing
  - inference preprocessing
- Support both **offline evaluation** and **realtime audio capture**
- Be understandable and extensible by adjacent fields (EE, DSP, ML)

---

## Repository Structure

```
fretListener/
├── data/               # Datasets and generated training mixes
├── artifacts/
│   └── models/         # Training runs, checkpoints, plots, summaries
├── recipes/            # Generated CSV training mixes
├── src/                # Core model and training code
│   ├── note_model.py
│   └── train_note_model.py
├── tools/              # Dataset + inference utilities
│   ├── build_training_mix.py
│   ├── infer_note_from_wav.py
│   └── realtime_note_inference.py
└── README.md
```

---

## Dataset Format

Each dataset root contains:

```
dataset_root/
├── audio/
├── organized/
│   └── s{string}_f{fret}/
│       └── s{string}_f{fret}_tXX_*.wav
└── meta/
    └── metadata.csv
```

- `string` is **1-based**
- Standard tuning assumed unless overridden:
  - s1 = high E (E4 / MIDI 64)
  - s6 = low E (E2 / MIDI 40)

---

## Building a Training Mix

```bash
python tools/build_training_mix.py   --datasets data/strat_neck_40 data/strat_bridge_40   --max-per-class 40   --train-ratio 0.8   --output recipes/training_mix.csv
```

Output CSVs are written to:

```
fretListener/recipes/
```

---

## Training the Model

```bash
python src/train_note_model.py   --mix-csv recipes/training_mix.csv   --batch-size 64   --epochs 30
```

Artifacts are written to:

```
artifacts/models/note_YYYYMMDD_HHMMSS/
```

Including:
- Best checkpoint (`note_cnn_best.pt`)
- Training curves
- Confusion matrix
- ONNX export
- Summary report

---

## Offline Inference (Single File)

```bash
python tools/infer_note_from_wav.py   --checkpoint artifacts/models/note_*/note_cnn_best.pt   --wav path/to/file.wav
```

---

## Batch Evaluation (Dataset)

```bash
python tools/infer_note_from_wav.py   --checkpoint artifacts/models/note_*/note_cnn_best.pt   --dataset-root data/strat_neck_40   --string-base-midi 64,59,55,50,45,40
```

This:
- Parses expected notes from `s{string}_f{fret}`
- Computes accuracy
- Prints per-file MISS diagnostics

Example result:

```
Evaluated: 2011
Correct:   1986
Accuracy:  0.9876
```

---

## Realtime Inference

```bash
python tools/realtime_note_inference.py   --checkpoint artifacts/models/note_*/note_cnn_best.pt
```

Features:
- Peak-based triggering (collector-matched)
- Pre/post buffers
- Sliding-window confidence sweep
- Optional WAV capture for debugging

---

## Design Principles

- **Preprocessing parity** between training and inference
- **Pitch-invariant augmentation avoided**
- **Attack-focused windows**
- CPU-based FFT to avoid MPS limitations
- Dataset-driven correctness checks

---

## Status

- High offline accuracy (~99%)
- Known edge errors around adjacent frets on low strings
- Realtime capture stable, regression-resistant

---

## License

MIT (or TBD)
