# Development log

How fretListener got from "kind of a mess" to a tiered realtime system, reconstructed from the commit history and per-run summaries in `artifacts/models/`. Dates are from commits (Dec 2025).

## Phase 0: One model to do everything (Dec 9)

- First attempt: a single CNN predicting note, string and fret jointly, plus a quick realtime listener.
- Test accuracy: **77.8% per output, 55.6% for all three at once**.
- The realtime path worked, but the codebase was "kind of a mess" (commit message).
- **Decision:** start over, beginning with the dataset and the recording tool.

## Phase 1: Data infrastructure first (Dec 9–11)

- Built a dataset builder with **take-aware metadata**: every clip records string, fret, MIDI pitch, take and source recording.
- Recorded `strat_neck_119`: 119 takes of each of 48 positions (6 strings × frets 0–7), 5,712 clips.
- `note_pilot_v1`, trained on two Strat neck-pickup datasets: **97.5%** validation accuracy across 32 pitches.

## Phase 2: A real recording tool (Dec 12–16)

- Wrote `tools/data_collection.py`. It prompts for a position, records continuous takes, detects onsets, and slices them into labeled clips.
- **Bug:** captured clips were getting chopped. The cause was doing processing inside the `sounddevice` audio callback.
- **Fix:** the callback only copies the buffer into a queue, and a worker thread does the processing. Same rule as an interrupt handler: do the minimum and defer the rest.
- Added PyInstaller build scripts for a standalone recording app.
- Added a realtime GUI with a visual fretboard, and a training GUI (`tools/train_gui.py`) that builds mixes and launches training.

## Phase 3: High accuracy, then an honest check (Dec 16)

| Run | Split | Val accuracy | n (val) |
|---|---|---|---|
| `note_pilot_v2` | random, 4 datasets | 99.8% | 1,824 |
| `note_pilot_v2_1` | **train on Strats, validate on unseen RG550** | **94.7%** | 960 |
| `note_pilot_v2_2` | random, 4 datasets | 99.6% | 1,815 |

- The v2.1 mix was rebuilt "to minimize leakage" (commit message): train on the Strats only, validate on the RG550 only.
- With a random split, the same guitar is on both sides, so a high score doesn't show whether the model works on a new guitar.
- Holding out an entire guitar cost about 5 points. That's the closer estimate of how it does on a new instrument.

## Phase 4: Which string? (Dec 16–19)

- Pitch alone is ambiguous: many notes can be played in 2–3 places.
- First approach: a direct 6-way `StringCNN`.
  - 97.6% on 4 datasets.
  - **93.2%** after adding two new full-neck (frets 0–12) Strat datasets, because more variety per label makes the task harder.
- Recorded `strat_mia_full_bridge_20` and `strat_mij_full_neck_20` to cover frets 0–12.
- `note_pilot_v3` (6 datasets, 37 pitches E2–E5): **99.8%** on a random split.
- Built `fretTrainer.py`, a practice tool for scales, arpeggios and intervals, on top of the note model.

## Phase 5: Refactor for parity (Dec 21)

- Split `common/audio` into `audio_training.py` and `audio_inference.py`, with shared primitives in `audio.py`.
- Goal: no silent drift between how training clips and live audio become spectrograms.

## Phase 6: Tiered disambiguation (Dec 22–26)

- **New idea:** don't ask "which of 6 strings?" Use the note model's pitch, look up the 2–3 places it can be played, and run a small model trained only on that decision.
- **Pairwise models** (E/A, A/D, D/G, G/B, B/e):
  - The first cut struggled with open strings.
  - After more data: **98–100%** validation, stable across 4 reruns (about 2 points spread at most).
  - Chained together ("smart chain") to handle 3-way cases.
- **Triplet models** (E/A/D, A/D/G, D/G/B, G/B/e): a direct 3-way decision, covering more of the neck.
  - Trained on 306–408 clips with 54–72 validation clips each.
  - Identical reruns ranged **71–100%**, a spread of up to 20 points.
  - Conclusion: these numbers can't be trusted yet. More data is needed before comparing architectures.

## Lessons

1. **Build the data pipeline before the model.** The restart in Phase 1 was the best decision in the project.
2. **Hold out what you'll meet in the wild.** For audio, that's the instrument, not the clip.
3. **Report n and run-to-run spread.** A 98% from 54 clips and a 98% from 2,400 are different claims.
4. **Decompose ambiguous problems.** Small specialist models for small decisions beat one model for everything, when each has enough data.
5. **Treat audio callbacks like interrupt handlers.**
