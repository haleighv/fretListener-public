import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio
import torch

from pitch_yin import yin, hz_to_midi


def parse_class_name(name: str) -> tuple[int, int, int]:
    s_part, f_part = name.split("_f")
    string = int(s_part[1:])
    fret = int(f_part)
    pos = (string - 1) * 8 + fret
    return string, fret, pos


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio

    wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    wav_rs = torchaudio.functional.resample(wav, orig_sr, target_sr)
    return wav_rs.squeeze(0).numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Build fusion_dataset.csv from attack CNN embeddings + YIN pitch"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
    )
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--yin-frame-size", type=int, default=2048)
    args = parser.parse_args()

    print("Loading features:", args.features)
    npz = np.load(args.features, allow_pickle=True)

    embeddings = npz["embeddings"]
    pos_labels = npz["pos"]

    rows = []
    idx = 0
    total = len(pos_labels)

    print("Scanning dataset for pitch (YIN)...")

    for class_dir in sorted(args.dataset_root.iterdir()):
        if not class_dir.is_dir():
            continue

        string, fret, pos = parse_class_name(class_dir.name)
        wavs = sorted(class_dir.glob("*.wav"))[:40]

        for wav_path in wavs:
            audio, sr = sf.read(str(wav_path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            if sr != args.sample_rate:
                print(f"Resampling {wav_path.name}: {sr} -> {args.sample_rate}")
                audio = resample_audio(audio, sr, args.sample_rate)
                sr = args.sample_rate

            pitch_hz = yin(audio, sr, frame_size=args.yin_frame_size)
            pitch_midi = hz_to_midi(pitch_hz)
            if pitch_midi is None:
                pitch_midi = -1.0

            emb = embeddings[idx]
            emb_str = " ".join(f"{v:.6f}" for v in emb)

            rows.append(
                {
                    "path": str(wav_path.relative_to(args.dataset_root)),
                    "embedding": emb_str,
                    "pitch_midi": float(pitch_midi),
                    "string": int(string),
                    "fret": int(fret),
                    "pos": int(pos),
                }
            )

            idx += 1
            if idx % 100 == 0:
                print(f"Processed {idx}/{total}")

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)

    print("Fusion dataset written to:", args.output_csv)
    print("Total rows:", len(df))


if __name__ == "__main__":
    main()
