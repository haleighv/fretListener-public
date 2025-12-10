import argparse
from pathlib import Path
import torch
import torchaudio
import numpy as np
from attack_model import AttackPositionNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("mps")

    model = AttackPositionNet()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model_state"])
    model.to(device).eval()

    mel = torchaudio.transforms.MelSpectrogram(48000, 1024, 256, 64, 55, 2000)
    db = torchaudio.transforms.AmplitudeToDB()

    embeddings = []
    labels = []

    for d in sorted(args.dataset_root.iterdir()):
        if not d.is_dir(): continue
        pos = int(d.name.split("_f")[0][1:]) * 8 + int(d.name.split("_f")[1])
        for w in sorted(d.glob("*.wav"))[:40]:
            wav, _ = torchaudio.load(w)
            wav = wav.mean(0, keepdim=True)
            wav = wav[:, :9120]

            spec = db(mel(wav))[:, :, :32]
            spec = spec.unsqueeze(0).to(device)

            with torch.no_grad():
                _, emb = model(spec, return_embedding=True)

            embeddings.append(emb.cpu().numpy()[0])
            labels.append(pos)

    np.savez(args.output, embeddings=np.array(embeddings), pos=np.array(labels))
    print("Saved:", args.output)

if __name__ == "__main__":
    main()
