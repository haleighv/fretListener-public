#!/usr/bin/env python3
"""
test_disambiguator_live.py - Test disambiguator on live audio.

Simple live audio test for the A/D (or other) disambiguator model.
Captures audio, runs inference, and shows predictions.

Usage:
    python test_disambiguator_live.py --model artifacts/models/disambig_a_d/best.pt
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
import torchaudio

# Add parent directory to path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.disambiguator_model import DisambiguatorCNN


# =============================================================================
# AUDIO CONFIG
# =============================================================================

SAMPLE_RATE = 16000
ATTACK_MS = 250
ATTACK_SAMPLES = int(SAMPLE_RATE * ATTACK_MS / 1000)
TRIGGER_THRESHOLD = 0.02  # RMS threshold to trigger capture
BLOCK_SIZE = 1024

# String pair info
STRING_PAIRS = {
    'E_A': (6, 5, 'Low E', 'A'),
    'A_D': (5, 4, 'A', 'D'),
    'D_G': (4, 3, 'D', 'G'),
    'G_B': (3, 2, 'G', 'B'),
    'B_e': (2, 1, 'B', 'High e'),
}

# MIDI to note name
def midi_to_name(midi):
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return names[midi % 12] + str(midi // 12 - 1)

# Get ambiguous notes for a string pair
def get_ambiguities(string_a, string_b, max_fret=12):
    open_midi = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
    open_a, open_b = open_midi[string_a], open_midi[string_b]
    
    range_a = set(range(open_a, open_a + max_fret + 1))
    range_b = set(range(open_b, open_b + max_fret + 1))
    overlap = sorted(range_a & range_b)
    
    result = {}
    for midi in overlap:
        result[midi] = {
            'fret_a': midi - open_a,
            'fret_b': midi - open_b,
        }
    return result


class LiveDisambiguatorTest:
    def __init__(self, model_path: Path, pair: str = 'A_D', device_idx: int = None):
        self.pair = pair
        self.string_a, self.string_b, self.name_a, self.name_b = STRING_PAIRS[pair]
        self.ambiguities = get_ambiguities(self.string_a, self.string_b)
        
        # Device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        self.model = DisambiguatorCNN(n_mels=64, num_frets=13)
        
        # Initialize classifier by running a dummy forward pass
        dummy_mel = torch.zeros(1, 1, 64, 32)
        dummy_fret = torch.zeros(1, dtype=torch.long)
        with torch.no_grad():
            self.model(dummy_mel, dummy_fret, dummy_fret)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded (val_acc: {checkpoint.get('val_acc', 'N/A'):.4f})")
        
        # Mel transform (on CPU)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=128,
            n_mels=64,
            f_min=55.0,
            f_max=3000.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
        
        # Calculate num_frames
        self.num_frames = (ATTACK_SAMPLES // 128) + 1
        
        # Audio state
        self.audio_device = device_idx
        self.buffer = np.zeros(ATTACK_SAMPLES, dtype=np.float32)
        self.buffer_pos = 0
        self.triggered = False
        self.capture_remaining = 0
        self.captured = None
        
        # Current MIDI note to test (user selects)
        self.current_midi = None
        
        print(f"\nString pair: {self.name_a} vs {self.name_b}")
        print(f"Ambiguous notes: {[midi_to_name(m) for m in self.ambiguities.keys()]}")
    
    def _audio_to_mel(self, audio: np.ndarray) -> torch.Tensor:
        """Convert audio to mel spectrogram."""
        # Peak normalize
        audio = audio.astype(np.float32)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95
        
        # DC removal
        audio = audio - np.mean(audio)
        
        # To tensor
        wav_t = torch.from_numpy(audio).float().unsqueeze(0)
        
        # Mel spectrogram
        mel = self.mel_transform(wav_t)
        mel_db = self.db_transform(mel)
        
        # Normalize
        mean = mel_db.mean(dim=2, keepdim=True)
        std = mel_db.std(dim=2, keepdim=True) + 1e-6
        mel_db = (mel_db - mean) / std
        
        # Pad/truncate
        if mel_db.size(2) < self.num_frames:
            pad = self.num_frames - mel_db.size(2)
            mel_db = torch.nn.functional.pad(mel_db, (0, pad))
        elif mel_db.size(2) > self.num_frames:
            mel_db = mel_db[:, :, :self.num_frames]
        
        return mel_db.unsqueeze(0).to(self.device)
    
    def predict(self, audio: np.ndarray, midi: int) -> dict:
        """Run prediction on audio for a specific MIDI note."""
        if midi not in self.ambiguities:
            return {'error': f'MIDI {midi} not ambiguous for {self.pair}'}
        
        info = self.ambiguities[midi]
        fret_a = info['fret_a']
        fret_b = info['fret_b']
        
        mel = self._audio_to_mel(audio)
        fret_a_t = torch.tensor([fret_a], dtype=torch.long).to(self.device)
        fret_b_t = torch.tensor([fret_b], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            logits = self.model(mel, fret_a_t, fret_b_t)
            probs = torch.softmax(logits, dim=1)
        
        prob_a = probs[0, 0].item()
        prob_b = probs[0, 1].item()
        
        predicted = self.name_a if prob_a > prob_b else self.name_b
        predicted_fret = fret_a if prob_a > prob_b else fret_b
        
        return {
            'midi': midi,
            'note': midi_to_name(midi),
            'predicted_string': predicted,
            'predicted_fret': predicted_fret,
            'prob_a': prob_a,
            'prob_b': prob_b,
            'confidence': max(prob_a, prob_b),
        }
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio input callback."""
        if status:
            print(f"Audio status: {status}")
        
        audio = indata[:, 0].astype(np.float32)
        rms = np.sqrt(np.mean(audio ** 2))
        
        if not self.triggered:
            if rms > TRIGGER_THRESHOLD:
                self.triggered = True
                self.capture_remaining = ATTACK_SAMPLES
                self.captured = np.zeros(ATTACK_SAMPLES, dtype=np.float32)
                self.buffer_pos = 0
                print("  [TRIGGERED]", end='', flush=True)
        
        if self.triggered:
            # Copy audio to buffer
            n = min(len(audio), self.capture_remaining)
            self.captured[self.buffer_pos:self.buffer_pos + n] = audio[:n]
            self.buffer_pos += n
            self.capture_remaining -= n
            
            if self.capture_remaining <= 0:
                self.triggered = False
                # Run prediction
                if self.current_midi is not None:
                    result = self.predict(self.captured, self.current_midi)
                    if 'error' not in result:
                        print(f"\r  → {result['note']}: {result['predicted_string']} string, "
                              f"fret {result['predicted_fret']} "
                              f"({result['confidence']*100:.1f}% confident)    ")
                    else:
                        print(f"\r  Error: {result['error']}")
    
    def run(self):
        """Run interactive test loop."""
        print("\n" + "="*60)
        print("LIVE DISAMBIGUATOR TEST")
        print("="*60)
        print(f"\nTesting: {self.name_a} vs {self.name_b} string disambiguation")
        print("\nCommands:")
        print("  <midi_number>  - Set MIDI note to test (e.g., 50, 51, 52)")
        print("  <note_name>    - Set by name (e.g., D3, E3)")
        print("  list           - Show ambiguous notes")
        print("  q              - Quit")
        print("\nPlay a note after setting the MIDI number.")
        print("="*60)
        
        # Note name to MIDI lookup
        name_to_midi = {midi_to_name(m): m for m in self.ambiguities.keys()}
        
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                blocksize=BLOCK_SIZE,
                device=self.audio_device,
                callback=self.audio_callback
            ):
                while True:
                    try:
                        cmd = input("\nMIDI note> ").strip()
                    except EOFError:
                        break
                    
                    if cmd.lower() == 'q':
                        break
                    elif cmd.lower() == 'list':
                        print("\nAmbiguous notes for this pair:")
                        for midi, info in self.ambiguities.items():
                            name = midi_to_name(midi)
                            print(f"  {midi} ({name}): {self.name_a} fret {info['fret_a']} vs "
                                  f"{self.name_b} fret {info['fret_b']}")
                    elif cmd.upper() in name_to_midi:
                        self.current_midi = name_to_midi[cmd.upper()]
                        info = self.ambiguities[self.current_midi]
                        print(f"Testing MIDI {self.current_midi} ({cmd.upper()}): "
                              f"{self.name_a} fret {info['fret_a']} vs {self.name_b} fret {info['fret_b']}")
                        print("Play the note...")
                    elif cmd.isdigit():
                        midi = int(cmd)
                        if midi in self.ambiguities:
                            self.current_midi = midi
                            info = self.ambiguities[midi]
                            print(f"Testing MIDI {midi} ({midi_to_name(midi)}): "
                                  f"{self.name_a} fret {info['fret_a']} vs {self.name_b} fret {info['fret_b']}")
                            print("Play the note...")
                        else:
                            print(f"MIDI {midi} is not ambiguous for {self.pair}. Use 'list' to see options.")
                    elif cmd:
                        print("Unknown command. Use a MIDI number, note name, 'list', or 'q'.")
        
        except KeyboardInterrupt:
            print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(description="Test disambiguator on live audio")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint (best.pt)')
    parser.add_argument('--pair', type=str, default='A_D',
                       choices=list(STRING_PAIRS.keys()),
                       help='String pair (default: A_D)')
    parser.add_argument('--device', type=int, default=None,
                       help='Audio input device index')
    parser.add_argument('--list-devices', action='store_true',
                       help='List audio devices and exit')
    
    args = parser.parse_args()
    
    if args.list_devices:
        print(sd.query_devices())
        return
    
    tester = LiveDisambiguatorTest(
        model_path=Path(args.model),
        pair=args.pair,
        device_idx=args.device,
    )
    tester.run()


if __name__ == '__main__':
    main()
