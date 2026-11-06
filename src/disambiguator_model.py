#!/usr/bin/env python3
"""
disambiguator_model.py - String Disambiguation Models

Triplet disambiguators for 3-way string disambiguation when the note model
has identified an ambiguous MIDI pitch that could be on 3 different strings.

Architecture:
- CNN backbone extracts timbral features from mel spectrogram
- 3-class classifier outputs probability for each string position

Example for E_A_D triplet:
- Input: mel spectrogram of detected note (e.g., D3 which could be S6F10, S5F5, or S4F0)
- Output: [prob_S6, prob_S5, prob_S4]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletDisambiguatorCNN(nn.Module):
    """
    3-way string disambiguation model.
    
    Classifies audio into one of 3 possible string positions for ambiguous notes.
    
    Args:
        n_mels: Number of mel frequency bins (default 48, matching note model)
        dropout_p: Dropout probability (default 0.3)
    """
    
    def __init__(
        self,
        n_mels: int = 48,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        
        self.n_mels = n_mels
        self.dropout_p = dropout_p
        
        # CNN backbone
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(64, 3),  # 3 classes: string_a, string_b, string_c
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Mel spectrogram [batch, 1, n_mels, num_frames]
        
        Returns:
            Logits [batch, 3] for [string_a, string_b, string_c]
        """
        features = self.features(x)
        features = self.gap(features)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get softmax probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


# Legacy model for backward compatibility
class DisambiguatorCNN(nn.Module):
    """
    DEPRECATED: Binary string disambiguation model.
    
    Kept for backward compatibility with old checkpoints.
    Use TripletDisambiguatorCNN for new training.
    """
    
    def __init__(
        self,
        n_mels: int = 48,
        num_frets: int = 13,
        embed_dim: int = 8,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        
        self.n_mels = n_mels
        self.num_frets = num_frets
        self.embed_dim = embed_dim
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.fret_embed_a = nn.Embedding(num_frets, embed_dim)
        self.fret_embed_b = nn.Embedding(num_frets, embed_dim)
        
        self._cnn_output_size = None
        self.classifier = None
        self.dropout_p = dropout_p
        
    def _init_classifier(self, cnn_size: int):
        input_size = cnn_size + 2 * self.embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
            nn.Linear(64, 2),
        )
        device = next(self.features.parameters()).device
        self.classifier = self.classifier.to(device)
    
    def forward(self, x: torch.Tensor, fret_a: torch.Tensor, fret_b: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        features = self.features(x)
        features = features.view(batch_size, -1)
        
        if self.classifier is None:
            self._cnn_output_size = features.size(1)
            self._init_classifier(self._cnn_output_size)
        
        embed_a = self.fret_embed_a(fret_a)
        embed_b = self.fret_embed_b(fret_b)
        combined = torch.cat([features, embed_a, embed_b], dim=1)
        logits = self.classifier(combined)
        return logits
    
    def predict_proba(self, x: torch.Tensor, fret_a: torch.Tensor, fret_b: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x, fret_a, fret_b)
        return F.softmax(logits, dim=1)


# =============================================================================
# AMBIGUITY MAPPING
# =============================================================================

# String numbers (1-indexed as in data) to open MIDI
# String 1 = high E (MIDI 64 open), String 6 = low E (MIDI 40 open)
STRING_OPEN_MIDI = {
    1: 64,  # high e
    2: 59,  # B
    3: 55,  # G
    4: 50,  # D
    5: 45,  # A
    6: 40,  # low E
}


def get_ambiguous_positions(midi_note: int, max_fret: int = 12) -> list:
    """
    Get all possible (string_num, fret) positions for a MIDI note.
    
    Returns list of (string_num, fret) tuples where string_num is 1-indexed.
    """
    positions = []
    for string_num, open_midi in STRING_OPEN_MIDI.items():
        fret = midi_note - open_midi
        if 0 <= fret <= max_fret:
            positions.append((string_num, fret))
    return positions


def get_triplet_ambiguities(string_a: int, string_b: int, string_c: int, max_fret: int = 12) -> dict:
    """
    Get all MIDI notes that are ambiguous between three strings.
    
    Args:
        string_a: First string number (1-indexed, lowest pitch = highest fret)
        string_b: Second string number (1-indexed, middle)
        string_c: Third string number (1-indexed, highest pitch = lowest fret)
        max_fret: Maximum fret in dataset
    
    Returns:
        Dict mapping MIDI note -> [(string_a, fret_a), (string_b, fret_b), (string_c, fret_c)]
    """
    open_a = STRING_OPEN_MIDI[string_a]
    open_b = STRING_OPEN_MIDI[string_b]
    open_c = STRING_OPEN_MIDI[string_c]
    
    # MIDI range for each string
    range_a = set(range(open_a, open_a + max_fret + 1))
    range_b = set(range(open_b, open_b + max_fret + 1))
    range_c = set(range(open_c, open_c + max_fret + 1))
    
    # Find notes that exist on all three strings
    ambiguous = range_a & range_b & range_c
    
    result = {}
    for midi in sorted(ambiguous):
        fret_a = midi - open_a
        fret_b = midi - open_b
        fret_c = midi - open_c
        result[midi] = [(string_a, fret_a), (string_b, fret_b), (string_c, fret_c)]
    
    return result


# Triplet ambiguities - each covers 3 adjacent strings
ALL_TRIPLETS = {
    'E_A_D': get_triplet_ambiguities(6, 5, 4, max_fret=12),  # low E, A, D
    'A_D_G': get_triplet_ambiguities(5, 4, 3, max_fret=12),  # A, D, G
    'D_G_B': get_triplet_ambiguities(4, 3, 2, max_fret=12),  # D, G, B
    'G_B_e': get_triplet_ambiguities(3, 2, 1, max_fret=12),  # G, B, high e
}

# Triplet string mappings: triplet_name -> (string_a, string_b, string_c)
TRIPLET_STRINGS = {
    'E_A_D': (6, 5, 4),
    'A_D_G': (5, 4, 3),
    'D_G_B': (4, 3, 2),
    'G_B_e': (3, 2, 1),
}


# Legacy pair ambiguities (kept for backward compatibility)
def get_string_pair_ambiguities(string_a: int, string_b: int, max_fret: int = 12) -> dict:
    """DEPRECATED: Use get_triplet_ambiguities instead."""
    open_a = STRING_OPEN_MIDI[string_a]
    open_b = STRING_OPEN_MIDI[string_b]
    range_a = set(range(open_a, open_a + max_fret + 1))
    range_b = set(range(open_b, open_b + max_fret + 1))
    ambiguous = range_a & range_b
    result = {}
    for midi in sorted(ambiguous):
        fret_a = midi - open_a
        fret_b = midi - open_b
        result[midi] = [(string_a, fret_a), (string_b, fret_b)]
    return result


# Legacy pairs (kept for backward compatibility with old models)
ALL_PAIRS = {
    'E_A': get_string_pair_ambiguities(6, 5, max_fret=12),
    'A_D': get_string_pair_ambiguities(5, 4, max_fret=12),
    'D_G': get_string_pair_ambiguities(4, 3, max_fret=12),
    'G_B': get_string_pair_ambiguities(3, 2, max_fret=12),
    'B_e': get_string_pair_ambiguities(2, 1, max_fret=12),
}


if __name__ == "__main__":
    # Test triplet model
    print("Testing TripletDisambiguatorCNN...")
    model = TripletDisambiguatorCNN(n_mels=64)
    
    batch_size = 4
    x = torch.randn(batch_size, 1, 64, 32)
    
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output: {logits}")
    
    probs = model.predict_proba(x)
    print(f"Probabilities: {probs}")
    
    # Print triplet ambiguities
    print("\n" + "="*60)
    print("TRIPLET AMBIGUITIES")
    print("="*60)
    
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    for triplet_name, triplet_info in ALL_TRIPLETS.items():
        print(f"\n{triplet_name}:")
        for midi, positions in triplet_info.items():
            name = note_names[midi % 12] + str(midi // 12 - 1)
            pos_str = " vs ".join([f"S{s}F{f}" for s, f in positions])
            print(f"  MIDI {midi} ({name}): {pos_str}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
