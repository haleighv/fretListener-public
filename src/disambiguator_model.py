#!/usr/bin/env python3
"""
disambiguator_model.py - String Disambiguation Model

A small CNN for disambiguating between two adjacent strings when the
note model has identified an ambiguous MIDI pitch.

Architecture:
- CNN backbone extracts timbral features from mel spectrogram
- Fret position embeddings provide context about which positions we're deciding between
- Binary classifier outputs probability for each string

Example for A/D disambiguator:
- Input: mel spectrogram of detected note + fret positions (e.g., A-fret5 vs D-open)
- Output: [prob_A_string, prob_D_string]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DisambiguatorCNN(nn.Module):
    """
    String disambiguation model with fret position conditioning.
    
    Args:
        n_mels: Number of mel frequency bins (default 64)
        num_frets: Maximum fret number + 1 (default 13 for frets 0-12)
        embed_dim: Dimension of fret embeddings (default 8)
        dropout_p: Dropout probability (default 0.3)
    """
    
    def __init__(
        self,
        n_mels: int = 64,
        num_frets: int = 13,
        embed_dim: int = 8,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        
        self.n_mels = n_mels
        self.num_frets = num_frets
        self.embed_dim = embed_dim
        
        # CNN backbone - smaller than note model since task is simpler
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
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Fret embeddings - one for each string position
        self.fret_embed_a = nn.Embedding(num_frets, embed_dim)
        self.fret_embed_b = nn.Embedding(num_frets, embed_dim)
        
        # Calculate CNN output size dynamically
        self._cnn_output_size = None
        
        # Classifier - will be initialized after first forward pass
        self.classifier = None
        self.dropout_p = dropout_p
        
    def _get_cnn_output_size(self, n_mels: int, num_frames: int) -> int:
        """Calculate flattened CNN output size."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, num_frames)
            out = self.features(dummy)
            return out.view(1, -1).size(1)
    
    def _init_classifier(self, cnn_size: int):
        """Initialize classifier layers."""
        # CNN features + 2 fret embeddings
        input_size = cnn_size + 2 * self.embed_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
            nn.Linear(64, 2),  # Binary: string A vs string B
        )
        
        # Move to same device as features
        device = next(self.features.parameters()).device
        self.classifier = self.classifier.to(device)
    
    def forward(self, x: torch.Tensor, fret_a: torch.Tensor, fret_b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Mel spectrogram [batch, 1, n_mels, num_frames]
            fret_a: Fret position on string A [batch] (LongTensor)
            fret_b: Fret position on string B [batch] (LongTensor)
        
        Returns:
            Logits [batch, 2] for [string_A, string_B]
        """
        batch_size = x.size(0)
        
        # CNN features
        features = self.features(x)
        features = features.view(batch_size, -1)
        
        # Initialize classifier on first forward pass
        if self.classifier is None:
            self._cnn_output_size = features.size(1)
            self._init_classifier(self._cnn_output_size)
        
        # Fret embeddings
        embed_a = self.fret_embed_a(fret_a)  # [batch, embed_dim]
        embed_b = self.fret_embed_b(fret_b)  # [batch, embed_dim]
        
        # Concatenate CNN features + fret embeddings
        combined = torch.cat([features, embed_a, embed_b], dim=1)
        
        # Classify
        logits = self.classifier(combined)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor, fret_a: torch.Tensor, fret_b: torch.Tensor) -> torch.Tensor:
        """Get softmax probabilities."""
        logits = self.forward(x, fret_a, fret_b)
        return F.softmax(logits, dim=1)


class DisambiguatorSimple(nn.Module):
    """
    Simplified disambiguator without fret embeddings.
    
    Use this as a baseline to compare against the fret-conditioned model.
    """
    
    def __init__(
        self,
        n_mels: int = 64,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        
        self.n_mels = n_mels
        
        # Same CNN backbone
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
        
        self.classifier = None
        self.dropout_p = dropout_p
        
    def _init_classifier(self, cnn_size: int):
        """Initialize classifier layers."""
        self.classifier = nn.Sequential(
            nn.Linear(cnn_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
            nn.Linear(64, 2),
        )
        device = next(self.features.parameters()).device
        self.classifier = self.classifier.to(device)
    
    def forward(self, x: torch.Tensor, fret_a: torch.Tensor = None, fret_b: torch.Tensor = None) -> torch.Tensor:
        """Forward pass (fret args ignored for API compatibility)."""
        batch_size = x.size(0)
        
        features = self.features(x)
        features = features.view(batch_size, -1)
        
        if self.classifier is None:
            self._init_classifier(features.size(1))
        
        return self.classifier(features)


# =============================================================================
# AMBIGUITY MAPPING
# =============================================================================

# String numbers (1-indexed as in data) to internal index (0-indexed)
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


def get_string_pair_ambiguities(string_a: int, string_b: int, max_fret: int = 12) -> dict:
    """
    Get all MIDI notes that are ambiguous between two strings.
    
    Args:
        string_a: First string number (1-indexed)
        string_b: Second string number (1-indexed)
        max_fret: Maximum fret in dataset
    
    Returns:
        Dict mapping MIDI note -> [(string_a, fret_a), (string_b, fret_b)]
    """
    open_a = STRING_OPEN_MIDI[string_a]
    open_b = STRING_OPEN_MIDI[string_b]
    
    # MIDI range for each string
    range_a = set(range(open_a, open_a + max_fret + 1))
    range_b = set(range(open_b, open_b + max_fret + 1))
    
    # Find overlap
    ambiguous = range_a & range_b
    
    result = {}
    for midi in sorted(ambiguous):
        fret_a = midi - open_a
        fret_b = midi - open_b
        result[midi] = [(string_a, fret_a), (string_b, fret_b)]
    
    return result


# A/D string ambiguities (strings 5 and 4)
A_D_AMBIGUITIES = get_string_pair_ambiguities(5, 4, max_fret=12)

# E/A string ambiguities (strings 6 and 5)  
E_A_AMBIGUITIES = get_string_pair_ambiguities(6, 5, max_fret=12)

# All string pair ambiguities
ALL_PAIRS = {
    'E_A': get_string_pair_ambiguities(6, 5, max_fret=12),  # low E vs A
    'A_D': get_string_pair_ambiguities(5, 4, max_fret=12),  # A vs D
    'D_G': get_string_pair_ambiguities(4, 3, max_fret=12),  # D vs G
    'G_B': get_string_pair_ambiguities(3, 2, max_fret=12),  # G vs B
    'B_e': get_string_pair_ambiguities(2, 1, max_fret=12),  # B vs high e
}


if __name__ == "__main__":
    # Test model creation
    print("Testing DisambiguatorCNN...")
    model = DisambiguatorCNN(n_mels=64, num_frets=13)
    
    # Dummy input
    batch_size = 4
    x = torch.randn(batch_size, 1, 64, 96)
    fret_a = torch.randint(0, 8, (batch_size,))
    fret_b = torch.randint(0, 8, (batch_size,))
    
    # Forward pass
    logits = model(x, fret_a, fret_b)
    print(f"Input shape: {x.shape}")
    print(f"Fret A: {fret_a.tolist()}")
    print(f"Fret B: {fret_b.tolist()}")
    print(f"Output shape: {logits.shape}")
    print(f"Output: {logits}")
    
    probs = model.predict_proba(x, fret_a, fret_b)
    print(f"Probabilities: {probs}")
    
    # Print A/D ambiguities
    print("\nA/D String Ambiguities:")
    for midi, positions in A_D_AMBIGUITIES.items():
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        name = note_names[midi % 12] + str(midi // 12 - 1)
        pos_str = " vs ".join([f"S{s}F{f}" for s, f in positions])
        print(f"  MIDI {midi} ({name}): {pos_str}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
