#!/usr/bin/env python3
"""
common/fretboard.py - Tkinter fretboard widget for guitar visualization.

Display convention:
- Top of screen = high E (string 1)
- Bottom of screen = low E (string 6)
- 12 frets displayed by default
"""

import tkinter as tk
from typing import Tuple, Set, Optional

from .guitar import STRING_NAMES_TOP_TO_BOTTOM, STANDARD_TUNING_LIST


class GuitarFretboard(tk.Canvas):
    """
    Visual guitar fretboard widget showing notes.
    
    Display layout:
    - Row 0 (top) = E4 (high E, string 1)
    - Row 5 (bottom) = E2 (low E, string 6)
    
    String indexing:
    - string_idx 0 = E2 (low E, bottom of display)
    - string_idx 5 = E4 (high E, top of display)
    
    String numbers (guitarist convention):
    - string_num 1 = E4 (high E)
    - string_num 6 = E2 (low E)
    """
    
    # Standard tuning MIDI values (index 0 = low E)
    STANDARD_TUNING = STANDARD_TUNING_LIST
    
    # Display order: index 0 = top of screen (high E)
    STRING_NAMES_TOP_TO_BOTTOM = STRING_NAMES_TOP_TO_BOTTOM

    def __init__(self, parent, frets: int = 12, **kwargs):
        super().__init__(parent, **kwargs)
        self.frets = int(frets)
        self.active_notes: Set[Tuple[int, int]] = set()

        self.string_spacing = 28
        self.fret_spacing = 50
        self.dot_radius = 10

        width = self.fret_spacing * (self.frets + 1) + 80
        height = self.string_spacing * 7 + 40

        self.config(width=width, height=height, bg="#2b2b2b", highlightthickness=0)
        self.draw_fretboard()

    def draw_fretboard(self):
        """Draw the fretboard with frets, strings, and position markers."""
        self.delete("all")

        # Draw frets
        for fret in range(self.frets + 1):
            x = 45 + fret * self.fret_spacing
            y_start = 25
            y_end = 25 + self.string_spacing * 5

            if fret == 0:
                # Nut (thicker)
                self.create_line(x, y_start, x, y_end, width=5, fill="#d4a574")
            else:
                # Regular fret
                self.create_line(x, y_start, x, y_end, width=2, fill="#8b7355")

        # Fret markers (between frets, not on them)
        single_dot_frets = [3, 5, 7, 9, 15, 17, 19, 21]
        double_dot_frets = [12, 24]
        
        for fret in single_dot_frets:
            if fret <= self.frets:
                x = 45 + fret * self.fret_spacing - self.fret_spacing // 2
                marker_y = 25 + self.string_spacing * 2.5
                self.create_oval(x - 4, marker_y - 4, x + 4, marker_y + 4,
                               fill="#555", outline="")
        
        for fret in double_dot_frets:
            if fret <= self.frets:
                x = 45 + fret * self.fret_spacing - self.fret_spacing // 2
                marker_y1 = 25 + self.string_spacing * 1.5
                marker_y2 = 25 + self.string_spacing * 3.5
                self.create_oval(x - 4, marker_y1 - 4, x + 4, marker_y1 + 4,
                               fill="#555", outline="")
                self.create_oval(x - 4, marker_y2 - 4, x + 4, marker_y2 + 4,
                               fill="#555", outline="")

        # Draw strings - top to bottom: E4(thin) -> E2(thick)
        for s in range(6):
            y = 25 + s * self.string_spacing
            x_start = 45
            x_end = 45 + self.frets * self.fret_spacing

            # s=0 is top (E4, thinnest), s=5 is bottom (E2, thickest)
            thickness = 1 + s * 0.4
            self.create_line(x_start, y, x_end, y, width=thickness, fill="#c0c0c0")

            # String label
            self.create_text(18, y, text=self.STRING_NAMES_TOP_TO_BOTTOM[s],
                           font=("Arial", 9), fill="#888")

    def clear_notes(self):
        """Clear all active note markers."""
        self.delete("note")
        self.active_notes.clear()

    def show_note(self, string_num: int, fret: int, color: str = "#ff6b6b"):
        """
        Show a note at given position.
        
        Args:
            string_num: 1-6 (1=high E, 6=low E) - guitarist convention
            fret: 0-N
            color: Fill color for the note marker
        """
        self.clear_notes()
        # Convert string_num (1-6) to display row (0-5)
        # string_num 1 (high E) -> display row 0 (top)
        # string_num 6 (low E) -> display row 5 (bottom)
        display_row = string_num - 1
        self._draw_note(display_row, fret, color)

    def show_note_by_idx(self, string_idx: int, fret: int, color: str = "#ff6b6b"):
        """
        Show a note using internal string index.
        
        Args:
            string_idx: 0-5 (0=low E, 5=high E) - internal index
            fret: 0-N
            color: Fill color for the note marker
        """
        self.clear_notes()
        # Convert string_idx (0-5) to display row (0-5)
        # string_idx 0 (low E) -> display row 5 (bottom)
        # string_idx 5 (high E) -> display row 0 (top)
        display_row = 5 - string_idx
        self._draw_note(display_row, fret, color)

    def add_note(self, string_num: int, fret: int, color: str = "#ff6b6b", tag: str = "note"):
        """
        Add a note marker without clearing existing ones.
        
        Args:
            string_num: 1-6 (1=high E, 6=low E)
            fret: 0-N
            color: Fill color
            tag: Canvas tag for the marker
        """
        display_row = string_num - 1
        self._draw_note(display_row, fret, color, tag)
        self.active_notes.add((string_num, fret))

    def _draw_note(self, display_row: int, fret: int, color: str = "#ff6b6b", tag: str = "note"):
        """
        Draw a note marker at display position.
        
        Args:
            display_row: 0-5 (0=top of screen, 5=bottom)
            fret: 0-N
            color: Fill color
            tag: Canvas tag
        """
        if fret < 0 or fret > self.frets:
            return
        if display_row < 0 or display_row >= 6:
            return

        y = 25 + display_row * self.string_spacing
        
        if fret == 0:
            # Open string - marker at nut
            x = 45
        else:
            # Fretted note - marker between frets
            x = 45 + fret * self.fret_spacing - self.fret_spacing // 2

        # Determine outline color based on fill
        outline_color = self._darken_color(color)
        
        self.create_oval(
            x - self.dot_radius, y - self.dot_radius,
            x + self.dot_radius, y + self.dot_radius,
            fill=color, outline=outline_color, width=2, tags=tag
        )

    def _darken_color(self, hex_color: str) -> str:
        """Darken a hex color for outline."""
        try:
            hex_color = hex_color.lstrip('#')
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            r = max(0, int(r * 0.7))
            g = max(0, int(g * 0.7))
            b = max(0, int(b * 0.7))
            
            return f"#{r:02x}{g:02x}{b:02x}"
        except:
            return "#cc0000"

    def highlight_scale(self, positions: list, root_color: str = "#ff6b6b", 
                       note_color: str = "#4fc3f7"):
        """
        Highlight multiple positions for scale display.
        
        Args:
            positions: List of (string_num, fret, is_root) tuples
            root_color: Color for root notes
            note_color: Color for other scale notes
        """
        self.clear_notes()
        for pos in positions:
            if len(pos) >= 3:
                string_num, fret, is_root = pos[0], pos[1], pos[2]
            else:
                string_num, fret = pos[0], pos[1]
                is_root = False
            
            color = root_color if is_root else note_color
            self.add_note(string_num, fret, color=color)

    def get_note_at_position(self, string_idx: int, fret: int) -> int:
        """
        Get MIDI note number at a position.
        
        Args:
            string_idx: 0-5 (0=low E, 5=high E)
            fret: 0-N
        
        Returns:
            MIDI note number
        """
        return self.STANDARD_TUNING[string_idx] + fret
