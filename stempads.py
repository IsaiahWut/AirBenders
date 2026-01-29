import cv2 as cv
import math

class StemPad:
    """
    Stem pad button for controlling individual stems
    """
    def __init__(self, center, width, height, stem_type, track_index, label=None):
        self.cx, self.cy = center
        self.width = width
        self.height = height
        self.stem_type = stem_type
        self.track_index = track_index
        self.label = label or stem_type.upper()[:3]  # First 3 letters
        
        # Button state
        self.is_enabled = False
        self.was_pinching = False
        
        # Colors
        self.color_inactive = (60, 60, 60)
        self.color_active = (0, 200, 0)  # Green when active
        self.color_border = (0, 0, 0)
        self.color_text = (255, 255, 255)
        
    def get_bounds(self):
        """Return button bounds (x1, y1, x2, y2)"""
        x1 = self.cx - self.width // 2
        y1 = self.cy - self.height // 2
        x2 = self.cx + self.width // 2
        y2 = self.cy + self.height // 2
        return (x1, y1, x2, y2)
    
    def contains(self, x, y):
        """Check if point (x, y) is inside the button"""
        x1, y1, x2, y2 = self.get_bounds()
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def update(self, pinch_positions, is_enabled):
        """
        Update button state
        
        Args:
            pinch_positions: List of (x, y) pinch positions
            is_enabled: Current stem enabled state from music module
        """
        self.is_enabled = is_enabled
        
        # Check if being pinched
        is_pinching_now = any(self.contains(px, py) for px, py in pinch_positions)
        
        # Detect new pinch (edge trigger)
        newly_pinched = is_pinching_now and not self.was_pinching
        
        self.was_pinching = is_pinching_now
        
        return newly_pinched
    
    def draw(self, frame):
        """Draw the stem pad button"""
        x1, y1, x2, y2 = self.get_bounds()
        
        # Choose color based on state
        color = self.color_active if self.is_enabled else self.color_inactive
        
        # Draw filled rectangle
        cv.rectangle(frame, (x1, y1), (x2, y2), color, cv.FILLED)
        
        # Draw border
        border_color = (0, 255, 0) if self.is_enabled else self.color_border
        cv.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)
        
        # Draw label text
        text_size = cv.getTextSize(self.label, cv.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
        text_x = self.cx - text_size[0] // 2
        text_y = self.cy + text_size[1] // 2
        
        cv.putText(frame, self.label, (text_x, text_y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.35, self.color_text, 1)


class StemPadBank:
    """
    Bank of stem pads that DYNAMICALLY shows all available stems
    """
    def __init__(self, position, track_index, deck_label="DECK"):
        """
        Args:
            position: (x, y) center position for the bank
            track_index: Which track this bank controls
            deck_label: Label for this deck (e.g., "LEFT" or "RIGHT")
        """
        self.x, self.y = position
        self.track_index = track_index
        self.deck_label = deck_label
        
        # Button dimensions
        self.button_width = 50
        self.button_height = 30
        self.spacing = 8
        
        # Pads will be created dynamically based on available stems
        self.pads = []
        self.current_stems = []
        
        # Stem type to label mapping
        self.stem_labels = {
            'drums': 'DRM',
            'vocals': 'VOX',
            'bass': 'BAS',
            'other': 'OTH',
            'instrumental': 'INST'
        }
    
    def _create_pads(self, available_stems):
        """Create pads based on available stems"""
        # Sort stems in a logical order
        stem_order = ['drums', 'bass', 'vocals', 'other', 'instrumental']
        sorted_stems = [s for s in stem_order if s in available_stems]
        # Add any stems not in the predefined order
        for s in available_stems:
            if s not in sorted_stems:
                sorted_stems.append(s)
        
        self.pads = []
        num_stems = len(sorted_stems)
        
        if num_stems == 0:
            return
        
        # Calculate positions (horizontal layout)
        total_width = num_stems * self.button_width + (num_stems - 1) * self.spacing
        start_x = self.x - total_width // 2 + self.button_width // 2
        
        for i, stem_type in enumerate(sorted_stems):
            pad_x = start_x + i * (self.button_width + self.spacing)
            label = self.stem_labels.get(stem_type, stem_type.upper()[:4])
            
            pad = StemPad(
                center=(pad_x, self.y),
                width=self.button_width,
                height=self.button_height,
                stem_type=stem_type,
                track_index=self.track_index,
                label=label
            )
            self.pads.append(pad)
        
        self.current_stems = sorted_stems
    
    def update(self, pinch_positions, stem_states):
        """
        Update all pads in the bank
        
        Args:
            pinch_positions: List of (x, y) pinch positions
            stem_states: Dict of {stem_type: enabled} from music module
            
        Returns:
            List of stem_types that were just pinched
        """
        # Check if we need to recreate pads (stems changed)
        available_stems = list(stem_states.keys())
        if set(available_stems) != set(self.current_stems):
            self._create_pads(available_stems)
        
        events = []
        
        for pad in self.pads:
            is_enabled = stem_states.get(pad.stem_type, False)
            newly_pinched = pad.update(pinch_positions, is_enabled)
            
            if newly_pinched:
                events.append(pad.stem_type)
        
        return events
    
    def draw(self, frame):
        """Draw all pads in the bank"""
        if not self.pads:
            return
        
        # Draw label above pads
        label_y = self.y - 22
        label_text = f"{self.deck_label} STEMS ({len(self.pads)})"
        cv.putText(frame, label_text, (self.x - 50, label_y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw all pads
        for pad in self.pads:
            pad.draw(frame)