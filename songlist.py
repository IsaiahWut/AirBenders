import cv2 as cv

class SongList:
    def __init__(self, songs, position=(0, 0), width=250, item_height=30):
        self.songs = songs
        self.x, self.y = position
        self.width = width
        self.item_height = item_height
        
        # Collapsed state
        self.is_collapsed = False
        self.collapsed_width = 100
        self.collapsed_height = 50
        
        # Dragging state
        self.is_dragging = False
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self.was_pinching_header = False
        self.drag_threshold = 10  # Minimum movement to be considered a drag
        self.initial_pinch_x = 0
        self.initial_pinch_y = 0
        
        # Header height (for dragging and collapsing)
        self.header_height = 35

    def get_header_bounds(self):
        """Return the bounds of the header (for dragging and collapsing)"""
        if self.is_collapsed:
            return (self.x, self.y, self.x + self.collapsed_width, self.y + self.collapsed_height)
        else:
            return (self.x, self.y, self.x + self.width, self.y + self.header_height)

    def check_header_pinch(self, pinch_positions):
        """Check if user is pinching the header"""
        x1, y1, x2, y2 = self.get_header_bounds()
        for px, py in pinch_positions:
            if x1 <= px <= x2 and y1 <= py <= y2:
                return True, px, py
        return False, None, None

    def update(self, pinch_positions):
        """Update dragging and toggle collapse state"""
        has_pinch = len(pinch_positions) > 0
        is_pinching_header, px, py = self.check_header_pinch(pinch_positions)
        
        if is_pinching_header and not self.was_pinching_header:
            # Just started pinching header
            self.initial_pinch_x = px
            self.initial_pinch_y = py
            self.drag_offset_x = px - self.x
            self.drag_offset_y = py - self.y
        
        if is_pinching_header and has_pinch:
            # Check if we've moved enough to be considered dragging
            if not self.is_dragging:
                distance_moved = ((px - self.initial_pinch_x)**2 + (py - self.initial_pinch_y)**2)**0.5
                if distance_moved > self.drag_threshold:
                    self.is_dragging = True
            
            # If dragging, update position
            if self.is_dragging:
                self.x = px - self.drag_offset_x
                self.y = py - self.drag_offset_y
        
        # Check if we released the pinch
        if self.was_pinching_header and not is_pinching_header:
            # Released the header
            if self.is_dragging:
                # We were dragging, now stop
                self.is_dragging = False
            else:
                # Quick tap without drag = toggle collapse
                self.is_collapsed = not self.is_collapsed
        
        self.was_pinching_header = is_pinching_header

    def draw(self, frame, highlight_index=None):
        """Draw the song list or collapsed box"""
        if self.is_collapsed:
            # Draw collapsed box
            box_x1, box_y1, box_x2, box_y2 = self.get_header_bounds()
            
            # Background
            cv.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (70, 70, 70), cv.FILLED)
            cv.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), 2)
            
            # Draw text
            text_x = box_x1 + 10
            text_y = box_y1 + 30
            cv.putText(frame, "SONGS", (text_x, text_y), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw expand indicator
            cv.putText(frame, "+", (box_x2 - 25, text_y), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # Draw header
            header_x1, header_y1, header_x2, header_y2 = self.get_header_bounds()
            cv.rectangle(frame, (header_x1, header_y1), (header_x2, header_y2), (80, 80, 80), cv.FILLED)
            cv.rectangle(frame, (header_x1, header_y1), (header_x2, header_y2), (0, 0, 0), 2)
            
            # Header text
            cv.putText(frame, "SONG LIST", (header_x1 + 10, header_y1 + 23), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Collapse indicator
            cv.putText(frame, "-", (header_x2 - 25, header_y1 + 23), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
            
            # Draw song list
            list_y_start = self.y + self.header_height
            for i, song in enumerate(self.songs):
                top_left = (self.x, list_y_start + i * self.item_height)
                bottom_right = (self.x + self.width, list_y_start + (i + 1) * self.item_height)

                # Highlight background if this song is selected
                if i == highlight_index:
                    cv.rectangle(frame, top_left, bottom_right, (0, 255, 0), cv.FILLED)
                else:
                    cv.rectangle(frame, top_left, bottom_right, (50, 50, 50), cv.FILLED)

                # Draw border
                cv.rectangle(frame, top_left, bottom_right, (0, 0, 0), 2)

                # Draw song name
                cv.putText(frame, song, (self.x + 5, list_y_start + (i + 1) * self.item_height - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def check_pinch(self, pinch_positions):
        """Return the index of the song being pinched, if any (only when expanded)"""
        if self.is_collapsed or self.is_dragging:
            return None
        
        list_y_start = self.y + self.header_height
        for i, song in enumerate(self.songs):
            top = list_y_start + i * self.item_height
            bottom = list_y_start + (i + 1) * self.item_height
            for px, py in pinch_positions:
                if self.x <= px <= self.x + self.width and top <= py <= bottom:
                    return i
        return None