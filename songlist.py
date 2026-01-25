import cv2 as cv

class SongList:
    def __init__(self, songs, position=(0, 0), width=250, item_height=30):
        self.songs = songs
        self.x, self.y = position
        self.width = width
        self.item_height = item_height

    def draw(self, frame, highlight_index=None):
        """Draw the song list"""
        for i, song in enumerate(self.songs):
            top_left = (self.x, self.y + i * self.item_height)
            bottom_right = (self.x + self.width, self.y + (i + 1) * self.item_height)

            # Highlight background if this song is selected
            if i == highlight_index:
                cv.rectangle(frame, top_left, bottom_right, (0, 255, 0), cv.FILLED)
            else:
                cv.rectangle(frame, top_left, bottom_right, (50, 50, 50), cv.FILLED)

            # Draw border
            cv.rectangle(frame, top_left, bottom_right, (0, 0, 0), 2)

            # Draw song name
            cv.putText(frame, song, (self.x + 5, self.y + (i + 1) * self.item_height - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def check_pinch(self, pinch_positions):
        """Return the index of the song being pinched, if any"""
        for i, song in enumerate(self.songs):
            top = self.y + i * self.item_height
            bottom = self.y + (i + 1) * self.item_height
            for px, py in pinch_positions:
                if self.x <= px <= self.x + self.width and top <= py <= bottom:
                    return i
        return None
