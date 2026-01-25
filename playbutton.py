import cv2 as cv
import math
import numpy as np

class PlayButton:
    def __init__(self, center, radius=30, label="PLAY"):
        self.center = center
        self.radius = radius
        self.label = label

    def draw(self, frame, active=False):
        """Draw the play button"""
        color = (0, 255, 0) if active else (255, 255, 255)
        x, y = self.center

        # Fill the circle for active/inactive state
        cv.circle(frame, (x, y), self.radius, color, cv.FILLED)
        # Draw circle border in black
        cv.circle(frame, (x, y), self.radius, (0, 0, 0), 3)

        # Draw the play triangle in black for contrast
        triangle = np.array([
            [x - 8, y - 15],
            [x - 8, y + 15],
            [x + 18, y]
        ], dtype=np.int32)
        cv.fillPoly(frame, [triangle], (0, 0, 0))

        # Draw label
        cv.putText(
            frame,
            self.label,
            (x - 30, y + self.radius + 25),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    def contains(self, px, py):
        """Check if a point (cursor/fingertip) is inside the button"""
        return math.hypot(px - self.center[0], py - self.center[1]) < self.radius
