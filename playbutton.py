import cv2 as cv
import math
import numpy as np

class PlayButton:
    def __init__(self, center, radius=30, label="PLAY"):
        self.center = center
        self.radius = radius
        self.label = label

    def draw(self, frame, state="empty"):
        x, y = self.center

        # Color by state
        if state == "empty":
            color = (255, 255, 255)   # white
        elif state == "loaded":
            color = (0, 165, 255)     # orange
        elif state == "playing":
            color = (173, 216, 230)   # light blue
        else:
            color = (255, 255, 255)

        # Draw button
        cv.circle(frame, (x, y), self.radius, color, -1)
        cv.circle(frame, (x, y), self.radius, (0, 0, 0), 3)

        # Draw play triangle (black)
        triangle = np.array([
            [x - 8, y - 15],
            [x - 8, y + 15],
            [x + 18, y]
        ], dtype=np.int32)
        cv.fillPoly(frame, [triangle], (0, 0, 0))

        # Label below button
        cv.putText(frame, self.label, (x - 30, y + self.radius + 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    def contains(self, px, py):
        return math.hypot(px - self.center[0], py - self.center[1]) < self.radius
