import cv2 as cv

class LoadButton:
    def __init__(self, center, radius=25, label="LOAD"):
        self.cx, self.cy = center
        self.radius = radius
        self.label = label

    def draw(self, frame, active=False):
        color = (0, 255, 0) if active else (100, 100, 100)
        cv.circle(frame, (self.cx, self.cy), self.radius, color, -1)
        cv.circle(frame, (self.cx, self.cy), self.radius, (0, 0, 0), 2)
        cv.putText(frame, self.label, (self.cx - 20, self.cy + 7),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    def contains(self, x, y):
        return ((x - self.cx) ** 2 + (y - self.cy) ** 2) ** 0.5 <= self.radius
