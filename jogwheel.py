import math
import cv2
import time

class JogWheel:
    def __init__(self, center, radius=180, label=""):
        self.cx, self.cy = center
        self.radius = radius
        self.label = label
        self.angle = 0.0
        self.last_angle = None
        self.was_pinching = False
        self.last_time = time.time()
        self.on_release_callback = None

    def contains(self, x, y):
        return math.hypot(x - self.cx, y - self.cy) <= self.radius

    def _angle_from_center(self, x, y):
        return math.atan2(y - self.cy, x - self.cx)

    def update(self, frame, cursor_x, cursor_y, is_pinching, on_scrub, on_release):
        now = time.time()
        self.last_time = now

        # Store release callback
        self.on_release_callback = on_release

        if is_pinching and self.contains(cursor_x, cursor_y):
            if not self.was_pinching:
                self.was_pinching = True
                self.last_angle = self._angle_from_center(cursor_x, cursor_y)

            current_angle = self._angle_from_center(cursor_x, cursor_y)
            delta = current_angle - self.last_angle
            if delta > math.pi:
                delta -= 2 * math.pi
            elif delta < -math.pi:
                delta += 2 * math.pi
            self.angle += delta
            self.last_angle = current_angle

            on_scrub(delta)  # only move track position

    def check_release(self):
        if self.was_pinching:
            self.was_pinching = False
            self.last_angle = None
            if self.on_release_callback:
                self.on_release_callback()
                self.on_release_callback = None

    def draw(self, frame):
        cv2.circle(frame, (self.cx, self.cy), self.radius, (60, 60, 60), 2)
        x2 = int(self.cx + math.cos(self.angle) * self.radius)
        y2 = int(self.cy + math.sin(self.angle) * self.radius)
        cv2.line(frame, (self.cx, self.cy), (x2, y2), (0, 0, 255), 4)
        if self.label:
            cv2.putText(frame,
                        self.label,
                        (self.cx - 40, self.cy + self.radius + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (200, 200, 200),
                        2)