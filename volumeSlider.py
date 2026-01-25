import cv2 as cv

def clamp(val, lo, hi):
    return max(lo, min(val, hi))

def is_claw(hand_landmarks, threshold=0.1):
    """
    Returns True if all 5 fingers are extended (claw).
    hand_landmarks: list of 21 landmarks
    """
    if not hand_landmarks:
        return False

    tips = [4, 8, 12, 16, 20]
    mcps = [1, 5, 9, 13, 17]

    extended = []
    for tip_idx, mcp_idx in zip(tips, mcps):
        tip = hand_landmarks[tip_idx]
        mcp = hand_landmarks[mcp_idx]
        extended.append(tip.y < mcp.y - threshold)

    return all(extended)

class VolumeSlider:
    def __init__(self, x, y, width, height, track_index):
        self.x = x
        self.y = y
        self.w = width
        self.h = height
        self.track_index = track_index
        self.volume = 1.0
        self.active = False
        self.fingertips = []  # clear each frame

    def update(self, hand_landmarks, frame_width, frame_height):
        """
        Updates volume if the hand is in claw formation.
        Stores fingertip positions for visual feedback.
        """
        self.fingertips = []
        if hand_landmarks and is_claw(hand_landmarks):
            # Volume based on wrist y
            wrist = hand_landmarks[0]
            y_px = int(wrist.y * frame_height)
            rel_y = clamp(y_px - self.y, 0, self.h)
            self.volume = 1.0 - (rel_y / self.h)
            self.volume = clamp(self.volume, 0.0, 1.0)
            self.active = True

            # Fingertips for visual feedback
            tips = [4, 8, 12, 16, 20]
            self.fingertips = [(int(hand_landmarks[i].x * frame_width),
                                int(hand_landmarks[i].y * frame_height)) for i in tips]
        else:
            self.active = False
            self.fingertips = []

    def draw(self, frame):
        # Track
        cv.rectangle(frame, (self.x, self.y), (self.x+self.w, self.y+self.h), (180,180,180), 2)
        # Fill
        fill_h = int(self.h * self.volume)
        fill_y = self.y + (self.h - fill_h)
        color = (0,255,0) if self.active else (255,255,255)
        cv.rectangle(frame, (self.x, fill_y), (self.x+self.w, self.y+self.h), color, -1)
        # Label
        cv.putText(frame, f"{int(self.volume*100)}%", (self.x-5, self.y-10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        # Fingertips
        for cx, cy in self.fingertips:
            cv.circle(frame, (cx, cy), 8, (255,255,255), -1)
