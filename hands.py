import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from visualizer import DJVisualizer
import math
from playbutton import PlayButton
from jogwheel import JogWheel
import music as mc
import time

# -----------------------------
# Load Music
# -----------------------------
MUSIC_FOLDER = "MP3"
mc.load_music_folder(MUSIC_FOLDER)

left_song_index = 0
right_song_index = 1 if len(mc.songs) > 1 else 0

# -----------------------------
# MediaPipe Hand Landmarker Setup
# -----------------------------
MODEL_PATH = "hand_landmarker.task"
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

landmarker = HandLandmarker.create_from_options(options)

# -----------------------------
# Pinch Detection
# -----------------------------
def is_pinching(hand_landmarks, w, h, threshold=40):
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
    ix, iy = int(index_tip.x * w), int(index_tip.y * h)
    return math.hypot(tx - ix, ty - iy) < threshold

def format_time(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

class ScrubCallback:
    def __init__(self, index):
        self.index = index
    def __call__(self, delta):
        mc.scrub(delta, self.index)

class ReleaseCallback:
    def __init__(self, index):
        self.index = index
    def __call__(self):
        pass  # do nothing now

# -----------------------------
# Camera
# -----------------------------
cam = cv.VideoCapture(0)
frame_idx = 0

# -----------------------------
# UI Setup
# -----------------------------
h, w = 720, 1280
center_x = w // 2
button_y = 600

left_button = PlayButton(center=(center_x - 200, button_y), radius=30, label="PLAY 1")
right_button = PlayButton(center=(center_x + 200, button_y), radius=30, label="PLAY 2")

left_jog = JogWheel(center=(center_x - 350, 400), radius=160)
right_jog = JogWheel(center=(center_x + 350, 400), radius=160)

pinching_previous = set()

# -----------------------------
# Main Loop
# -----------------------------
while cam.isOpened():
    success, frame = cam.read()
    if not success:
        continue
    
    frame = cv.flip(frame, 1)

    h, w, _ = frame.shape

    # Convert to RGB for MediaPipe
    if 'visualizer' not in locals():
        visualizer = DJVisualizer(w, h)

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect_for_video(mp_image, frame_idx)
    frame_idx += 1

    mc.update_active_track_position()
    pinch_positions = []

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            pinching = is_pinching(hand_landmarks, w, h)
            index_tip = hand_landmarks[8]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            if pinching:
                pinch_positions.append((cx, cy))

            for i in [4, 8, 12, 16, 20]:
                lm = hand_landmarks[i]
                fx, fy = int(lm.x * w), int(lm.y * h)
                cv.circle(frame, (fx, fy), 10, (255, 255, 255), 2)

    # -----------------------------
    # Play Button Logic
    # -----------------------------
    left_active = any(left_button.contains(x, y) for x, y in pinch_positions)
    right_active = any(right_button.contains(x, y) for x, y in pinch_positions)

    left_button.draw(frame, active=left_active)
    right_button.draw(frame, active=right_active)

    if left_active and left_button.center not in pinching_previous:
        mc.toggle_play(left_song_index)
    if right_active and right_button.center not in pinching_previous:
        mc.toggle_play(right_song_index)

    pinching_previous = set()
    if left_active:
        pinching_previous.add(left_button.center)
    if right_active:
        pinching_previous.add(right_button.center)

    # -----------------------------
    # Jog Wheels Only Scrub
    # -----------------------------
    left_pinching_jog = False
    right_pinching_jog = False

    for cx, cy in pinch_positions:
        if left_jog.contains(cx, cy):
            left_pinching_jog = True
            left_jog.update(frame, cx, cy, True, ScrubCallback(left_song_index), ReleaseCallback(left_song_index))
        if right_jog.contains(cx, cy):
            right_pinching_jog = True
            right_jog.update(frame, cx, cy, True, ScrubCallback(right_song_index), ReleaseCallback(right_song_index))

    if not left_pinching_jog:
        left_jog.check_release()
    if not right_pinching_jog:
        right_jog.check_release()

    # -----------------------------
    # Spin visuals
    # -----------------------------
    if mc.get_active_track() == left_song_index and mc.is_playing(left_song_index):
        left_jog.angle += 0.05
    if mc.get_active_track() == right_song_index and mc.is_playing(right_song_index):
        right_jog.angle += 0.05

    left_jog.draw(frame)
    right_jog.draw(frame)

    # -----------------------------
    # Display Status
    # -----------------------------
    left_time = mc.get_position(left_song_index)
    right_time = mc.get_position(right_song_index)
    active = mc.get_active_track()

    cv.putText(frame, f"PLAY 1: {'Active' if left_active else 'Inactive'}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if left_active else (0, 0, 255), 2)
    cv.putText(frame, f"PLAY 2: {'Active' if right_active else 'Inactive'}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if right_active else (0, 0, 255), 2)
    cv.putText(frame, f"Left Song: {mc.get_current_song_name(left_song_index)}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv.putText(frame, f"Right Song: {mc.get_current_song_name(right_song_index)}", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    left_color = (0, 255, 255) if active == left_song_index else (100, 100, 100)
    right_color = (0, 255, 255) if active == right_song_index else (100, 100, 100)

    cv.putText(frame, f"Left Time: {format_time(left_time)}", (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.6, left_color, 2)
    cv.putText(frame, f"Right Time: {format_time(right_time)}", (10, 180), cv.FONT_HERSHEY_SIMPLEX, 0.6, right_color, 2)

    if active >= 0:
        cv.putText(frame, f"NOW PLAYING: {mc.get_current_song_name(active)}", (10, 210), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw visualizer
    visualizer.draw_all(frame)
    
    cv.imshow("Show Video", frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()
