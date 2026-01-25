import os
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
import songlist
from load import LoadButton
from volumeSlider import VolumeSlider, clamp, is_claw

# -----------------------------
# Load Music
# -----------------------------
MUSIC_FOLDER = "MP3"
mc.load_music_folder(MUSIC_FOLDER)

# -----------------------------
# Deck Initialization
# -----------------------------
left_song_index = -1
right_song_index = -1

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
# Song List Panel
# -----------------------------
song_names = [os.path.basename(s) for s in mc.songs]
song_list_panel = None
song_list_width = 250
item_height = 30
highlighted_index = None

# -----------------------------
# Pinch Detection (for play & load only)
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
    def __init__(self, index): self.index = index
    def __call__(self, delta):
        if self.index >= 0: mc.scrub(delta, self.index)

class ReleaseCallback:
    def __init__(self, index): self.index = index
    def __call__(self):
        if self.index >= 0: mc.end_scrub(self.index)

# -----------------------------
# Camera & UI
# -----------------------------
cam = cv.VideoCapture(0)
frame_idx = 0
left_button = right_button = left_load_button = right_load_button = None
left_jog = right_jog = None
left_volume = right_volume = None
visualizer = None
pinching_previous = set()

# -----------------------------
# Main Loop
# -----------------------------
while cam.isOpened():
    success, frame = cam.read()
    if not success: continue

    mc.update_active_track_position()
    frame = cv.flip(frame, 1)
    h, w, _ = frame.shape

    # Initialize song list
    if song_list_panel is None:
        song_list_y = h - item_height * len(song_names) - 120
        song_list_panel = songlist.SongList(song_names, position=(10, song_list_y), width=song_list_width, item_height=item_height)

    # Initialize UI elements
    if left_button is None:
        center_x = w//2
        button_y = int(h*0.83)
        left_button = PlayButton(center=(center_x-200, button_y), radius=30, label="PLAY 1")
        right_button = PlayButton(center=(center_x+200, button_y), radius=30, label="PLAY 2")
        left_load_button = LoadButton(center=(center_x-200, button_y-70), radius=25, label="LOAD")
        right_load_button = LoadButton(center=(center_x+200, button_y-70), radius=25, label="LOAD")
        jog_y = int(h*0.55)
        left_jog = JogWheel(center=(center_x-350, jog_y), radius=160)
        right_jog = JogWheel(center=(center_x+350, jog_y), radius=160)

        slider_width = 30
        slider_height = 200
        slider_offset = 220
        left_volume = VolumeSlider(x=left_jog.cx - slider_offset - slider_width, y=left_jog.cy - slider_height//2, width=slider_width, height=slider_height, track_index=0)
        right_volume = VolumeSlider(x=right_jog.cx + slider_offset, y=right_jog.cy - slider_height//2, width=slider_width, height=slider_height, track_index=1)

    if visualizer is None:
        visualizer = DJVisualizer(w, h)

    # -----------------------------
    # Hand Detection
    # -----------------------------
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(time.time() * 1000)
    detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

    pinch_positions = []
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            if is_pinching(hand_landmarks, w, h):
                thumb_tip = hand_landmarks[4]
                index_tip = hand_landmarks[8]
                cx = int((thumb_tip.x + index_tip.x)/2 * w)
                cy = int((thumb_tip.y + index_tip.y)/2 * h)
                pinch_positions.append((cx, cy))
                cv.circle(frame, (cx, cy), 10, (0,255,0), -1)

    # -----------------------------
    # Song List Pinch Toggle
    # -----------------------------
    for px, py in pinch_positions:
        idx = song_list_panel.check_pinch([(px, py)])
        if idx is not None:
            highlighted_index = None if highlighted_index == idx else idx

    song_list_panel.draw(frame, highlight_index=highlighted_index)

    # -----------------------------
    # Load Buttons Logic
    # -----------------------------
    left_load_active = right_load_active = False
    for px, py in pinch_positions:
        if highlighted_index is not None:
            if left_load_button.contains(px, py):
                left_song_index = highlighted_index
                highlighted_index = None
                left_load_active = True
            if right_load_button.contains(px, py):
                right_song_index = highlighted_index
                highlighted_index = None
                right_load_active = True

    # -----------------------------
    # Play & Load Buttons
    # -----------------------------
    left_state = "empty" if left_song_index<0 else "playing" if mc.is_playing(left_song_index) else "loaded"
    right_state = "empty" if right_song_index<0 else "playing" if mc.is_playing(right_song_index) else "loaded"
    left_button.draw(frame, state=left_state)
    right_button.draw(frame, state=right_state)
    left_load_button.draw(frame, active=left_load_active)
    right_load_button.draw(frame, active=right_load_active)

    # -----------------------------
    # Trigger Play if Pinched
    # -----------------------------
    left_active = left_song_index>=0 and any(left_button.contains(x,y) for x,y in pinch_positions)
    right_active = right_song_index>=0 and any(right_button.contains(x,y) for x,y in pinch_positions)
    if left_active and left_button.center not in pinching_previous: mc.toggle_play(left_song_index)
    if right_active and right_button.center not in pinching_previous: mc.toggle_play(right_song_index)
    pinching_previous = set()
    if left_active: pinching_previous.add(left_button.center)
    if right_active: pinching_previous.add(right_button.center)

    # -----------------------------
    # Jog Wheels
    # -----------------------------
    left_pinching_jog = right_pinching_jog = False
    for cx, cy in pinch_positions:
        if left_song_index>=0 and left_jog.contains(cx, cy):
            left_pinching_jog = True
            left_jog.update(frame, cx, cy, True, ScrubCallback(left_song_index), ReleaseCallback(left_song_index))
        if right_song_index>=0 and right_jog.contains(cx, cy):
            right_pinching_jog = True
            right_jog.update(frame, cx, cy, True, ScrubCallback(right_song_index), ReleaseCallback(right_song_index))
    if not left_pinching_jog: left_jog.check_release()
    if not right_pinching_jog: right_jog.check_release()

    # Spin visuals
    if left_song_index>=0 and mc.get_active_track()==left_song_index and mc.is_playing(left_song_index):
        left_jog.angle += 0.05
    if right_song_index>=0 and mc.get_active_track()==right_song_index and mc.is_playing(right_song_index):
        right_jog.angle += 0.05
    left_jog.draw(frame)
    right_jog.draw(frame)

   # -----------------------------
    # Volume Sliders (flipped hands)
    # -----------------------------
    # Use actual left/right hand detection, but swap due to mirrored camera
    if detection_result.hand_landmarks and detection_result.handedness:
        for hand_info, hand_landmarks in zip(detection_result.handedness, detection_result.hand_landmarks):
            label = hand_info[0].category_name  # 'Left' or 'Right'
            # Swap left/right because the frame is mirrored
            if label == "Left":
                right_volume.update(hand_landmarks, w, h)
                if right_song_index >= 0:
                    mc.set_volume(right_song_index, right_volume.volume)
            elif label == "Right":
                left_volume.update(hand_landmarks, w, h)
                if left_song_index >= 0:
                    mc.set_volume(left_song_index, left_volume.volume)

    left_volume.draw(frame)
    right_volume.draw(frame)


    # -----------------------------
    # Display Time & Colors
    # -----------------------------
    left_time = mc.get_position(left_song_index) if left_song_index>=0 else 0
    right_time = mc.get_position(right_song_index) if right_song_index>=0 else 0
    active = mc.get_active_track()
    left_color = (0,255,255) if active==left_song_index else (100,100,100)
    right_color = (0,255,255) if active==right_song_index else (100,100,100)
    left_time_pos = (left_jog.cx-40, left_jog.cy+left_jog.radius+50)
    right_time_pos = (right_jog.cx-40, right_jog.cy+right_jog.radius+50)
    cv.putText(frame, format_time(left_time), left_time_pos, cv.FONT_HERSHEY_SIMPLEX, 0.6, left_color, 2)
    cv.putText(frame, format_time(right_time), right_time_pos, cv.FONT_HERSHEY_SIMPLEX, 0.6, right_color, 2)
    if active>=0:
        cv.putText(frame, f"NOW PLAYING: {mc.get_current_song_name(active)}", (10,210), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

    # Visualizer
    left_playing = left_song_index>=0 and mc.get_active_track()==left_song_index and mc.is_playing(left_song_index)
    right_playing = right_song_index>=0 and mc.get_active_track()==right_song_index and mc.is_playing(right_song_index)
    visualizer.draw_all(frame, left_playing, right_playing)

    # Show Frame
    cv.imshow("Show Video", frame)
    if cv.waitKey(20) & 0xFF == ord('q'): break

cam.release()
cv.destroyAllWindows()
