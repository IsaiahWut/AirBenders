import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import music as mc

# -----------------------------
# Load Music
# -----------------------------
MUSIC_FOLDER = "music"  # Change this to your music folder path
mc.load_music_folder(MUSIC_FOLDER)

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
# Pinch Detection Function
# -----------------------------
def is_pinching(hand_landmarks, w, h, threshold=40):
    """Check if thumb tip and index tip are close together"""
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    
    thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
    index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
    
    distance = math.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
    
    return distance < threshold

# -----------------------------
# Camera Setup
# -----------------------------
cam = cv.VideoCapture(1)
frame_idx = 0

# Pinch state tracking
left_hand_pinching = False
right_hand_pinching = False
left_was_pinching = False
right_was_pinching = False

while cam.isOpened():
    success, frame = cam.read()
    if not success:
        print("Camera Frame not available")
        continue

    h, w, _ = frame.shape

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect_for_video(mp_image, frame_idx)
    frame_idx += 1

    # Reset pinch states
    left_hand_pinching = False
    right_hand_pinching = False

    if result.hand_landmarks and result.handedness:
        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            
            # Determine which hand (left or right)
            hand_label = handedness[0].category_name  # "Left" or "Right"
            
            # Check for pinch
            pinching = is_pinching(hand_landmarks, w, h)
            
            if hand_label == "Left":
                left_hand_pinching = pinching
            else:  # "Right"
                right_hand_pinching = pinching

            # -----------------------------
            # All 5 Finger Chains
            # -----------------------------
            fingers_to_draw = [
                [0, 1, 2, 3, 4],        # Thumb
                [0, 5, 6, 7, 8],        # Index
                [0, 9, 10, 11, 12],     # Middle
                [0, 13, 14, 15, 16],    # Ring
                [0, 17, 18, 19, 20]     # Pinky
            ]

            for finger in fingers_to_draw:
                poly_coords = []

                for i in finger:
                    lm = hand_landmarks[i]
                    px, py = int(lm.x * w), int(lm.y * h)
                    poly_coords.append((px, py))

                    # Joint
                    cv.circle(frame, (px, py), 5, (0, 255, 0), cv.FILLED)

                # Bone lines
                for i in range(len(poly_coords) - 1):
                    cv.line(
                        frame,
                        poly_coords[i],
                        poly_coords[i + 1],
                        (255, 255, 255),
                        2
                    )

            # -----------------------------
            # Fingertips (All 5)
            # -----------------------------
            tip_indices = [4, 8, 12, 16, 20]
            tip_colors = [
                (255, 255, 0),    # Thumb
                (0, 255, 255),    # Index
                (255, 0, 255),    # Middle
                (0, 255, 0),      # Ring
                (0, 0, 255)       # Pinky
            ]

            for tip_idx, color in zip(tip_indices, tip_colors):
                lm = hand_landmarks[tip_idx]
                x, y = int(lm.x * w), int(lm.y * h)

                cv.circle(frame, (x, y), 15, color, cv.FILLED)

                cv.putText(
                    frame,
                    f"({x}, {y})",
                    (x + 20, y - 20),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

    # -----------------------------
    # Music Control Logic
    # -----------------------------
    # Left pinch = play next song
    if left_hand_pinching and not left_was_pinching:
        mc.play_next()
        print("Left pinch: Next song")
    
    # Right pinch = play previous song
    if right_hand_pinching and not right_was_pinching:
        mc.play_previous()
        print("Right pinch: Previous song")

    # Update previous states
    left_was_pinching = left_hand_pinching
    right_was_pinching = right_hand_pinching

    # Display pinch status
    cv.putText(
        frame,
        f"Left Pinch (Next): {left_hand_pinching}",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0) if left_hand_pinching else (0, 0, 255),
        2
    )
    
    cv.putText(
        frame,
        f"Right Pinch (Prev): {right_hand_pinching}",
        (10, 60),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0) if right_hand_pinching else (0, 0, 255),
        2
    )

    # Display current song
    song_name = mc.get_current_song_name()
    if song_name:
        cv.putText(
            frame,
            f"Playing: {song_name}",
            (10, 90),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

    cv.imshow("Show Video", frame)

    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()