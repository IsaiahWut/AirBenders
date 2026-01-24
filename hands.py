import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
from playbutton import PlayButton

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
    distance = math.hypot(thumb_x - index_x, thumb_y - index_y)
    return distance < threshold

# -----------------------------
# Camera Setup
# -----------------------------
cam = cv.VideoCapture(0)
frame_idx = 0

# -----------------------------
# Play Buttons
# -----------------------------
left_button = PlayButton(center=(400, 600), radius=30, label="PLAY 1")
right_button = PlayButton(center=(800, 600), radius=30, label="PLAY 2")

while cam.isOpened():
    success, frame = cam.read()
    if not success:
        print("Camera Frame not available")
        continue
    
    # Flip frame for mirror effect
    frame = cv.flip(frame, 1)

    h, w, _ = frame.shape
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = landmarker.detect_for_video(mp_image, frame_idx)
    frame_idx += 1

    # Reset cursor states
    cursors = []

    # -----------------------------
    # Hand Detection
    # -----------------------------
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            pinching = is_pinching(hand_landmarks, w, h)
            if pinching:
                index_tip = hand_landmarks[8]
                cursor_x, cursor_y = int(index_tip.x * w), int(index_tip.y * h)
                cursors.append((cursor_x, cursor_y))
            
            # Draw fingers and bones (optional)
            fingers_to_draw = [
                [0, 1, 2, 3, 4], [0, 5, 6, 7, 8],
                [0, 9, 10, 11, 12], [0, 13, 14, 15, 16],
                [0, 17, 18, 19, 20]
            ]
            for finger in fingers_to_draw:
                poly_coords = []
                for i in finger:
                    lm = hand_landmarks[i]
                    px, py = int(lm.x * w), int(lm.y * h)
                    poly_coords.append((px, py))
                    cv.circle(frame, (px, py), 5, (0, 255, 0), cv.FILLED)
                for i in range(len(poly_coords) - 1):
                    cv.line(frame, poly_coords[i], poly_coords[i + 1], (255, 255, 255), 2)

    # -----------------------------
    # Button Activation Logic
    # -----------------------------
    left_active = any(left_button.contains(x, y) for x, y in cursors)
    right_active = any(right_button.contains(x, y) for x, y in cursors)

    # -----------------------------
    # Draw Play Buttons
    # -----------------------------
    left_button.draw(frame, active=left_active)
    right_button.draw(frame, active=right_active)

    # -----------------------------
    # Display Status
    # -----------------------------
    cv.putText(frame, f"Left Button Status: {'Active' if left_active else 'Inactive'}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if not left_active else (0, 255, 0), 2)

    cv.putText(frame, f"Right Button Status: {'Active' if right_active else 'Inactive'}", (10, 60),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if not right_active else (0, 255, 0), 2)

    cv.imshow("Show Video", frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()