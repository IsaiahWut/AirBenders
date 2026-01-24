import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
# Camera Setup
# -----------------------------
cam = cv.VideoCapture(1)
frame_idx = 0

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

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

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

                # Bone lines (changed to white)
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

    cv.imshow("Show Video", frame)

    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()