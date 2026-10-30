
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

try:
    import winsound
except ImportError:
    winsound = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"

# Initial experimental settings; these may need calibration.
EAR_THRESHOLD = 0.21
CLOSED_EYE_SECONDS = 1.5
ALERT_COOLDOWN_SECONDS = 2.0

# Six landmark indices for each eye.
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def calculate_ear(landmarks, indices, width, height):
    """Calculate the eye aspect ratio from six eye landmarks."""
    points = np.array(
        [
            [landmarks[i].x * width, landmarks[i].y * height]
            for i in indices
        ],
        dtype=np.float64,
    )

    # Vertical eye distances.
    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])

    # Horizontal eye distance.
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal < 1e-6:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def main():
    if not MODEL_PATH.is_file():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        return

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    closed_since = None
    last_alert_time = 0.0
    previous_timestamp_ms = -1

    print("Eye-closure detection started.")
    print("Look at the camera, blink normally, and press Q to quit.")

    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:
                success, frame = camera.read()

                if not success:
                    print("Could not read webcam frame.")
                    break

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                timestamp_ms = time.monotonic_ns() // 1_000_000
                timestamp_ms = max(
                    timestamp_ms, previous_timestamp_ms + 1
                )
                previous_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(
                    mp_image, timestamp_ms
                )

                now = time.monotonic()
                status = "NO FACE DETECTED"
                color = (0, 0, 255)
                ear = None
                closed_duration = 0.0

                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]

                    left_ear = calculate_ear(
                        landmarks, LEFT_EYE, width, height
                    )
                    right_ear = calculate_ear(
                        landmarks, RIGHT_EYE, width, height
                    )
                    ear = (left_ear + right_ear) / 2.0

                    # Draw landmarks around both eyes.
                    for eye_indices in (LEFT_EYE, RIGHT_EYE):
                        for index in eye_indices:
                            point = landmarks[index]
                            x = int(point.x * width)
                            y = int(point.y * height)
                            cv2.circle(
                                frame, (x, y), 3, (0, 255, 255), -1
                            )

                    if ear < EAR_THRESHOLD:
                        if closed_since is None:
                            closed_since = now

                        closed_duration = now - closed_since

                        if closed_duration >= CLOSED_EYE_SECONDS:
                            status = "DROWSINESS WARNING"
                            color = (0, 0, 255)

                            if (
                                now - last_alert_time
                                >= ALERT_COOLDOWN_SECONDS
                            ):
                                if winsound is not None:
                                    winsound.MessageBeep(
                                        winsound.MB_ICONEXCLAMATION
                                    )
                                last_alert_time = now
                        else:
                            status = "EYES CLOSED - MONITORING"
                            color = (0, 165, 255)
                    else:
                        closed_since = None
                        closed_duration = 0.0
                        status = "EYES OPEN"
                        color = (0, 255, 0)

                else:
                    # Lost tracking must not retain an old eye timer.
                    closed_since = None

                cv2.putText(
                    frame, status, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2
                )

                if ear is not None:
                    cv2.putText(
                        frame, f"EAR: {ear:.3f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2
                    )

                    cv2.putText(
                        frame,
                        f"Closure: {closed_duration:.1f}s",
                        (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2
                    )

                cv2.putText(
                    frame, "Press Q to quit", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2
                )

                cv2.imshow("Driver Drowsiness - Eye Closure Test", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except Exception as exc:
        print(f"Detection error: {exc}")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam released.")


if __name__ == "__main__":
    main()
