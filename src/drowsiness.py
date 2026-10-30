
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"

# Initial values; we will tune them using your webcam tests.
EAR_THRESHOLD = 0.18
CLOSED_EYE_SECONDS = 1.5
SMOOTHING_FRAMES = 5

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def calculate_ear(landmarks, indices, width, height):
    """Calculate eye aspect ratio and reject invalid eye geometry."""
    points = np.array(
        [
            [landmarks[i].x * width, landmarks[i].y * height]
            for i in indices
        ],
        dtype=np.float64,
    )

    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    # An eye that is too small or badly tracked is unreliable.
    if horizontal < 5:
        return None

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    if not np.isfinite(ear) or ear < 0 or ear > 1.0:
        return None

    return float(ear)


class DrowsinessDetector:
    def __init__(self):
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Face landmark model not found: {MODEL_PATH}"
            )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(MODEL_PATH)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
        )

        self.landmarker = (
            mp.tasks.vision.FaceLandmarker.create_from_options(options)
        )

        self.closed_since = None
        self.previous_timestamp_ms = -1
        self.left_history = deque(maxlen=SMOOTHING_FRAMES)
        self.right_history = deque(maxlen=SMOOTHING_FRAMES)

    def _reset_tracking(self):
        self.closed_since = None
        self.left_history.clear()
        self.right_history.clear()

    def process(self, frame):
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        timestamp_ms = time.monotonic_ns() // 1_000_000
        timestamp_ms = max(
            timestamp_ms, self.previous_timestamp_ms + 1
        )
        self.previous_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(
            mp_image, timestamp_ms
        )
        now = time.monotonic()

        if not result.face_landmarks:
            self._reset_tracking()
            return {
                "status": "NO FACE DETECTED",
                "ear": None,
                "closed_duration": 0.0,
                "eye_points": [],
            }

        landmarks = result.face_landmarks[0]
        eye_points = []
        for indices in (LEFT_EYE, RIGHT_EYE):
            for index in indices:
                point = landmarks[index]
                eye_points.append(
                    (
                        int(point.x * width),
                        int(point.y * height),
                    )
                )

        left_ear = calculate_ear(
            landmarks, LEFT_EYE, width, height
        )
        right_ear = calculate_ear(
            landmarks, RIGHT_EYE, width, height
        )

        # Do not guess eye state when either eye is unreliable.
        if left_ear is None or right_ear is None:
            self._reset_tracking()
            return {
                "status": "EYE TRACKING UNCERTAIN",
                "ear": None,
                "closed_duration": 0.0,
                "eye_points": eye_points,
            }

        # A large difference may indicate head rotation or
        # inaccurate landmarks. Treat it as uncertain.
        if abs(left_ear - right_ear) > 0.20:
            self._reset_tracking()
            return {
                "status": "EYE TRACKING UNCERTAIN",
                "ear": (left_ear + right_ear) / 2.0,
                "closed_duration": 0.0,
                "eye_points": eye_points,
            }

        self.left_history.append(left_ear)
        self.right_history.append(right_ear)

        smooth_left = float(np.mean(self.left_history))
        smooth_right = float(np.mean(self.right_history))
        ear = (smooth_left + smooth_right) / 2.0

        # Require several frames before classifying closure.
        enough_history = (
            len(self.left_history) == SMOOTHING_FRAMES
            and len(self.right_history) == SMOOTHING_FRAMES
        )

        both_eyes_closed = (
            enough_history
            and smooth_left < EAR_THRESHOLD
            and smooth_right < EAR_THRESHOLD
        )

        if both_eyes_closed:
            if self.closed_since is None:
                self.closed_since = now

            closed_duration = now - self.closed_since

            if closed_duration >= CLOSED_EYE_SECONDS:
                status = "DROWSINESS WARNING"
            else:
                status = "EYES CLOSED - MONITORING"
        else:
            self.closed_since = None
            closed_duration = 0.0

            if enough_history:
                status = "EYES OPEN"
            else:
                status = "CALIBRATING EYES"

        return {
            "status": status,
            "ear": ear,
            "closed_duration": closed_duration,
            "eye_points": eye_points,
        }

    def close(self):
        self.landmarker.close()