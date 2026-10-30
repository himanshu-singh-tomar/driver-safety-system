
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lost_found_yolo11n.pt"
)

CONFIDENCE_THRESHOLD = 0.35
IMAGE_SIZE = 416


class CustomObjectDetector:
    """Detects the project's 10 custom personal-object classes."""

    def __init__(self):
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Custom YOLO model not found: {MODEL_PATH}"
            )

        self.model = YOLO(str(MODEL_PATH))

        print("Custom object detection model loaded.")
        print("Classes:", self.model.names)

    def process(self, frame):
        result = self.model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=IMAGE_SIZE,
            device="cpu",
            verbose=False,
        )[0]

        detections = []

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            x1, y1, x2, y2 = (
                int(value)
                for value in box.xyxy[0].tolist()
            )

            detections.append({
                "class_id": class_id,
                "label": self.model.names[class_id],
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
                "source": "custom",
            })

        return detections