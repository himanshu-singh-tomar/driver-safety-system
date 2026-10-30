
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolo11n.pt"

CONFIDENCE_THRESHOLD = 0.35
IMAGE_SIZE = 416


class ObjectDetector:
    """Detects common objects using the original COCO YOLO model."""

    def __init__(self):
        model_source = (
            str(MODEL_PATH)
            if MODEL_PATH.is_file()
            else "yolo11n.pt"
        )

        self.model = YOLO(model_source)
        print("COCO object detection model loaded.")

    def process(self, frame):
        """
        Detect objects in a BGR frame.

        Returns a list of detections containing:
        class_id, label, confidence, bbox and source.
        """

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
                "source": "coco",
            })

        return detections