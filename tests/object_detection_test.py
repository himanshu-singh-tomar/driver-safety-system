
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Use a lightweight pretrained model.
# Ultralytics downloads the weights automatically on first use.
MODEL_NAME = "yolo11n.pt"

CONFIDENCE_THRESHOLD = 0.35
IMAGE_SIZE = 416


def main():
    print("Loading YOLO11n model...")
    print("The first run may take time to download model weights.")

    try:
        model = YOLO(MODEL_NAME)
    except Exception as exc:
        print(f"ERROR: Could not load YOLO model: {exc}")
        return

    print("YOLO model loaded successfully.")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    previous_time = time.perf_counter()
    fps = 0.0

    try:
        print("Object detection started.")
        print("Show objects to the camera. Press Q to quit.")

        while True:
            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)

            # Run object detection on the current webcam frame.
            results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                imgsz=IMAGE_SIZE,
                device="cpu",
                verbose=False,
            )

            result = results[0]

            # Draw predicted boxes, class labels and confidence scores.
            annotated_frame = result.plot()

            # Count detected objects.
            object_count = len(result.boxes)

            current_time = time.perf_counter()
            elapsed = current_time - previous_time

            if elapsed > 0:
                instantaneous_fps = 1.0 / elapsed
                fps = (
                    instantaneous_fps
                    if fps == 0
                    else 0.8 * fps + 0.2 * instantaneous_fps
                )

            previous_time = current_time

            cv2.putText(
                annotated_frame,
                f"Objects detected: {object_count}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated_frame,
                "Press Q to quit",
                (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "YOLO Real-Time Object Detection",
                annotated_frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as exc:
        print(f"Object detection error: {exc}")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam released.")


if __name__ == "__main__":
    main()