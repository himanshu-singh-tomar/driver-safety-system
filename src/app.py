
import time

import cv2

from src.drowsiness import DrowsinessDetector
from src.object_detection import ObjectDetector
from src.custom_object_detection import CustomObjectDetector
from src.alerts import AlertManager


# Run YOLO models at different intervals.
COCO_INTERVAL = 3
CUSTOM_INTERVAL = 8

# IoU threshold for duplicate phone detection.
PHONE_IOU_THRESHOLD = 0.50


# Consistent colors for detected object categories (BGR format).
OBJECT_COLORS = {
    "phone": (255, 0, 0),
    "wallet": (0, 165, 255),
    "keys": (255, 0, 255),
    "card": (255, 255, 0),
    "charger": (0, 128, 255),
    "earphones": (180, 105, 255),
    "glasses": (0, 255, 0),
    "sunglasses": (50, 205, 154),
    "watch": (0, 0, 255),
    "pen": (128, 128, 128),
}

DEFAULT_OBJECT_COLOR = (255, 255, 255)


def get_object_color(label):
    """Return a consistent color for an object category."""
    return OBJECT_COLORS.get(
        label.strip().lower(),
        DEFAULT_OBJECT_COLOR
    )


def calculate_iou(box_a, box_b):
    """Calculate Intersection over Union of two bounding boxes."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = intersection_width * intersection_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def combine_detections(coco_detections, custom_detections):
    """
    Combine both models' detections.

    Normalize cell phone and Phone to one category.
    Remove overlapping duplicate phone detections.
    """

    combined = []

    for detection in coco_detections:
        item = detection.copy()

        if item["label"].lower() == "cell phone":
            item["label"] = "Phone"
            item["category"] = "Phone"
        else:
            item["category"] = item["label"]

        combined.append(item)

    for detection in custom_detections:
        item = detection.copy()

        if item["label"].lower() == "phone":
            item["label"] = "Phone"
            item["category"] = "Phone"

            duplicate_found = False

            for existing in combined:
                if existing["category"] != "Phone":
                    continue

                overlap = calculate_iou(
                    existing["bbox"],
                    item["bbox"],
                )

                if overlap >= PHONE_IOU_THRESHOLD:
                    duplicate_found = True

                    # Keep the detection with higher confidence.
                    if item["confidence"] > existing["confidence"]:
                        existing.update(item)

                    break

            if duplicate_found:
                continue

        else:
            item["category"] = item["label"]

        combined.append(item)

    return combined


def main():
    print("Initializing Driver Safety System...")

    camera = None
    drowsiness = None

    try:
        drowsiness = DrowsinessDetector()

        objects = ObjectDetector()
        custom_objects = CustomObjectDetector()

        alerts = AlertManager(cooldown_seconds=2.0)

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            print("ERROR: Could not open laptop webcam.")
            return

        print("Driver Safety System started.")
        print("Press Q to quit.")

        frame_count = 0

        coco_detections = []
        custom_detections = []

        previous_time = time.perf_counter()
        fps = 0.0

        while True:
            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            frame_count += 1

            # Priority 1: Driver monitoring every frame.
            driver_result = drowsiness.process(frame)

            # Priority 2: Run COCO detection periodically.
            if frame_count % COCO_INTERVAL == 1:
                try:
                    coco_detections = objects.process(frame)
                except Exception as exc:
                    print(f"COCO detection error: {exc}")
                    coco_detections = []

            # Priority 3: Run custom detection periodically.
            if frame_count % CUSTOM_INTERVAL == 1:
                try:
                    custom_detections = custom_objects.process(frame)
                except Exception as exc:
                    print(f"Custom detection error: {exc}")
                    custom_detections = []

            # Combine and filter duplicate phone detections.
            detections = combine_detections(
                coco_detections,
                custom_detections,
            )

            # Draw all object detections.
            for detection in detections:
                x1, y1, x2, y2 = detection["bbox"]

                box_color = get_object_color(detection["label"])

                label = (
                    f'{detection["label"]} '
                    f'{detection["confidence"]:.2f}'
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2,
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_color,
                    2,
                )

            # Draw eye landmarks.
            for x, y in driver_result["eye_points"]:
                if (
                    0 <= x < frame.shape[1]
                    and 0 <= y < frame.shape[0]
                ):
                    cv2.circle(
                        frame,
                        (x, y),
                        3,
                        (0, 255, 255),
                        -1,
                    )

            # Driver status.
            status = driver_result["status"]

            if status == "DROWSINESS WARNING":
                alerts.trigger_drowsiness_alert()
                status_color = (0, 0, 255)

            elif status in (
                "NO FACE DETECTED",
                "EYE TRACKING UNCERTAIN",
                "EYES CLOSED - MONITORING",
                "CALIBRATING EYES",
            ):
                status_color = (0, 165, 255)

            else:
                status_color = (0, 255, 0)

            cv2.putText(
                frame,
                f"Driver: {status}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
            )

            # Eye aspect ratio.
            ear = driver_result["ear"]

            if ear is not None:
                cv2.putText(
                    frame,
                    f"EAR: {ear:.3f}",
                    (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            # Object count.
            cv2.putText(
                frame,
                f"Objects detected: {len(detections)}",
                (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # FPS.
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
                frame,
                f"Processing FPS: {fps:.1f}",
                (15, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "Q: Quit",
                (15, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("AI Driver Safety System", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as exc:
        print(f"Application error: {exc}")

    finally:
        if camera is not None:
            camera.release()

        if drowsiness is not None:
            drowsiness.close()

        cv2.destroyAllWindows()

        print("Driver Safety System stopped.")


if __name__ == "__main__":
    main()