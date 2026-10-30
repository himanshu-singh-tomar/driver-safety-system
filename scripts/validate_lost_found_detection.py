
from pathlib import Path
from collections import Counter
import cv2

ROOT = Path(r"D:\driver-safety-system")
DATASET = ROOT / "datasets" / "custom_objects" / "lost_found_detection"

SPLITS = ("train", "valid", "test")
CLASS_NAMES = [
    "Card", "Charger", "Earphones", "Glasses", "Keys",
    "Pen", "Phone", "Sunglasses", "Wallet", "Watch",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    errors = []
    total_images = 0
    total_boxes = 0
    total_empty = 0
    class_counts = Counter()

    for split in SPLITS:
        image_dir = DATASET / "images" / split
        label_dir = DATASET / "labels" / split

        images = {
            p.stem: p
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        }
        labels = {p.stem: p for p in label_dir.glob("*.txt") if p.is_file()}

        missing_labels = set(images) - set(labels)
        missing_images = set(labels) - set(images)

        if missing_labels:
            errors.append(
                f"{split}: {len(missing_labels)} images have no label file"
            )
        if missing_images:
            errors.append(
                f"{split}: {len(missing_images)} label files have no image"
            )

        split_boxes = 0
        split_empty = 0

        for stem, image_path in images.items():
            image = cv2.imread(str(image_path))

            if image is None:
                errors.append(f"Unreadable image: {image_path}")
                continue

            label_path = labels.get(stem)
            if label_path is None:
                continue

            lines = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            if not lines:
                split_empty += 1
                continue

            for line_number, line in enumerate(lines, start=1):
                parts = line.split()

                if len(parts) != 5:
                    errors.append(
                        f"{label_path}:{line_number}: expected 5 fields"
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    x, y, width, height = map(float, parts[1:])
                except ValueError:
                    errors.append(
                        f"{label_path}:{line_number}: invalid number"
                    )
                    continue

                if not 0 <= class_id < len(CLASS_NAMES):
                    errors.append(
                        f"{label_path}:{line_number}: invalid class ID"
                    )
                    continue

                if not all(0 <= v <= 1 for v in (x, y, width, height)):
                    errors.append(
                        f"{label_path}:{line_number}: coordinate outside [0,1]"
                    )
                    continue

                if width <= 0 or height <= 0:
                    errors.append(
                        f"{label_path}:{line_number}: zero-sized box"
                    )
                    continue

                if (
                    x - width / 2 < -1e-5
                    or y - height / 2 < -1e-5
                    or x + width / 2 > 1 + 1e-5
                    or y + height / 2 > 1 + 1e-5
                ):
                    errors.append(
                        f"{label_path}:{line_number}: box extends outside image"
                    )
                    continue

                class_counts[class_id] += 1
                split_boxes += 1

        total_images += len(images)
        total_boxes += split_boxes
        total_empty += split_empty

        print(
            f"{split}: {len(images)} images, "
            f"{split_boxes} valid boxes, {split_empty} empty labels"
        )

    print("\n--- CLASS DISTRIBUTION ---")
    for class_id, name in enumerate(CLASS_NAMES):
        print(f"{class_id} - {name}: {class_counts[class_id]} boxes")

    print("\n--- VALIDATION SUMMARY ---")
    print(f"Images checked: {total_images}")
    print(f"Valid bounding boxes: {total_boxes}")
    print(f"Empty labels: {total_empty}")
    print(f"Errors: {len(errors)}")

    for error in errors[:25]:
        print("ERROR:", error)

    if len(errors) > 25:
        print(f"... and {len(errors) - 25} additional errors")

    if errors:
        raise SystemExit(
            "\nValidation FAILED. Do not train until the errors are reviewed."
        )

    print("\nVALIDATION PASSED")


if __name__ == "__main__":
    main()