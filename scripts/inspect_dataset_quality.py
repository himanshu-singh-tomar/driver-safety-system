
from pathlib import Path
from collections import defaultdict
from statistics import mean, median
import csv

import yaml
from PIL import Image


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(r"D:\driver-safety-system")

DATASET_DIR = (
    ROOT
    / "datasets"
    / "custom_objects"
    / "lost_found_detection"
)

DATA_YAML = DATASET_DIR / "data.yaml"

OUTPUT_CSV = (
    ROOT
    / "reports"
    / "dataset_quality_report.csv"
)

SPLITS = ["train", "valid"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp"
}


def inspect_dataset():
    if not DATA_YAML.is_file():
        raise FileNotFoundError(DATA_YAML)

    with open(DATA_YAML, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    names = config["names"]

    if isinstance(names, dict):
        class_names = {
            int(key): value
            for key, value in names.items()
        }
    else:
        class_names = dict(enumerate(names))

    results = []
    errors = []

    for split in SPLITS:
        print(f"\nInspecting {split.upper()} dataset...")

        image_dir = DATASET_DIR / "images" / split
        label_dir = DATASET_DIR / "labels" / split

        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)

        if not label_dir.is_dir():
            raise FileNotFoundError(label_dir)

        images = [
            path for path in image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        stats = defaultdict(lambda: {
            "instances": 0,
            "images": 0,
            "widths": [],
            "heights": [],
            "areas": []
        })

        split_errors = 0
        missing_labels = 0

        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"

            try:
                with Image.open(image_path) as image:
                    image_width, image_height = image.size
            except Exception as exc:
                errors.append(
                    f"Image error: {image_path.name}: {exc}"
                )
                split_errors += 1
                continue

            if not label_path.is_file():
                missing_labels += 1
                continue

            try:
                lines = label_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            except Exception as exc:
                errors.append(
                    f"Label error: {label_path.name}: {exc}"
                )
                split_errors += 1
                continue

            image_classes = set()

            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue

                parts = line.split()

                try:
                    if len(parts) != 5:
                        raise ValueError(
                            "Expected 5 YOLO values"
                        )

                    class_id = int(parts[0])
                    x_center, y_center, box_w, box_h = (
                        map(float, parts[1:])
                    )

                    if class_id not in class_names:
                        raise ValueError("Unknown class ID")

                    values = [
                        x_center, y_center, box_w, box_h
                    ]

                    if not all(
                        0 <= value <= 1 for value in values
                    ):
                        raise ValueError(
                            "Normalized value outside [0, 1]"
                        )

                    if box_w <= 0 or box_h <= 0:
                        raise ValueError(
                            "Bounding box has zero size"
                        )

                    pixel_width = box_w * image_width
                    pixel_height = box_h * image_height

                    item = stats[class_id]

                    item["instances"] += 1
                    item["widths"].append(pixel_width)
                    item["heights"].append(pixel_height)
                    item["areas"].append(
                        pixel_width * pixel_height
                    )

                    image_classes.add(class_id)

                except Exception as exc:
                    errors.append(
                        f"{label_path.name}, line "
                        f"{line_number}: {exc}"
                    )
                    split_errors += 1

            for class_id in image_classes:
                stats[class_id]["images"] += 1

        for class_id, class_name in class_names.items():
            item = stats[class_id]

            widths = item["widths"]
            heights = item["heights"]
            areas = item["areas"]

            results.append({
                "split": split,
                "class_id": class_id,
                "class_name": class_name,
                "image_count": item["images"],
                "instances": item["instances"],
                "avg_width_px": round(mean(widths), 2)
                    if widths else 0,
                "median_width_px": round(median(widths), 2)
                    if widths else 0,
                "avg_height_px": round(mean(heights), 2)
                    if heights else 0,
                "median_height_px": round(median(heights), 2)
                    if heights else 0,
                "avg_area_px": round(mean(areas), 2)
                    if areas else 0
            })

        print(f"Images found: {len(images)}")
        print(f"Images missing label files: {missing_labels}")
        print(f"Malformed annotations/image errors: {split_errors}")

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if OUTPUT_CSV.exists():
        raise FileExistsError(
            f"Report already exists: {OUTPUT_CSV}"
        )

    fieldnames = list(results[0].keys())

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print("DATASET QUALITY SUMMARY")
    print("=" * 60)

    for split in SPLITS:
        print(f"\n{split.upper()}")

        for row in results:
            if row["split"] == split:
                print(
                    f'{row["class_name"]:12} '
                    f'Images: {row["image_count"]:4} '
                    f'Instances: {row["instances"]:5} '
                    f'Avg size: '
                    f'{row["avg_width_px"]} x '
                    f'{row["avg_height_px"]} px'
                )

    print("\nInspection completed.")
    print(f"Report saved to: {OUTPUT_CSV}")

    if errors:
        print(f"\nTotal errors: {len(errors)}")
        print("First 20 errors:")

        for error in errors[:20]:
            print(error)

    else:
        print("No malformed annotations or image errors found.")


if __name__ == "__main__":
    inspect_dataset()
