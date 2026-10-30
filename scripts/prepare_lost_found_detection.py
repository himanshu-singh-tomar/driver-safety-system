
from pathlib import Path
import shutil

ROOT = Path(r"D:\driver-safety-system")
SOURCE = ROOT / "datasets" / "custom_objects" / "source_lost_found"
OUTPUT = ROOT / "datasets" / "custom_objects" / "lost_found_detection"

SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

CLASS_NAMES = [
    "Card",
    "Charger",
    "Earphones",
    "Glasses",
    "Keys",
    "Pen",
    "Phone",
    "Sunglasses",
    "Wallet",
    "Watch",
]


def convert_label_line(line: str, label_path: Path, line_number: int) -> str:
    parts = line.split()

    if len(parts) < 5:
        raise ValueError(
            f"{label_path}, line {line_number}: too few values"
        )

    try:
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError(
            f"{label_path}, line {line_number}: invalid number"
        ) from exc

    if not 0 <= class_id < len(CLASS_NAMES):
        raise ValueError(
            f"{label_path}, line {line_number}: invalid class ID {class_id}"
        )

    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError(
            f"{label_path}, line {line_number}: coordinates outside [0, 1]"
        )

    # Standard YOLO detection box: class, x_center, y_center, width, height.
    if len(parts) == 5:
        x_center, y_center, width, height = values

        if width <= 0 or height <= 0:
            raise ValueError(
                f"{label_path}, line {line_number}: zero-sized box"
            )

    # YOLO polygon: class followed by x,y coordinate pairs.
    elif len(values) >= 6 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        width = x_max - x_min
        height = y_max - y_min

        if width <= 0 or height <= 0:
            raise ValueError(
                f"{label_path}, line {line_number}: degenerate polygon"
            )

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

    else:
        raise ValueError(
            f"{label_path}, line {line_number}: unsupported annotation format"
        )

    return (
        f"{class_id} {x_center:.6f} {y_center:.6f} "
        f"{width:.6f} {height:.6f}"
    )


def main():
    if not SOURCE.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE}")

    # Never overwrite a previous output automatically.
    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT}\n"
            "Inspect it before deciding how to proceed. "
            "This script will not overwrite it."
        )

    for split in SPLITS:
        if not (SOURCE / split / "images").is_dir():
            raise FileNotFoundError(f"Missing images for split: {split}")
        if not (SOURCE / split / "labels").is_dir():
            raise FileNotFoundError(f"Missing labels for split: {split}")

    total_images = 0
    total_boxes = 0
    total_polygons = 0
    total_empty = 0

    try:
        for split in SPLITS:
            source_images = SOURCE / split / "images"
            source_labels = SOURCE / split / "labels"
            output_images = OUTPUT / "images" / split
            output_labels = OUTPUT / "labels" / split

            output_images.mkdir(parents=True)
            output_labels.mkdir(parents=True)

            image_by_stem = {}

            for image_path in source_images.iterdir():
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    if image_path.stem in image_by_stem:
                        raise ValueError(
                            f"Duplicate image stem in {split}: {image_path.stem}"
                        )
                    image_by_stem[image_path.stem] = image_path

            label_by_stem = {
                path.stem: path
                for path in source_labels.glob("*.txt")
                if path.is_file()
            }

            missing_labels = set(image_by_stem) - set(label_by_stem)
            missing_images = set(label_by_stem) - set(image_by_stem)

            if missing_labels or missing_images:
                raise ValueError(
                    f"{split}: {len(missing_labels)} images lack labels; "
                    f"{len(missing_images)} labels lack images. "
                    "Resolve these before conversion."
                )

            split_boxes = 0
            split_polygons = 0
            split_empty = 0

            for stem, image_path in image_by_stem.items():
                label_path = label_by_stem[stem]
                converted_lines = []

                for line_number, line in enumerate(
                    label_path.read_text(encoding="utf-8").splitlines(),
                    start=1,
                ):
                    if not line.strip():
                        continue

                    parts = line.split()
                    converted = convert_label_line(
                        line, label_path, line_number
                    )
                    converted_lines.append(converted)

                    if len(parts) == 5:
                        split_boxes += 1
                    else:
                        split_polygons += 1

                if not converted_lines:
                    split_empty += 1

                shutil.copy2(image_path, output_images / image_path.name)
                (output_labels / f"{stem}.txt").write_text(
                    "\n".join(converted_lines) + ("\n" if converted_lines else ""),
                    encoding="utf-8",
                )

            print(
                f"{split}: {len(image_by_stem)} images, "
                f"{split_boxes} boxes retained, "
                f"{split_polygons} polygons converted, "
                f"{split_empty} empty labels"
            )

            total_images += len(image_by_stem)
            total_boxes += split_boxes
            total_polygons += split_polygons
            total_empty += split_empty

        yaml_text = (
            f"path: {OUTPUT.as_posix()}\n"
            "train: images/train\n"
            "val: images/valid\n"
            "test: images/test\n"
            f"nc: {len(CLASS_NAMES)}\n"
            "names:\n"
            + "".join(
                f"  {index}: {name}\n"
                for index, name in enumerate(CLASS_NAMES)
            )
        )

        (OUTPUT / "data.yaml").write_text(
            yaml_text, encoding="utf-8"
        )

    except Exception:
        # Remove only the newly created output from this failed run.
        if OUTPUT.exists():
            shutil.rmtree(OUTPUT)
        raise

    print("\nCONVERSION COMPLETE")
    print(f"Output: {OUTPUT}")
    print(f"Total images: {total_images}")
    print(f"Original box annotations retained: {total_boxes}")
    print(f"Polygon annotations converted: {total_polygons}")
    print(f"Empty label files: {total_empty}")
    print(f"Class count: {len(CLASS_NAMES)}")
    print(f"Configuration: {OUTPUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
