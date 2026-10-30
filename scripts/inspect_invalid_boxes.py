
from pathlib import Path

ROOT = Path(r"D:\driver-safety-system")
DATASET = ROOT / "datasets" / "custom_objects" / "lost_found_detection"

SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    total = 0

    for split in SPLITS:
        image_dir = DATASET / "images" / split
        label_dir = DATASET / "labels" / split

        images = {
            p.stem: p
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        }

        for label_path in label_dir.glob("*.txt"):
            image_path = images.get(label_path.stem)
            if image_path is None:
                continue

            for line_no, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), 1
            ):
                parts = line.split()
                if len(parts) != 5:
                    continue

                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])

                left = x - w / 2
                top = y - h / 2
                right = x + w / 2
                bottom = y + h / 2

                if (
                    left < -1e-5 or top < -1e-5
                    or right > 1 + 1e-5
                    or bottom > 1 + 1e-5
                ):
                    total += 1
                    print(f"\n{split}: {label_path.name}, line {line_no}")
                    print(f"Class ID: {class_id}")
                    print(f"Image: {image_path.name}")
                    print(f"Box: x={x:.8f}, y={y:.8f}, w={w:.8f}, h={h:.8f}")
                    print(
                        f"Edges: left={left:.8f}, top={top:.8f}, "
                        f"right={right:.8f}, bottom={bottom:.8f}"
                    )

    print(f"\nTotal out-of-bounds boxes found: {total}")


if __name__ == "__main__":
    main()