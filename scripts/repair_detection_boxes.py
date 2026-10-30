
from pathlib import Path
import shutil

ROOT = Path(r"D:\driver-safety-system")
DATASET = ROOT / "datasets" / "custom_objects" / "lost_found_detection"
BACKUP = ROOT / "datasets" / "custom_objects" / "lost_found_detection_labels_backup"

SPLITS = ("train", "valid", "test")


def main():
    if not DATASET.is_dir():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    if BACKUP.exists():
        raise FileExistsError(
            f"Backup already exists: {BACKUP}\n"
            "No files changed. Inspect the backup before proceeding."
        )

    label_dirs = [DATASET / "labels" / split for split in SPLITS]

    for folder in label_dirs:
        if not folder.is_dir():
            raise FileNotFoundError(f"Label folder not found: {folder}")

    # Back up all converted label files before making any changes.
    shutil.copytree(DATASET / "labels", BACKUP)

    repaired_boxes = 0
    changed_files = 0

    try:
        for folder in label_dirs:
            for label_path in folder.glob("*.txt"):
                original = label_path.read_text(encoding="utf-8")
                output_lines = []
                file_changed = False

                for line_number, line in enumerate(original.splitlines(), 1):
                    if not line.strip():
                        output_lines.append("")
                        continue

                    parts = line.split()

                    if len(parts) != 5:
                        raise ValueError(
                            f"{label_path}:{line_number}: expected 5 fields"
                        )

                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])

                    left = x - w / 2
                    top = y - h / 2
                    right = x + w / 2
                    bottom = y + h / 2

                    if (
                        left < 0 or top < 0
                        or right > 1 or bottom > 1
                    ):
                        # Reject boxes that extend substantially outside.
                        excess = max(
                            -left, -top, right - 1, bottom - 1, 0.0
                        )

                        if excess > 0.0001:
                            raise ValueError(
                                f"{label_path}:{line_number}: "
                                f"excess {excess:.8f} is too large to auto-repair"
                            )

                        left = max(0.0, min(1.0, left))
                        top = max(0.0, min(1.0, top))
                        right = max(0.0, min(1.0, right))
                        bottom = max(0.0, min(1.0, bottom))

                        w = right - left
                        h = bottom - top
                        x = (left + right) / 2
                        y = (top + bottom) / 2

                        if w <= 0 or h <= 0:
                            raise ValueError(
                                f"{label_path}:{line_number}: "
                                "box became degenerate"
                            )

                        repaired_boxes += 1
                        file_changed = True

                    output_lines.append(
                        f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                    )

                if file_changed:
                    label_path.write_text(
                        "\n".join(output_lines) + "\n",
                        encoding="utf-8",
                    )
                    changed_files += 1

    except Exception:
        # Restore every label file if any repair fails.
        shutil.rmtree(DATASET / "labels")
        shutil.copytree(BACKUP, DATASET / "labels")
        raise

    print("REPAIR COMPLETE")
    print(f"Boxes repaired: {repaired_boxes}")
    print(f"Files changed: {changed_files}")
    print(f"Backup: {BACKUP}")
    print("Original source dataset and model were not modified.")


if __name__ == "__main__":
    main()