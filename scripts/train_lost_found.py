
from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(r"D:\driver-safety-system")

WEIGHTS = ROOT / "models" / "yolo11n.pt"
DATA = (
    ROOT
    / "datasets"
    / "custom_objects"
    / "lost_found_detection"
    / "data.yaml"
)

OUTPUT = ROOT / "runs" / "lost_found_training"

def main():
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Missing pretrained weights: {WEIGHTS}")

    if not DATA.is_file():
        raise FileNotFoundError(f"Missing dataset configuration: {DATA}")

    # Prevent accidental reuse of a previous training run.
    if OUTPUT.exists():
        raise FileExistsError(
            f"Training output already exists: {OUTPUT}\n"
            "Choose a new project folder before training again."
        )

    torch.set_num_threads(min(4, torch.get_num_threads()))

    model = YOLO(str(WEIGHTS))

    model.train(
        data=str(DATA),
        epochs=30,
        imgsz=416,
        batch=4,
        device="cpu",
        workers=2,
        patience=7,
        project=str(OUTPUT.parent),
        name=OUTPUT.name,
        exist_ok=False,
        pretrained=True,
        plots=True,
        save=True,
        val=True,
        seed=42,
    )

    print("\nTRAINING FINISHED")
    print(f"Run directory: {OUTPUT}")
    print(f"Best weights: {OUTPUT / 'weights' / 'best.pt'}")
    print(f"Last weights: {OUTPUT / 'weights' / 'last.pt'}")
    print("The original model weights were not modified.")


if __name__ == "__main__":
    main()