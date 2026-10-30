
from pathlib import Path
from ultralytics import YOLO


# Project paths
PROJECT_ROOT = Path(r"D:\driver-safety-system")

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lost_found_yolo11n.pt"
)

DATA_YAML = (
    PROJECT_ROOT
    / "datasets"
    / "custom_objects"
    / "lost_found_detection"
    / "data.yaml"
)

OUTPUT_DIR = PROJECT_ROOT / "runs"

EVALUATION_NAME = "lost_found_test_evaluation"


def main():
    print("=" * 60)
    print("CUSTOM YOLO MODEL - TEST DATASET EVALUATION")
    print("=" * 60)

    # Check required files
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not DATA_YAML.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_YAML}"
        )

    # Check test images directory
    test_images = (
        DATA_YAML.parent
        / "images"
        / "test"
    )

    if not test_images.is_dir():
        raise FileNotFoundError(
            f"Test images directory not found: {test_images}"
        )

    # Prevent accidental overwriting of previous results
    evaluation_folder = OUTPUT_DIR / EVALUATION_NAME

    if evaluation_folder.exists():
        raise FileExistsError(
            "Evaluation output already exists. "
            f"Choose a new name before proceeding: {evaluation_folder}"
        )

    print(f"\nModel: {MODEL_PATH}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Test images: {test_images}")
    print(f"Output: {evaluation_folder}")

    print("\nLoading trained model...")

    model = YOLO(str(MODEL_PATH))

    print("Model loaded successfully.")

    print("\nStarting evaluation on the TEST split...")
    print("Device: CPU")
    print("Image size: 416")
    print("Batch size: 4")

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=416,
        batch=4,
        device="cpu",
        workers=0,
        project=str(OUTPUT_DIR),
        name=EVALUATION_NAME,
        exist_ok=False,
        plots=True,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("TEST EVALUATION SUMMARY")
    print("=" * 60)

    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"mAP@50:     {metrics.box.map50:.4f}")
    print(f"mAP@50-95:  {metrics.box.map:.4f}")

    print("\nEvaluation completed successfully.")

    print(f"\nResults saved in:\n{evaluation_folder}")


if __name__ == "__main__":
    main()