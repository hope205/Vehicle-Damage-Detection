from pathlib import Path
import pandas as pd
from ultralytics import YOLO
from core.config import settings

MODEL_PATH = Path("runs/dent_scratch_detector/weight/best.ptt")
OUTPUT_DIR = Path("runs/evaluation/error_analysis")
CSV_PATH = OUTPUT_DIR / "error_cases.csv"




CONFIDENCE_THRESHOLD = 0.25
LOW_CONFIDENCE_THRESHOLD = 0.50

def get_validation_images() -> list[Path]:
    """Get all validation images from the dataset configuration."""
    val_path = Path(settings.data.val)
    if not val_path.exists():
        raise FileNotFoundError(f"Validation path not found: {val_path}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in val_path.rglob("*") if p.suffix.lower() in image_extensions])

def analyse_image(model: YOLO, image_path: Path) -> dict:
    """Run YOLO inference on one validation image and identify potential error cases."""
    results = model.predict(
        source=str(image_path),
        conf=CONFIDENCE_THRESHOLD,
        device=settings.training.device,
        verbose=False,
    )
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return {
            "image": str(image_path),
            "error_type": "no_prediction",
            "num_predictions": 0,
            "classes": "",
            "confidences": "",
        }

    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    class_names = settings.data.names
    predicted_classes = [class_names[class_id] for class_id in class_ids]

    low_confidence = any(conf < LOW_CONFIDENCE_THRESHOLD for conf in confidences)

    if low_confidence:
        error_type = "low_confidence"
    elif len(class_ids) > 1:
        error_type = "multiple_predictions"
    else:
        error_type = "prediction"

    return {
        "image": str(image_path),
        "error_type": error_type,
        "num_predictions": len(class_ids),
        "classes": ", ".join(predicted_classes),
        "confidences": ", ".join(f"{conf:.4f}" for conf in confidences),
    }

def save_error_image(model: YOLO, image_path: Path, error_type: str) -> None:
    """Generate and save an annotated prediction image."""
    output_dir = OUTPUT_DIR / "visualizations" / error_type
    output_dir.mkdir(parents=True, exist_ok=True)

    model.predict(
        source=str(image_path),
        conf=CONFIDENCE_THRESHOLD,
        device=settings.training.device,
        save=True,
        project=str(OUTPUT_DIR / "visualizations"),
        name=error_type,
        exist_ok=True,
        verbose=False,
    )

def run_error_analysis():
    """Run simplified image-level error analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    print("\nLoading validation images...")
    image_paths = get_validation_images()
    print(f"Found {len(image_paths)} validation images.")

    analysis_results = []
    for index, image_path in enumerate(image_paths, start=1):
        print(f"\rAnalysing {index}/{len(image_paths)}: {image_path.name}", end="")
        try:
            result = analyse_image(model=model, image_path=image_path)
            analysis_results.append(result)

            if result["error_type"] in {"no_prediction", "low_confidence"}:
                save_error_image(model=model, image_path=image_path, error_type=result["error_type"])
        except Exception as error:
            print(f"\nFailed to analyse {image_path}: {error}")
    print()

    dataframe = pd.DataFrame(analysis_results)
    dataframe.to_csv(CSV_PATH, index=False)

    print("\nError analysis completed.")
    print(f"Results saved to:\n{CSV_PATH}")

    print("\n" + "=" * 50)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 50)
    print("\nCases by category:\n", dataframe["error_type"].value_counts().to_string())
    print("\nPotential no-prediction cases:", len(dataframe[dataframe["error_type"] == "no_prediction"]))
    print("Low-confidence cases:", len(dataframe[dataframe["error_type"] == "low_confidence"]))
    print(f"\nVisualisations saved to:\n{OUTPUT_DIR / 'visualizations'}")

if __name__ == "__main__":
    run_error_analysis()