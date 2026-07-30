import os
from pathlib import Path
from ultralytics import YOLO
from src.core.config import settings
from src.ml.evaluation.metrics import extract_all_metrics
from datetime import datetime
from pathlib import Path
import json



# Generate a timestamp like '20260729_144500'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
EVALUATION_NAME = f"eval_{timestamp}"

ROOT_DIR = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT_DIR / "evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


MODEL_PATH = settings.model.model_path  # Path to the trained YOLO model weights    
DATA_PATH = Path(settings.data.filtered_data_path)
CLASS_NAMES = settings.data.classes




def evaluate_model():
    """
    Evaluate the trained YOLO detection model.
    """
    print(f"Loading model from: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    print("\nStarting model evaluation...")

    results = model.val(
        data=str(DATA_PATH),
        split="test",
        imgsz=settings.training.image_size,
        batch=settings.training.batch_size,
        device=settings.training.device,
        plots=True,
        project=str(EVAL_DIR),
        name=str(EVALUATION_NAME),
        exist_ok=True,
        conf=settings.model.confidence_threshold,
        iou=settings.model.iou_threshold,
        # iou=0.5,  # 
    
    )

    metrics = extract_all_metrics(results, CLASS_NAMES)

   
    # Overall Metrics
    print("\n" + "=" * 60)
    print("OVERALL EVALUATION METRICS")
    print("=" * 60)

    # Box Metrics
    print("\nBOX / DETECTION METRICS")
    print("-" * 30)
    box_metrics = metrics["overall"]["box"]
    print(f"Precision:     {box_metrics['precision']:.4f}")
    print(f"Recall:        {box_metrics['recall']:.4f}")
    print(f"F1 Score:      {box_metrics['f1_score']:.4f}")
    print(f"mAP50:         {box_metrics['map50']:.4f}")
    print(f"mAP50-95:      {box_metrics['map50_95']:.4f}")

   
    # Per-Class Metrics
    print("\n" + "=" * 60)
    print("PER-CLASS EVALUATION METRICS")
    print("=" * 60)

    for class_name, class_metrics in metrics["per_class"].items():
        print(f"\nCLASS: {class_name.upper()}")
        print("-" * 30)

        # Box Metrics
        print("\nBox / Detection:")
        box = class_metrics["box"]
        print(f"  Precision:     {box['precision']:.4f}")
        print(f"  Recall:        {box['recall']:.4f}")
        print(f"  F1 Score:      {box['f1_score']:.4f}")
        print(f"  mAP50:         {box['map50']:.4f}")
        print(f"  mAP50-95:      {box['map50_95']:.4f}")


    # SAVE METRICS TO JSON

    # Construct the path to the current specific evaluation run
    current_run_dir = EVAL_DIR / EVALUATION_NAME
    
    # Ensure the directory exists (YOLO creates it, but this is a safe fallback)
    current_run_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the target JSON file path
    metrics_file_path = current_run_dir / "metrics.json"
    
    # Save the dictionary as a formatted JSON file
    with open(metrics_file_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\n[INFO] Metrics successfully saved to: {metrics_file_path}")
   


    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 60)


    return results, metrics

if __name__ == "__main__":
    evaluate_model()