from pathlib import Path
from ultralytics import YOLO
from core.config import settings
from ml.evaluation.metrics import extract_all_metrics

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = Path("runs/dent_scratch_detector/weight/best.pt")
DATA_PATH = Path("data_carrd/filtered/data.yaml")
CLASS_NAMES = settings.data.classes

# Directory where evaluation results and plots will be saved
# EVALUATION_DIR = Path("evaluations")
EVALUATION_NAME = "vehicle_damage_evaluation"



def evaluate_model():
    """
    Evaluate the trained YOLO detection model.

    Generates:
    - Overall box/detection metrics
    - Per-class box metrics
    - Confusion matrix
    - Normalized confusion matrix
    - Precision-Recall curve
    - F1 curve
    - Precision curve
    - Recall curve
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
        project=str(EVALUATION_NAME),
        name=EVALUATION_NAME,
        exist_ok=True,
        
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

  
    # Evaluation Output
   

    evaluation_path = EVALUATION_NAME

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 60)


    return results, metrics

if __name__ == "__main__":
    evaluate_model()