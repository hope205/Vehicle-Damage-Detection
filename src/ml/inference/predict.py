from pathlib import Path
from ultralytics import YOLO
from core.config import settings

MODEL_PATH = Path("runs/train/yolov8n_seg_dent_scratch-2/weights/best.pt")
model = YOLO(str(MODEL_PATH))

def predict(image_path: str | Path) -> list[dict]:
    """Run vehicle damage segmentation on an image."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model.predict(
        source=str(image_path),
        conf=settings.inference.confidence_threshold,
        imgsz=settings.training.image_size,
        device=settings.training.device,
        verbose=False,
    )
    result = results[0]
    predictions = []

    if result.boxes is None or len(result.boxes) == 0:
        return predictions

    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    masks = result.masks.data.cpu().numpy() if result.masks is not None else None

    for index, class_id in enumerate(class_ids):
        predictions.append({
            "class_id": int(class_id),
            "class_name": settings.data.names[int(class_id)],
            "confidence": float(confidences[index]),
            "bounding_box": [float(value) for value in boxes[index]],
            "mask": masks[index] if masks is not None else None,
        })

    return predictions

if __name__ == "__main__":
    image_path = "data/processed/images/val/example.jpg"
    predictions = predict(image_path)

    print("\nVehicle Damage Predictions")
    print("=" * 40)

    if not predictions:
        print("No vehicle damage detected.")
    else:
        for index, prediction in enumerate(predictions, start=1):
            print(f"\nDetection {index}")
            print(f"Class: {prediction['class_name']}")
            print(f"Confidence: {prediction['confidence']:.4f}")
            print(f"Bounding Box: {prediction['bounding_box']}")
            print(f"Segmentation Mask: {'Available' if prediction['mask'] is not None else 'Not available'}")