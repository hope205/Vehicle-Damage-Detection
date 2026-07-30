import io
import logging
import time
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO
from src.core.config import settings
from src.app.schema.schemas import BoundingBox, Detection, PredictionResponse

logger = logging.getLogger(__name__)       

class DamagePredictor:
    """Singleton wrapper around the YOLO car-damage-detection model."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_path: Path | None = None):
        if getattr(self, "_initialized", False):
            return
        self.model_path = model_path or settings.model.model_path
        self.model: YOLO | None = None
        self._initialized = True

    def load(self) -> None:
        """Load model weights into memory. Safe to call multiple times."""
        if self.model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found at: {self.model_path}")
        logger.info("Loading model from %s", self.model_path)
        self.model = YOLO(str(self.model_path))
        logger.info("Model loaded. Classes: %s", self.model.names)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def _validate_and_open_image(self, image_bytes: bytes) -> Image.Image:
        """
        Validates actual file contents (magic bytes) to prevent spoofing,
        and safely opens the image.
        """
        try:
            img_io = io.BytesIO(image_bytes)
            image = Image.open(img_io)
            
            # 4. Deep Validation
            # .verify() checks file signatures and headers without loading the full raster data.
            # If someone renamed a .txt file to .jpg, this will catch it.
            image.verify() 
            
            # .verify() moves the file pointer. Reset it to 0 to actually read the image.
            img_io.seek(0)
            return Image.open(img_io).convert("RGB")
            
        except (UnidentifiedImageError, SyntaxError, ValueError, TypeError) as e:
            raise ValueError("The uploaded file is corrupt, structurally invalid, or not a supported image format.") from e


    def predict(self, image_bytes: bytes, filename: str = "upload") -> PredictionResponse:
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        start = time.perf_counter()
        
        # Use the secure loading method
        image = self._validate_and_open_image(image_bytes)
        width, height = image.size

        results = self.model.predict(
            source=image,
            verbose=False,
            conf=settings.model.confidence_threshold,
            iou=settings.model.iou_threshold,

        )
        result = results[0]

        detections: list[Detection] = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return PredictionResponse(
            filename=filename,
            image_width=width,
            image_height=height,
            detections=detections,
            inference_time_ms=round(elapsed_ms, 2),
        )
    
    def predict_annotated(self, image_bytes: bytes) -> bytes:
        """Return JPEG bytes of the image with bounding boxes drawn on it."""
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        # Use the secure loading method
        image = self._validate_and_open_image(image_bytes)
        
        results = self.model.predict(
            source=image,
            conf=settings.model.confidence_threshold,
            iou=settings.model.iou_threshold,
            verbose=False,
        )
        annotated_array = results[0].plot()  # returns a BGR numpy array
        annotated_image = Image.fromarray(annotated_array[..., ::-1])  # BGR -> RGB

        buffer = io.BytesIO()
        annotated_image.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()


# Module-level singleton used by the FastAPI app.
predictor = DamagePredictor()