from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    """Pixel coordinates of a detection box, top-left to bottom-right."""
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_name: str = Field(..., description="Detected damage class, e.g. 'dent' or 'scratch'.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox

class PredictionResponse(BaseModel):
    filename: str
    image_width: int
    image_height: int
    detections: list[Detection]
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool