import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from src.core.config import settings
from src.app.schema import HealthResponse, PredictionResponse
from src.app.services.predictive_service import predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()
    yield


app = FastAPI(
    title="Car Damage Detection API",
    description="Detects dents and scratches on vehicle images using a YOLO model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_upload(file: UploadFile) -> None:
    if file.content_type not in settings.model.allowed_content_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type: {file.content_type}",
        )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=predictor.is_loaded)


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    _validate_upload(file)
    image_bytes = await file.read()

    if len(image_bytes) > settings.max_image_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds maximum allowed size.")

    try:
        return predictor.predict(image_bytes, filename=file.filename or "upload")
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Failed to process image.")


@app.post("/predict/annotated", tags=["Inference"])
async def predict_annotated(file: UploadFile = File(...)) -> Response:
    _validate_upload(file)
    image_bytes = await file.read()

    if len(image_bytes) > settings.max_image_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds maximum allowed size.")

    try:
        annotated_bytes = predictor.predict_annotated(image_bytes)
    except Exception:
        logger.exception("Annotated prediction failed")
        raise HTTPException(status_code=500, detail="Failed to process image.")

    return Response(content=annotated_bytes, media_type="image/jpeg")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)