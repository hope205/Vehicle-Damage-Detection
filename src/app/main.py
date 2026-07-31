import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from src.core.config import settings
from src.app.schema.schemas import HealthResponse, PredictionResponse
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



# 1. CORS is Locked Down
# Retrieve allowed origins from settings, or default to specific safe domains.
ALLOWED_ORIGINS = getattr(settings, "allowed_origins", ["http://localhost:3000", "http://127.0.0.1:3000"])




app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"], # Restrict methods to only what is needed
    allow_headers=["Content-Type", "Authorization"], # Restrict headers
)



async def _read_and_validate_file(file: UploadFile) -> bytes:
    """
    Reads file in chunks to enforce size limit and prevent memory exhaustion.
    """
    # Fast first-pass check (Content-Type header)
    if file.content_type not in settings.model.allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type header: {file.content_type}",
        )

    max_size_bytes = settings.model.max_image_size_mb * 1024 * 1024
    image_bytes = bytearray()
    
    # 2. Chunked Reading for Size Validation
    # We read in 1MB chunks so we can abort immediately if the file is too large,
    # rather than loading a 5GB file into RAM before checking its size.
    chunk_size = 1024 * 1024 
    while chunk := await file.read(chunk_size):
        image_bytes.extend(chunk)
        if len(image_bytes) > max_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, 
                detail=f"Image exceeds maximum allowed size of {settings.model.max_image_size_mb}MB."
            )
            
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file uploaded.")

    return bytes(image_bytes)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=predictor.is_loaded)


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    '''
    Accepts an image file and returns detected damages with bounding boxes and confidence scores.
    '''
    image_bytes = await _read_and_validate_file(file)

    try:
        return predictor.predict(image_bytes, filename=file.filename or "upload")
    except ValueError as e:
        # 3. Handle Corrupt/Spoofed Payloads as 400 Bad Request
        logger.warning("Invalid image payload rejected.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process image.")


@app.post("/predict/annotated", tags=["Inference"])
async def predict_annotated(file: UploadFile = File(...)) -> Response:
    '''
    Returns the uploaded image with bounding boxes drawn around detected damages.
    '''
    image_bytes = await _read_and_validate_file(file)

    try:
        annotated_bytes = predictor.predict_annotated(image_bytes)
        return Response(content=annotated_bytes, media_type="image/jpeg")
    except ValueError as e:
        logger.warning("Invalid image payload rejected.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        logger.exception("Annotated prediction failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process image.")