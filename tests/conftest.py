import io
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.app.main import app
from src.app.services.predictive_service import predictor


@pytest.fixture(scope="session", autouse=True)
def restore_stock_pil_image_open() -> Iterator[None]:
    """
    Ultralytics monkey-patches PIL.Image.open and may try to auto-install pi-heif
    on UnidentifiedImageError. Restore stock open so corrupt-upload tests stay deterministic.
    """
    try:
        from ultralytics.utils import patches

        original = getattr(patches, "_image_open", None)
        if original is not None:
            Image.open = original
    except Exception:
        pass
    yield


@pytest.fixture
def tiny_jpeg_bytes() -> bytes:
    """Minimal valid JPEG used across API and service tests."""
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), color=(120, 40, 40)).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def mock_yolo_model() -> MagicMock:
    """YOLO-like object that returns one dent detection."""
    box = MagicMock()
    box.cls = [0]
    box.conf = [0.91]
    box.xyxy = [[10.0, 20.0, 30.0, 40.0]]

    result = MagicMock()
    result.boxes = [box]

    # Prefer numpy when available (ultralytics dependency); otherwise skip plot usage.
    try:
        import numpy as np

        result.plot.return_value = np.zeros((16, 16, 3), dtype=np.uint8)
    except ImportError:  # pragma: no cover
        result.plot.side_effect = RuntimeError("numpy required for annotated plot tests")

    model = MagicMock()
    model.names = {0: "dent", 1: "scratch", 2: "clean"}
    model.predict.return_value = [result]
    return model


@pytest.fixture
def client(mock_yolo_model: MagicMock) -> Iterator[TestClient]:
    """
    FastAPI TestClient with lifespan load() mocked so real .pt weights are never loaded.
    """
    previous_model = predictor.model
    original_load = predictor.load

    def fake_load() -> None:
        predictor.model = mock_yolo_model

    predictor.load = fake_load  # type: ignore[method-assign]
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        predictor.load = original_load  # type: ignore[method-assign]
        predictor.model = previous_model
