import pytest

from src.app.services.predictive_service import predictor


def test_validate_and_open_image_accepts_jpeg(tiny_jpeg_bytes, mock_yolo_model):
    predictor.model = mock_yolo_model
    image = predictor._validate_and_open_image(tiny_jpeg_bytes)

    assert image.size == (64, 48)
    assert image.mode == "RGB"


def test_validate_and_open_image_rejects_garbage():
    with pytest.raises(ValueError, match="corrupt|invalid|supported"):
        predictor._validate_and_open_image(b"definitely-not-an-image")


def test_predict_requires_loaded_model(tiny_jpeg_bytes):
    previous = predictor.model
    predictor.model = None
    try:
        with pytest.raises(RuntimeError, match="not loaded"):
            predictor.predict(tiny_jpeg_bytes)
    finally:
        predictor.model = previous


def test_predict_builds_response(tiny_jpeg_bytes, mock_yolo_model):
    predictor.model = mock_yolo_model

    response = predictor.predict(tiny_jpeg_bytes, filename="sample.jpg")

    assert response.filename == "sample.jpg"
    assert response.image_width == 64
    assert response.image_height == 48
    assert len(response.detections) == 1
    assert response.detections[0].class_name == "dent"
    assert response.detections[0].confidence == pytest.approx(0.91)
    assert response.detections[0].bbox.x1 == 10.0
    mock_yolo_model.predict.assert_called()
    _, kwargs = mock_yolo_model.predict.call_args
    assert kwargs.get("verbose") is False


def test_predict_annotated_returns_jpeg_bytes(tiny_jpeg_bytes, mock_yolo_model):
    pytest.importorskip("numpy")
    predictor.model = mock_yolo_model

    annotated = predictor.predict_annotated(tiny_jpeg_bytes)

    assert isinstance(annotated, bytes)
    assert annotated[:2] == b"\xff\xd8"


def test_load_is_idempotent(mock_yolo_model, monkeypatch, tmp_path):
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"fake-weights")

    calls = {"count": 0}

    def fake_yolo(_path: str):
        calls["count"] += 1
        return mock_yolo_model

    monkeypatch.setattr(
        "src.app.services.predictive_service.YOLO",
        fake_yolo,
    )

    # Use a fresh instance path via the singleton carefully
    previous_model = predictor.model
    previous_path = predictor.model_path
    predictor.model = None
    predictor.model_path = weights
    try:
        predictor.load()
        predictor.load()
        assert calls["count"] == 1
        assert predictor.is_loaded is True
    finally:
        predictor.model = previous_model
        predictor.model_path = previous_path
