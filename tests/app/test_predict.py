import pytest

from src.core.config import settings


def test_predict_returns_detections(client, tiny_jpeg_bytes):
    response = client.post(
        "/predict",
        files={"file": ("car.jpg", tiny_jpeg_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "car.jpg"
    assert payload["image_width"] == 64
    assert payload["image_height"] == 48
    assert payload["inference_time_ms"] >= 0
    assert len(payload["detections"]) == 1

    detection = payload["detections"][0]
    assert detection["class_name"] == "dent"
    assert detection["confidence"] == pytest.approx(0.91)
    assert detection["bbox"] == {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0}


def test_predict_rejects_unsupported_content_type(client, tiny_jpeg_bytes):
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", tiny_jpeg_bytes, "text/plain")},
    )

    assert response.status_code == 415


def test_predict_rejects_empty_file(client):
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )

    assert response.status_code == 400
    assert "Empty file" in response.json()["detail"]


def test_predict_rejects_oversized_file(client):
    max_bytes = settings.model.max_image_size_mb * 1024 * 1024
    oversized = b"\xff\xd8\xff" + (b"a" * (max_bytes + 1))

    response = client.post(
        "/predict",
        files={"file": ("huge.jpg", oversized, "image/jpeg")},
    )

    assert response.status_code == 413


def test_predict_rejects_corrupt_image_payload(client):
    response = client.post(
        "/predict",
        files={"file": ("fake.jpg", b"not-an-image", "image/jpeg")},
    )

    assert response.status_code == 400
    detail = response.json()["detail"].lower()
    assert "corrupt" in detail or "invalid" in detail
