def test_predict_annotated_returns_jpeg(client, tiny_jpeg_bytes):
    response = client.post(
        "/predict/annotated",
        files={"file": ("car.jpg", tiny_jpeg_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"  # JPEG SOI marker


def test_predict_annotated_rejects_corrupt_image(client):
    response = client.post(
        "/predict/annotated",
        files={"file": ("fake.jpg", b"not-an-image", "image/jpeg")},
    )

    assert response.status_code == 400
