import os
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
from fastapi_server import app  # Adjust if your FastAPI app is in a different file

client = TestClient(app)

@pytest.fixture
def test_image():
    # Create a simple 100x100 red image for testing
    image = Image.new("RGB", (100, 100), color="red")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr

def test_index():
    """Test the health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status_code"] == 200
    assert data["message"] == "OK"
    assert "model_loaded" in data["data"]

# @pytest.mark.skipif(os.environ.get("MODEL_AVAILABLE") != "1", reason="Model not available")
def test_predict(test_image):
    """Test the prediction endpoint with an example image."""
    # Simulate sending the image as form data
    files = {'image': ('test_image.jpg', test_image, 'image/jpeg')}
    response = client.post("/predict/", files=files, data={"threshold": "0.5"})
    
    assert response.status_code == 200
    data = response.json()

    # Ensure the response contains the predictions
    assert "predictions" in data
    for prediction in data["predictions"]:
        assert "class" in prediction
        assert "score" in prediction
        assert "box" in prediction
        assert len(prediction["box"]) == 4  # Ensure the box has 4 coordinates


def test_predict_invalid_file():
    """Test prediction with invalid file input."""
    files = {'image': ('test_image.txt', io.BytesIO(b"not an image"), 'text/plain')}
    response = client.post("/predict/", files=files, data={"threshold": "0.5"})

    # Expecting a 400 Bad Request for invalid image input
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image file"
