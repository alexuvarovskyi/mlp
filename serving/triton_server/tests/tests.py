import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from PIL import Image
import io

from fastapi_server import app, MODEL_LABEL_MAPPING

client = TestClient(app)


@pytest.fixture
def sample_image():
    """Creates a sample image for testing."""
    img = Image.new("RGB", (100, 100), color="white")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes


def test_health_check():
    """Test the / health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status_code"] == 200
    assert "model_loaded" in response.json()["data"]


@patch("fastapi_server.predict")
def test_prediction_success(mock_predict, sample_image):
    """Test the /predict/ route with a valid image."""
    mock_predict.return_value = {
        "scores": [0.9],
        "labels": [0],  # corresponding to 'person'
        "boxes": [[10, 20, 30, 40]]
    }

    files = {"image": ("test_image.jpg", sample_image, "image/jpeg")}
    response = client.post("/predict/?threshold=0.5", files=files)

    assert response.status_code == 200
    data = response.json()["predictions"]
    assert len(data) == 1
    assert data[0]["class"] == MODEL_LABEL_MAPPING[0]
    assert data[0]["score"] == 0.9
    assert data[0]["box"] == [10, 20, 30, 40]


def test_invalid_image():
    invalid_file = io.BytesIO(b"invalid data")

    files = {"image": ("invalid_image.jpg", invalid_file, "image/jpeg")}
    response = client.post("/predict/?threshold=0.5", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image file"


@patch("fastapi_server.AutoModelForObjectDetection.from_pretrained")
@patch("fastapi_server.AutoImageProcessor.from_pretrained")
def test_model_loading_success(mock_processor, mock_model):
    """Test that the model and processor are loaded correctly."""
    mock_model.return_value = MagicMock()
    mock_processor.return_value = MagicMock()

    from fastapi_server import model_loaded
    assert model_loaded == True


@patch("fastapi_server.AutoModelForObjectDetection.from_pretrained", side_effect=Exception("Model loading failed"))
def test_model_loading_failure(mock_model):
    from fastapi_server import model_loaded
    assert model_loaded == False
