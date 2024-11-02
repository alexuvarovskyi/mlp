import pytest
import ray
import ray.serve as serve
from httpx import AsyncClient
from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image
from fastapi_server import app, ObjectDetectionModel

@pytest.fixture(scope="module", autouse=True)
def setup_ray_serve():
    """Fixture to set up Ray and Ray Serve."""
    ray.init(ignore_reinit_error=True) 
    serve.start(detached=True)
    ObjectDetectionModel.bind() 

    yield  # Run tests

    serve.shutdown()
    ray.shutdown()


@pytest.mark.asyncio
async def test_health_check():
    """Test the health check endpoint at `/`."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
    assert response.json()["data"]["model_loaded"] == True


def create_test_image():
    """Creates an in-memory test image."""
    image = Image.new("RGB", (100, 100), color="white")  # Create a simple white image
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    return img_byte_arr


@pytest.mark.asyncio
async def test_predict_endpoint():
    image = create_test_image()

    files = {"image": ("test.jpg", image, "image/jpeg")}

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/predict/", files=files, data={"threshold": "0.5"})

    assert response.status_code == 200

    json_response = response.json()
    assert "predictions" in json_response
    assert isinstance(json_response["predictions"], list)


@pytest.mark.asyncio
async def test_invalid_image_upload():
    """Test uploading an invalid image file."""
    invalid_image_data = BytesIO(b"this is not an image")

    files = {"image": ("test.txt", invalid_image_data, "text/plain")}

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/predict/", files=files, data={"threshold": "0.5"})

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image file"
