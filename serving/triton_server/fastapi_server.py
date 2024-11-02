import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from io import BytesIO
from http import HTTPStatus
from typing import Dict

from fastapi import HTTPException
from PIL import UnidentifiedImageError
import tritonclient.http as httpclient
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()


model_path = "./rtdetr_model"

try:
    model = AutoModelForObjectDetection.from_pretrained(model_path).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(model_path)
    model_loaded = True
except Exception as e:
    model_loaded = False
    print(f"Error loading model: {e}")

MODEL_LABEL_MAPPING = {0: "person", 1: "car", 2: "pet"}

@app.get("/")
def _index() -> Dict:
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status_code": HTTPStatus.OK,
        "data": {"model_loaded": model_loaded},
    }
    return response
TRITON_URL = "localhost:8000"
MODEL_NAME = "rtdetr_onnx"

client = httpclient.InferenceServerClient(url=TRITON_URL)

def predict(image: Image.Image, threshold: float):
    image_np = np.array(image).astype(np.float32)
    image_np = np.transpose(image_np, (2, 0, 1))  # Convert to CHW format

    inputs = [
        httpclient.InferInput("input_ids", [1, 3, 800, 800], "FP32"),
    ]
    inputs[0].set_data_from_numpy(np.expand_dims(image_np, axis=0), binary_data=True)

    outputs = [
        httpclient.InferRequestedOutput("boxes", binary_data=True),
        httpclient.InferRequestedOutput("scores", binary_data=True),
        httpclient.InferRequestedOutput("labels", binary_data=True),
    ]

    response = client.infer(MODEL_NAME, inputs, outputs=outputs)

    boxes = response.as_numpy("boxes")
    scores = response.as_numpy("scores")
    labels = response.as_numpy("labels")

    return {"boxes": boxes, "scores": scores, "labels": labels}

@app.post("/predict/")
async def inference(image: UploadFile = File(...), threshold: float = 0.5):
    try:
        image_data = await image.read()
        image = Image.open(BytesIO(image_data))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")

    results = predict(image, threshold)

    output_data = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        output_data.append({
            "class": MODEL_LABEL_MAPPING[label.item()],
            "score": score.item(),
            "box": [box[0].item(), box[1].item(), box[2].item(), box[3].item()]
        })

    return JSONResponse(content={"predictions": output_data})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
