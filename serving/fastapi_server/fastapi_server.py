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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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


def predict(image: Image.Image, threshold: float):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])  # target size in (height, width)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
    results = {k: v.detach().cpu() for k, v in results.items()}

    return results

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
