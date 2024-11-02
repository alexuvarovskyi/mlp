import os
from PIL import Image
import torch
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from io import BytesIO
from typing import Dict, List
from PIL import UnidentifiedImageError
import base64

device = torch.device("cpu")

model_path = "./rtdetr_model"
MODEL_LABEL_MAPPING = {0: "person", 1: "car", 2: "pet"}

class ObjectDetectionModel:
    def __init__(self):
        try:
            self.model = AutoModelForObjectDetection.from_pretrained(model_path).to(device).eval()
            self.processor = AutoImageProcessor.from_pretrained(model_path)
            self.model_loaded = True
        except Exception as e:
            self.model_loaded = False
            print(f"Error loading model: {e}")
            raise e 

    def predict(self, image_data: bytes, threshold: float = 0.5) -> Dict:
        try:
            image = Image.open(BytesIO(image_data))

            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = self.model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]]) 
            results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
            results = {k: v.detach().cpu() for k, v in results.items()}

            output_data = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                output_data.append({
                    "class": MODEL_LABEL_MAPPING[label.item()],
                    "score": score.item(),
                    "box": [box[0].item(), box[1].item(), box[2].item(), box[3].item()]
                })

            return {"predictions": output_data}

        except UnidentifiedImageError:
            return {"error": "Invalid image file"}
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {"error": str(e)} 

    def health_status(self) -> Dict:
        return {
            "message": "Model is loaded" if self.model_loaded else "Model loading failed",
            "status": "OK" if self.model_loaded else "ERROR"
        }

model_instance = ObjectDetectionModel()

def predict_raw(input_data: Dict) -> Dict:
    try:
        image_data_base64 = input_data.get("data").get("ndarray")[0].get("image_bytes")
        image_data = base64.b64decode(image_data_base64)

        threshold = input_data.get("parameters", {}).get("threshold", 0.5)

        return model_instance.predict(image_data, threshold)

    except Exception as e:
        print(f"Error in predict_raw: {e}")
        return {"error": str(e)}

def health() -> Dict:
    return model_instance.health_status()
