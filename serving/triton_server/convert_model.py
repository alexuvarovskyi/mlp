import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection
)
        
CHECKPOINT = "./rtdetr_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
model = AutoModelForObjectDetection.from_pretrained(CHECKPOINT).to(DEVICE)
processor = AutoImageProcessor.from_pretrained(CHECKPOINT) 
model.eval()
        
dummy_input = torch.randn(1, 3, 640, 640).to(DEVICE)

torch.onnx.export(
    model, 
    dummy_input, 
    "rt-detr_model.onnx",
    opset_version=18, 
    input_names=["input"],
    output_names=["output"]
)