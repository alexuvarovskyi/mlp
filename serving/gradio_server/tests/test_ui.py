import pytest
from PIL import Image, ImageDraw
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from gradio_ui import load_model_and_processor, predict, draw_boxes_pillow 

@pytest.fixture(scope='module')
def setup_model():
    model_path = "./rtdetr_model"  
    model, processor = load_model_and_processor(model_path)
    yield model, processor

def test_load_model_and_processor(setup_model):
    model, processor = setup_model
    assert isinstance(model, RTDetrForObjectDetection)
    assert isinstance(processor, RTDetrImageProcessor)

def test_predict(setup_model):
    model, processor = setup_model
    dummy_image = Image.new('RGB', (224, 224), color='white')
    threshold = 0.5

    results = predict(dummy_image, threshold, model, processor)
    assert "scores" in results
    assert "labels" in results
    assert "boxes" in results
    assert len(results["scores"]) == len(results["labels"]) == len(results["boxes"])

def test_draw_boxes_pillow(setup_model):
    model, processor = setup_model
    dummy_image = Image.new('RGB', (224, 224), color='white')
    
    results = {
        "scores": torch.tensor([0.9, 0.8]),
        "labels": torch.tensor([0, 1]),  
        "boxes": torch.tensor([[10, 10, 100, 100], [150, 150, 200, 200]])
    }

    image_with_boxes = draw_boxes_pillow(dummy_image.copy(), results)

    assert image_with_boxes != dummy_image

if __name__ == "__main__":
    pytest.main()
