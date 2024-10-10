import pytest
from unittest.mock import MagicMock, patch
from PIL import Image, ImageDraw
import torch
import json
from streamlit_ui import (load_model_and_processor, predict, draw_boxes_pillow, 
                              CLASS_COLOR_MAPPING, MODEL_LABEL_MAPPING)

# Mock for the model and processor
@pytest.fixture
def mock_model_and_processor():
    model = MagicMock()
    processor = MagicMock()
    return model, processor

@pytest.fixture
def dummy_image():
    # Create a dummy image for testing
    return Image.new('RGB', (100, 100), color='white')

def test_load_model_and_processor(mock_model_and_processor):
    model, processor = mock_model_and_processor
    model_path = "dummy/model/path"
    
    with patch("transformers.AutoModelForObjectDetection.from_pretrained", return_value=model) as mock_model:
        with patch("transformers.AutoImageProcessor.from_pretrained", return_value=processor) as mock_processor:
            loaded_model, loaded_processor = load_model_and_processor(model_path)
    
            assert loaded_model == model
            assert loaded_processor == processor
            mock_model.assert_called_once_with(model_path)
            mock_processor.assert_called_once_with(model_path)

def test_predict(mock_model_and_processor, dummy_image):
    model, processor = mock_model_and_processor
    threshold = 0.5
    # Mock processor output
    processor.post_process_object_detection.return_value = [{
        "scores": torch.tensor([0.9, 0.8]),
        "labels": torch.tensor([0, 1]),
        "boxes": torch.tensor([[10, 10, 50, 50], [60, 60, 90, 90]])
    }]
    
    results = predict(dummy_image, threshold, model, processor)
    
    assert "scores" in results
    assert "labels" in results
    assert "boxes" in results
    assert len(results["scores"]) == 2
    assert len(results["labels"]) == 2
    assert round(results["scores"][0].item(), 2) == 0.9
    assert results["labels"][0].item() == 0
    assert results["boxes"][0].tolist() == [10, 10, 50, 50]

def test_draw_boxes_pillow(dummy_image):
    results = {
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([0]),
        "boxes": torch.tensor([[10, 10, 50, 50]])
    }

    image_with_boxes = draw_boxes_pillow(dummy_image.copy(), results)
    draw = ImageDraw.Draw(image_with_boxes)

    # Check if the color and text are correct
    class_label = MODEL_LABEL_MAPPING[results["labels"][0].item()]
    expected_color = CLASS_COLOR_MAPPING[class_label]
    assert expected_color == "red"  # For person class
    assert image_with_boxes != dummy_image
