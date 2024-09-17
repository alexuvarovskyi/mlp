import pytest
import torch
from transformers import AutoModelForObjectDetection, RTDetrForObjectDetection, AutoImageProcessor


TEST_CONFIG = {
    "model": {
        "name": "PekingU/rtdetr_r18vd",
        "device": "cpu"
    },
    "data": {
        "size": 512
    }
}


@pytest.fixture
def model_and_processor():
    """Fixture to initialize model and processor."""
    model_name = TEST_CONFIG['model']['name']
    processor = AutoImageProcessor.from_pretrained(
        model_name,
        do_resize=True,
        size={"width": TEST_CONFIG['data']['size'], "height": TEST_CONFIG['data']['size']}
    )
    model = AutoModelForObjectDetection.from_pretrained(
        model_name,
        ignore_mismatched_sizes=True,
    ).to(TEST_CONFIG['model']['device'])
    
    return model, processor

def test_model_initialization(model_and_processor):
    """Test if the model initializes correctly."""
    model, _ = model_and_processor
    assert isinstance(model, RTDetrForObjectDetection), "Model did not initialize correctly."

def test_forward_pass(model_and_processor):
    """Test if the model can perform a forward pass with dummy inputs."""
    model, processor = model_and_processor
    
    dummy_images = torch.randn(2, 3, TEST_CONFIG['data']['size'], TEST_CONFIG['data']['size'])
    
    outputs = model(pixel_values=dummy_images)
    
    assert 'logits' in outputs, "Model output should contain 'logits'."
    assert 'pred_boxes' in outputs, "Model output should contain 'pred_boxes'."
    assert outputs['logits'].shape[0] == 2, "Logits output should match the batch size."

def test_label_mapping():
    """Test if id2label and label2id mappings are correct."""
    id2label = {0: "person", 1: "car"}
    label2id = {v: k for k, v in id2label.items()}
    
    assert label2id["person"] == 0, "Label 'person' should map to 0."
    assert label2id["car"] == 1, "Label 'car' should map to 1."
    assert id2label[0] == "person", "ID 0 should map to 'person'."
    assert id2label[1] == "car", "ID 1 should map to 'car'."


def test_model_output_shape(model_and_processor):
    """Test if the model outputs have the expected shape."""
    model, processor = model_and_processor
    
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, TEST_CONFIG['data']['size'], TEST_CONFIG['data']['size'])
    
    outputs = model(pixel_values=dummy_images)
    
    logits_shape = outputs['logits'].shape
    pred_boxes_shape = outputs['pred_boxes'].shape
    
    assert logits_shape[0] == batch_size, f"Expected batch size {batch_size} in logits, got {logits_shape[0]}."
    assert pred_boxes_shape[0] == batch_size, f"Expected batch size {batch_size} in pred_boxes, got {pred_boxes_shape[0]}."

    num_classes = logits_shape[-1]
    assert num_classes == len(model.config.id2label), f"Expected {len(model.config.id2label)} classes, got {num_classes}."