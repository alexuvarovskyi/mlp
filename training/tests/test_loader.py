import pytest
import torch
import numpy as np
from torch.utils.data import Dataset
from unittest.mock import Mock

from src.dataloader import PyTorchDetectionDataset, get_augmentations 

class MockDetectionDataset(Dataset):
    def __init__(self):
        self.data = [
            (0, np.expand_dims(np.random.rand(224, 224, 3), 0), Mock(xyxy=np.array([[10 / 224, 20 / 224, 30 / 224, 40 / 224]]), class_id=np.array([0]))),
            (1, np.expand_dims(np.random.rand(224, 224, 3), 0), Mock(xyxy=np.array([[50 / 224, 60 / 224, 70 / 224, 80 / 224]]), class_id=np.array([1]))),
            
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class MockProcessor:
    def __call__(self, images, annotations, return_tensors):
        return {"pixel_values": torch.tensor(images).permute(0, 3, 1, 2).float(), "labels": annotations["annotations"]}

def test_pytorch_detection_dataset_len():
    dataset = MockDetectionDataset()
    processor = MockProcessor()
    pytorch_dataset = PyTorchDetectionDataset(dataset, processor)
    
    assert len(pytorch_dataset) == len(dataset)

def test_pytorch_detection_dataset_getitem():
    dataset = MockDetectionDataset()
    processor = MockProcessor()
    transform = get_augmentations("val")
    
    pytorch_dataset = PyTorchDetectionDataset(dataset, processor, transform)
    result = pytorch_dataset[0]

    
    assert isinstance(result, dict)
    assert "pixel_values" in result
    assert "labels" in result
    assert result["pixel_values"].shape == torch.Size((3, 224, 224)) 

def test_annotations_as_coco():
    dataset = MockDetectionDataset()
    processor = MockProcessor()
    transform = get_augmentations("train")
    
    pytorch_dataset = PyTorchDetectionDataset(dataset, processor, transform)
    
    annotations = pytorch_dataset.annotations_as_coco(0, [1], [[10, 20, 30, 40]])
    expected_annotations = {
        "image_id": 0,
        "annotations": [{
            "image_id": 0,
            "category_id": 1,
            "bbox": [10, 20, 20, 20],  # xywh
            "iscrowd": 0,
            "area": 400
        }]
    }
    
    assert annotations == expected_annotations

if __name__ == "__main__":
    pytest.main()
