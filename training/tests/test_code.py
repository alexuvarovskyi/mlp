import torch
import pytest
import numpy as np
import supervision as sv
from transformers.trainer_utils import EvalPrediction
from dataclasses import dataclass

from src.evaluation import MAPEvaluator


@dataclass
class MockImageProcessor:
    def post_process_object_detection(self, output, threshold, target_sizes):
        boxes = output.pred_boxes
        boxes = sv.xcycwh_to_xyxy(np.array(boxes))
        target_sizes = np.array(target_sizes)
        boxes[:, 0] = boxes[:, 0] * target_sizes[:, 0]
        boxes[:, 1] = boxes[:, 1] * target_sizes[:, 1]
        boxes[:, 2] = boxes[:, 2] * target_sizes[:, 0]
        boxes[:, 3] = boxes[:, 3] * target_sizes[:, 1]
        return [{"boxes": torch.Tensor(boxes), 
                 "labels": torch.argmax(output.logits, dim=1), 
                 "scores": torch.max(output.logits, dim=1).values}]

@pytest.fixture
def mock_image_processor():
    return MockImageProcessor()

@pytest.fixture
def mock_evaluation_results():
    predictions = [
    (None, np.array([[0.1, 0.9], [0.8, 0.2]]), np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]]))
]
    targets = [
        [{"size": np.array([100, 100]), "boxes": np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]]), "class_labels": np.array([1, 0])}]
    ]
    return EvalPrediction(predictions=predictions, label_ids=targets)

@pytest.fixture
def map_evaluator(mock_image_processor):
    return MAPEvaluator(image_processor=mock_image_processor, threshold=0.3, id2label={0: "class_0", 1: "class_1"})

def test_map_evaluator_call(map_evaluator, mock_evaluation_results):
    metrics = map_evaluator(mock_evaluation_results)
    print(metrics)
    assert metrics["map_class_0"] == 1.0
    assert metrics["map_class_1"] == 1.0


if __name__ == "__main__":
    pytest.main()

