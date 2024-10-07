import os
import torch
import supervision as sv
import albumentations as A

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from transformers import AutoModelForObjectDetection, AutoImageProcessor


def save_predictions(predictions, ann_save_path):
    ann_save_path = Path(ann_save_path)
    ann_save_path.mkdir(exist_ok=True, parents=True)
    for img_name, pred in predictions.items():
        ann_name = img_name.split(".")[0] + ".txt"
        ann_path = ann_save_path / ann_name
        with open(ann_path, "w") as f:
            for bbox in pred:
                f.write(" ".join(map(str, bbox)) + "\n")


def postprocess_detections(detections: sv.Detections) -> list[list]:
    predictions = []
    for bbox, class_id in zip(detections.xyxy, detections.class_id):
        predictions.append([
            *list(bbox),
            class_id,
        ])
    return predictions


def inference_model(
    model_path: str,
    data_path: str,
    ann_save_path: str,
    conf_thresh: float = 0.2,
):
    processor = AutoImageProcessor.from_pretrained(model_path)

    id2label = {0: 'person', 1: 'car', 2: 'pet'}
    label2id = {'person': 0, 'car': 1, 'pet': 2}

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForObjectDetection.from_pretrained(
        model_path,
        id2label=id2label,
        label2id=label2id,
        anchor_image_size=None,
        ignore_mismatched_sizes=True,
    ).to(device)
    data_path = Path(data_path)
    ann_save_path = Path(ann_save_path)

    predictions = {}

    for img_path in tqdm(data_path.iterdir()):
        if not os.path.exists(img_path):
            continue
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error while opening image {img_path}: {e}")
            continue
        inputs = processor(image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        w, h = image.size
        results = processor.post_process_object_detection(
            outputs, target_sizes=[(h, w)], threshold=conf_thresh)

        detections = sv.Detections.from_transformers(results[0])

        predictions[img_path.name] = postprocess_detections(detections)

    save_predictions(predictions, ann_save_path)

