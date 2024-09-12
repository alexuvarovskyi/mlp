import os
import torch
import argparse
import supervision as sv
import albumentations as A

from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForObjectDetection, AutoImageProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--ann_path", type=str, required=True)
    return parser.parse_args()


def evaluate(
    model_path: str,
    data_path: str,
    ann_path: str,
):
    processor = AutoImageProcessor.from_pretrained(model_path)

    ds_valid, _ = sv.DetectionDataset.from_coco(
        images_directory_path=data_path,
        annotations_path=ann_path,
    )


    id2label = {id: label for id, label in enumerate(ds_valid.classes)}
    label2id = {label: id for id, label in enumerate(ds_valid.classes)}

    device = torch.device('cuda:0')
    model = AutoModelForObjectDetection.from_pretrained(
        model_path,
        id2label=id2label,
        label2id=label2id,
        anchor_image_size=None,
        ignore_mismatched_sizes=True,
    ).to(device)

    targets = []
    predictions = []

    for i in tqdm(range(len(ds_valid))):
        path, sourece_image, annotations = ds_valid[i]
        if not os.path.exists(path):
            continue
        image = Image.open(path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        w, h = image.size
        results = processor.post_process_object_detection(
            outputs, target_sizes=[(h, w)], threshold=0.2)

        detections = sv.Detections.from_transformers(results[0])

        targets.append(annotations)
        predictions.append(detections)


    mean_average_precision = sv.MeanAveragePrecision.from_detections(
        predictions=predictions,
        targets=targets,
    )


    print(f"map50_95: {mean_average_precision.map50_95:.2f}")
    print(f"map50: {mean_average_precision.map50:.2f}")
    print(f"map75: {mean_average_precision.map75:.2f}")



if __name__ == "__main__":
    args = parse_args()
    evaluate(args.checkpoint, args.data_path, args.ann_path)