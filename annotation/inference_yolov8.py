import os
import json
from ultralytics import YOLO
from PIL import Image
import argparse

# Mapping class names to indices for filtering
CLASS_NAMES = {0: "person", 1: "car"}

def yolo_to_coco(yolo_results, image_id, start_annotation_id):
    coco_annotations = []
    annotation_id = start_annotation_id

    for result in yolo_results:
        boxes = result.boxes.xywh.cpu().numpy()  
        scores = result.boxes.conf.cpu().numpy()  
        class_ids = result.boxes.cls.cpu().numpy()  

        for i, box in enumerate(boxes):
            if int(class_ids[i]) not in CLASS_NAMES:
                continue  

            x_center, y_center, width, height = box
            x_min = x_center - (width / 2)
            y_min = y_center - (height / 2)
            x_min = int(x_min)
            y_min = int(y_min)
            width = int(width)
            height = int(height)

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(class_ids[i]),
                "bbox": [x_min, y_min, width, height],
                "score": float(scores[i]),
                "area": float(width * height),
                "iscrowd": 0
            }
            coco_annotations.append(annotation)
            annotation_id += 1

    return coco_annotations, annotation_id

def inference_yolov8_to_coco(image_dir, model_path, output_path):
    model = YOLO(model_path).to('mps')
    
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "person"},
            {"id": 1, "name": "car"}
        ]
    }

    annotation_id = 1
    images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

    for image_index, image_filename in enumerate(images):
        image_path = os.path.join(image_dir, image_filename)
        image = Image.open(image_path)

        results = model(image)

        coco_annotations, annotation_id = yolo_to_coco(results, image_index, annotation_id)

        coco_output["images"].append({
            "id": image_index,
            "file_name": image_filename,
            "width": image.width,
            "height": image.height
        })
        coco_output["annotations"].extend(coco_annotations)

    with open(output_path, 'w') as f:
        json.dump(coco_output, f, indent=4)

    print(f"COCO annotations saved to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, help='Path to the directory with images')
    parser.add_argument('--model_path', type=str, help='Path to the YOLOv5 model')
    parser.add_argument('--output_path', type=str, help='Path to save the COCO annotations')


if __name__ == "__main__":
    args = parse_args()

    inference_yolov8_to_coco(
        args.image_dir, 
        args.model_path, 
        args.output_path
    )
