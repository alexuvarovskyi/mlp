import os
import json
from PIL import Image
from streaming import MDSWriter
import argparse


def yolo_to_streaming_dataset(yolo_dir, output_dir):
    image_dir = os.path.join(yolo_dir, "images")
    label_dir = os.path.join(yolo_dir, "labels")
    
    columns = {
        "image": "jpeg",
        "label": "json",
    }

    with MDSWriter(out=output_dir, columns=columns, compression='zstd') as writer:
        for image_name in os.listdir(image_dir):
            if image_name.endswith(('.png', '.jpg', '.jpeg')):
                # Read the image
                image_path = os.path.join(image_dir, image_name)
                image_data = Image.open(image_path)
                
                # Read the corresponding label
                label_name = os.path.splitext(image_name)[0] + ".txt"
                label_path = os.path.join(label_dir, label_name)
                if os.path.exists(label_path):
                    with open(label_path, "r") as f:
                        label_data = []
                        for line in f:
                            parts = line.strip().split()
                            class_id = int(parts[0])
                            bbox = list(map(float, parts[1:]))
                            label_data.append({"class_id": class_id, "bbox": bbox})
                else:
                    label_data = []

                # Serialize the label data to JSON
                label_json = json.dumps(label_data)
                
                # Add the sample to the dataset
                sample = {
                    "image": image_data,
                    "label": label_json,
                }
                writer.write(sample)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert YOLO dataset to a streaming dataset")
    parser.add_argument("yolo_dir", help="Path to the YOLO dataset directory")
    parser.add_argument("output_dir", help="Path to the output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    yolo_to_streaming_dataset(args.yolo_dir, args.output_dir)
