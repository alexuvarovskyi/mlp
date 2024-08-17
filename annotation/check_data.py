import json
import numpy as np
from cleanlab.object_detection.filter import find_label_issues
from cleanlab.object_detection.rank import get_label_quality_scores
from cleanlab.object_detection.summary import visualize

# Path to your COCO annotations

def generate_mock_predictions(coco_data, num_classes=2, max_boxes_per_image=10):
    """
    Generate mock predictions for a COCO dataset.

    Parameters:
    - coco_data: The COCO dataset in JSON format.
    - num_classes: The number of classes in the dataset.
    - max_boxes_per_image: The maximum number of bounding boxes per image.
    
    Returns:
    - predictions: A list of numpy arrays containing mock predictions for each image.
    """
    predictions = []
    for i, image in enumerate(coco_data['images']):
        predictions.append(np.array([
            np.empty((0, 5), dtype=np.float32),
            np.empty((0, 5), dtype=np.float32),
        ], dtype=object))

    return predictions


# Convert COCO annotations to Cleanlab format
def coco_to_cleanlab(coco_data):
    labels = []
    for image in coco_data['images']:
        image_id = image['id']
        image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
        
        bboxes = []
        class_labels = []
        for ann in image_annotations:
            bbox = ann['bbox']
            # COCO format is [x, y, width, height], convert to [x1, y1, x2, y2]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            bboxes.append(bbox)
            class_labels.append(ann['category_id'])
        
        labels.append({
            'bboxes': np.array(bboxes, dtype=np.float32) if bboxes else np.empty((0, 4), dtype=np.float32),
            'labels': np.array(class_labels, dtype=np.int32),
            'seg_map': image['file_name']  # assuming the image file name is stored here
        })
    return labels


def coco_to_mmdet(coco_annotations, num_classes):
    mmdet_predictions = {}

    # Map image IDs to file names
    image_id_to_filename = {image_info['id']: "images/" + image_info['file_name'] for image_info in coco_annotations['images']}

    # Initialize predictions dictionary with empty arrays for each class
    for image_info in coco_annotations['images']:
        filename = "images/" + image_info['file_name']
        mmdet_predictions[filename] = [np.empty((0, 5), dtype=np.float32) for _ in range(num_classes)]

    # Populate the predictions with bounding boxes from COCO annotations
    for annotation in coco_annotations['annotations']:
        image_id = annotation['image_id']
        filename = image_id_to_filename[image_id]
        category_id = annotation['category_id']
        
        # Only process category IDs that are within the number of classes
        if category_id < num_classes:
            bbox = annotation['bbox']  # COCO bbox is [x_min, y_min, width, height]
            score = annotation.get('score', 1.0)  # Default score to 1.0 if not provided
            
            # Convert COCO bbox to [x_min, y_min, x_max, y_max, score]
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height
            mmdet_bbox = np.array([x_min, y_min, x_max, y_max, score], dtype=np.float32)

            # Append the bounding box to the corresponding class array
            mmdet_predictions[filename][category_id] = np.vstack([mmdet_predictions[filename][category_id], mmdet_bbox])

    return mmdet_predictions


def compute_mistakes_score(
    gt_coco_path: str,
    preds_coco_path: str,
    classes_mapping: dict[int, str],
    data_dir: str = None
):
    # Load COCO annotations
    with open(gt_coco_path, 'r') as f:
        coco_data = json.load(f)

    with open(preds_coco_path, 'r') as f:
        pred_data = json.load(f)

    labels = coco_to_cleanlab(coco_data)
    predictions = list(coco_to_mmdet(pred_data, num_classes=2).values())
    label_issue_idx = find_label_issues(labels, predictions, return_indices_ranked_by_score=True)

    scores = get_label_quality_scores(labels, predictions)

    print(f" Number of images with issue{len(label_issue_idx)}")
    print(f"Average quality score {np.mean(scores)}")

    if data_dir:
        for idx in label_issue_idx[:5]:
            label = labels[idx]
            prediction = predictions[idx]
            score = scores[idx]
            image_path = data_dir + label['seg_map']  # Path to the image file

            print(f"Image: {image_path} | Label quality score: {score} | Is issue: True")
            visualize(image_path, label=label, prediction=prediction, class_names=classes_mapping, overlay=False)


if __name__ == "__main__":
    coco_annotations_path = '/Users/alexuvarovskiy/Documents/mlp_dataset/data/result.json'
    preds_path = '/Users/alexuvarovskiy/Documents/mlp/annotation/predictions.json'
    classes_mapping = {0: "person", 1: "car"}

    compute_mistakes_score(
        coco_annotations_path, 
        preds_path, 
        classes_mapping, 
        data_dir='/Users/alexuvarovskiy/Documents/mlp_dataset/data/images/'
    )
