import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
import json
import argparse
from transformers import AutoModelForObjectDetection, AutoImageProcessor

device = torch.device("cpu")

def load_model_and_processor(model_path):
    model = AutoModelForObjectDetection.from_pretrained(model_path)
    processor = AutoImageProcessor.from_pretrained(model_path)
    
    model.to(device)
    model.eval()
    
    return model, processor

CLASS_COLOR_MAPPING = {
    "person": "red",
    "car": "blue",
    "pet": "green"
}

MODEL_LABEL_MAPPING = {0: "person", 1: "car", 2: "pet"}

def predict(image: Image.Image, threshold: float, model, processor):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])  # target size in (height, width)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
    results = {k: v.detach().cpu() for k, v in results.items()}
    return results


def draw_boxes_pillow(image: Image.Image, results):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=25)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        xmin, ymin, xmax, ymax = box
        class_label = MODEL_LABEL_MAPPING[label.item()]
        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=CLASS_COLOR_MAPPING[class_label], width=3)
        text = f'{class_label}: {score.item():.2f}'
        text_bbox = draw.textbbox((xmin, ymin), text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        text_position = (xmin, ymin - text_height)
        draw.rectangle([text_position, (xmin + text_width, ymin)], fill=CLASS_COLOR_MAPPING[class_label])
        draw.text((xmin, ymin - text_height), text, fill="white", font=font)
    return image


def main(model_path):
    model, processor = load_model_and_processor(model_path)

    threshold = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.5, 0.05)

    st.title('Object Detection Inference')

    uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)

        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Start Inference'):
            st.write("Running inference...")
            results = predict(image, threshold, model, processor)

            st.write("Inference complete! Displaying image with bounding boxes.")
            image_with_boxes = image.copy()
            image_with_boxes = draw_boxes_pillow(image_with_boxes, results)

            st.image(image_with_boxes, caption='Detected Objects', use_column_width=True)

            output_data = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                output_data.append({
                    "class": MODEL_LABEL_MAPPING[label.item()],
                    "score": score.item(),
                    "box": [box[0].item(), box[1].item(), box[2].item(), box[3].item()]
                })

            st.write("Predictions in JSON format:")
            st.json(output_data)

            st.write("Copy-pasteable JSON:")
            st.code(json.dumps(output_data, indent=2), language='json')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Object Detection Streamlit app.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the pre-trained model directory.")
    
    args = parser.parse_args()
    main(args.model_path)
