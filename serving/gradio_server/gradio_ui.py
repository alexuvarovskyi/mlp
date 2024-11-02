import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import torch
import json
from transformers import AutoModelForObjectDetection, AutoImageProcessor

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_model_and_processor(model_path):
    # Load model and processor
    model = AutoModelForObjectDetection.from_pretrained(model_path)
    processor = AutoImageProcessor.from_pretrained(model_path)
    
    model.to(device)
    model.eval()
    
    return model, processor

# Define color mapping for classes
CLASS_COLOR_MAPPING = {
    "person": "red",
    "car": "blue",
    "pet": "green"
}

# Define the model's label mapping (adjust as per your model)
MODEL_LABEL_MAPPING = {0: "person", 1: "car", 2: "pet"}

def predict(image: Image.Image, threshold: float, model, processor):
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform inference
    outputs = model(**inputs)

    # Convert outputs to numpy array
    target_sizes = torch.tensor([image.size[::-1]])  # target size in (height, width)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
    results = {k: v.detach().cpu() for k, v in results.items()}
    
    return results

def draw_boxes_pillow(image: Image.Image, results):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=25)

    # Add bounding boxes
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        # Un-normalize the bounding boxes
        xmin, ymin, xmax, ymax = box
        class_label = MODEL_LABEL_MAPPING[label.item()]

        # Draw rectangle
        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=CLASS_COLOR_MAPPING[class_label], width=3)

        # Add class label and score
        text = f'{class_label}: {score.item():.2f}'

        text_bbox = draw.textbbox((xmin, ymin), text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        text_position = (xmin, ymin - text_height)

        # Draw text background and text
        draw.rectangle([text_position, (xmin + text_width, ymin)], fill=CLASS_COLOR_MAPPING[class_label])
        draw.text((xmin, ymin - text_height), text, fill="white", font=font)

    return image

def gradio_interface(model_path):
    model, processor = load_model_and_processor(model_path)

    def inference(image, threshold):
        results = predict(image, threshold, model, processor)
        image_with_boxes = draw_boxes_pillow(image.copy(), results)
        
        # Prepare JSON output for predictions
        output_data = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            output_data.append({
                "class": MODEL_LABEL_MAPPING[label.item()],
                "score": score.item(),
                "box": [box[0].item(), box[1].item(), box[2].item(), box[3].item()]
            })
        
        return image_with_boxes, output_data

    # Create Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# Object Detection Inference")
        
        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload an image")
            threshold_input = gr.Slider(minimum=0, maximum=1, step=0.05, value=0.5, label="Confidence Threshold")
        
        submit_button = gr.Button("Start Inference")
        image_output = gr.Image(label="Detected Objects")
        json_output = gr.JSON(label="Predictions in JSON format")
        
        submit_button.click(inference, inputs=[image_input, threshold_input], outputs=[image_output, json_output])

    demo.launch()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the Object Detection Gradio app.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the pre-trained model directory.")

    args = parser.parse_args()
    gradio_interface(args.model_path)
