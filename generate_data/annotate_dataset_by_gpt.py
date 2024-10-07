import ast
import base64
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
import argparse

MODEL="gpt-4o"


def parse_args():
    parser = argparse.ArgumentParser(description='Create a new Label Studio project and upload data from a local directory')
    parser.add_argument('--api_key', type=str, required=True, help='Label Studio API key')
    parser.add_argument('--images_dir', type=str, required=True, help='Local directory with images to upload')
    parser.add_argument('--labels_dir', type=str, required=True, help='Local directory to save labels')
    return parser.parse_args()

# Open the image file and encode it as a base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def write_list_of_lists_to_txt(data, filepath):
    # writes numbers from lists without any other symbols
    with open(filepath, 'w') as f:
        for item in data:
            for i in item:
                f.write(str(i) + ' ')
            f.write('\n')


def label_dataset(api_key, images_dir, labels_dir):
    client = OpenAI(api_key=api_key)
    images_dir = Path(images_dir)
    labels_path = Path(labels_dir)
    labels_path.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(images_dir.iterdir()):
        base64_image = encode_image(image_path)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an image annotation tool that is trying to find  objects on the image and get a responce . You should solve object detection annotaition problems!"},
                {"role": "user", "content": [
                    {"type": "text", "text": "What is bounding box coordinates for the person and car in the image? Where persin is index 0, and car is index 1. Provide in answer only a list of lists with relative coordinates in format class, x, y, w, h. If there are no objeces on the image just return an empty list"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"}
                    }
                ]}
            ],
        )

        bboxes = ast.literal_eval(response.choices[0].message.content)
        dst_path = labels_path / f"{image_path.stem}.txt"
        write_list_of_lists_to_txt(bboxes, dst_path)



if __name__ == "__main__":
    args = parse_args()

    label_dataset(
        api_key=args.api_key,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir
    )
