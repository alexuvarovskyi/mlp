import requests
import argparse
from pathlib import Path
from openai import OpenAI


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--save_data_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    return parser.parse_args()


def save_image(url, path):
    response = requests.get(url)
    with open (path, 'wb') as file:
        file.write(response.content)


def generate_images(api_key, save_data_dir, prompt):
    client = OpenAI(api_key=api_key)
    save_data_dir = Path(save_data_dir)
    save_data_dir.mkdir(exist_ok=True)
    for i in range(10):
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
            style="natural",
        )
        save_image(response.data[0].url, save_data_dir / f"image_{i}.jpg")
        print(f"Image {i} saved")


if __name__ == "__main__":
    args = parse_args()

    generate_images(
        api_key=args.api_key,
        save_data_dir=args.save_data_dir,
        prompt=args.prompt,
    )
