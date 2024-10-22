import requests
import base64
import json

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    return parser.parse_args()


def make_request(image_path):

    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()

    encoded_image = base64.b64encode(image_bytes).decode('utf-8')

    request_data = {
        "data": {
            "ndarray": [{"image_bytes": encoded_image}]
        },
        "parameters": {
            "threshold": 0.5
        }
    }

    response = requests.post("http://localhost:5001/api/v1.0/predictions", json=request_data)

    return response.json()

if __name__ == "__main__":
    args = parse_args()
    response = make_request(args.image_path)
    print(response)
