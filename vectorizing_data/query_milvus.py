import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import argparse

from PIL import Image
from torch.utils.data import Dataset
from pymilvus import connections, Collection, MilvusClient
from transformers import CLIPProcessor, CLIPModel


MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

DB_NAME = "data_embeddings"
COLLECTION_NAME = "data_clip_embeddings"

TOP_K = 8


def read_json(file_path: str):
    with open(file_path, 'r') as f:
        return json.load(f)


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, str(image_path)


class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
        model_name = "openai/clip-vit-large-patch14-336"
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def forward(self, x) -> torch.Tensor:
        inputs = self.processor(images=x, return_tensors="pt", do_resize=False, convert_to_rgb=False, do_center_crop=False, do_normalize=True, do_rescale=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return self.model.get_image_features(**inputs)

    def to(self, device):
        self.model.to(device)
        return self


def load_image_embeddings(img_path: str):
    transform = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)
    clip = CLIP()
    clip.eval()
    with torch.no_grad():
        img_embedding = clip(img).cpu().numpy()
    return img_embedding


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path")
    return parser.parse_args()


if __name__ == "__main__":    
    args = parse_args()

    connection = connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, db_name=DB_NAME)
    client = MilvusClient(host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(COLLECTION_NAME)
    collection.load()

    embeds = load_image_embeddings(args.img_path)

    res = collection.search(
        data=embeds,
        anns_field='embedding',
        limit=TOP_K, 
        output_fields=['path'],
        param={
            'metric_type': 'COSINE', 
            'nprobe': 512
        },
    )

    for hist_i, hist in enumerate(res):
        for hit_i, hit in enumerate(hist):
            print(hit.entity.get('path'))

