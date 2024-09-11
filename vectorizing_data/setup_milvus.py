import json
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import argparse

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from pymilvus import connections, Collection, db, MilvusClient

from transformers import CLIPProcessor, CLIPModel
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection, connections, db


MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

DB_NAME = "data_embeddings"
COLLECTION_NAME = "data_clip_embeddings"


def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def create_db():
    conn = connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    database = db.create_database(DB_NAME)


def create_video_collection():
    conn = connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, db_name=DB_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=255),
    ]

    schema = CollectionSchema(fields=fields, description="CLIP video embeddings collection")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    index_params = {
        'metric_type': 'COSINE',
        'index_type': 'IVF_FLAT',
        'index_name': 'embeddings_index',
        'params':{'nlist': 16384},
    }

    collection.create_index(field_name='embedding', index_params=index_params)
    return collection


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


def fill_db(images_path: str):
    client = MilvusClient(
        uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}",
        db_name=DB_NAME,
    )

    transform = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
    ])

    images_dir = Path(images_path)
    image_paths = [p for p in images_dir.rglob("**/*.*") if p.is_file()]
    image_dataset = ImageDataset(image_paths, transform=transform)
    image_dataloader = DataLoader(image_dataset, batch_size=2, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = CLIP().to(device)
    model.eval()

    with torch.no_grad():
        for sample in tqdm(image_dataloader):
            X, paths = sample
            embeddings = model(X).cpu().tolist()

            insert_batch = []
            for embedding, path in zip(embeddings, paths):
                insert_batch.append(
                    {
                        'embedding': embedding,
                        'path': path,
                    }
                )
            client.insert(collection_name=COLLECTION_NAME, data=insert_batch)
    client.flush(COLLECTION_NAME)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, required=True, help='Path to images directory')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    create_db()
    create_video_collection()
    fill_db(images_path=args.images_path)
