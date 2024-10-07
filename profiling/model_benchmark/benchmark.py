import os
import ray
import time
import torch
import argparse
import multiprocessing

from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).cpu()


def infer_batch(model, batch):
    return model(batch)


def infer_multiprocessing(batch):
    return infer_batch(MODEL, batch)


def single_process_inference(model, dataloader):
    results = []
    for batch in dataloader:
        results.append(infer_batch(model, batch))
    return results


def multiprocessing_inference(dataloader):
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(executor.map(infer_multiprocessing, dataloader))
    return results


def threading_inference(dataloader):
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(executor.map(lambda batch: infer_batch(MODEL, batch), dataloader))
    return results


@ray.remote
class ModelActor:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).cpu()

    def infer(self, batch):
        return infer_batch(self.model, batch)

def ray_inference(dataloader):
    ray.init(ignore_reinit_error=True)
    model_actor = ModelActor.remote()
    futures = [model_actor.infer.remote(batch) for batch in dataloader]
    results = ray.get(futures)
    ray.shutdown()
    return results

# Benchmark function
def benchmark_inference(method_name, inference_function, *args):
    start_time = time.time()
    inference_function(*args)
    end_time = time.time()
    print(f"{method_name} took {end_time - start_time:.2f} seconds")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking different methods for inference")
    parser.add_argument("--image_directory", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args() 


if __name__ == "__main__":
    args = parse_args()

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])

    dataset = ImageFolderDataset(args.image_directory, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


    benchmark_inference("Single Process Inference", single_process_inference, MODEL, dataloader)
    benchmark_inference("Multiprocessing Inference", multiprocessing_inference, dataloader)
    benchmark_inference("Threading Inference", threading_inference, dataloader)
    benchmark_inference("RAY Inference", ray_inference, dataloader)
