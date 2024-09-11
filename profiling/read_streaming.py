from torch.utils.data import DataLoader
from streaming import StreamingDataset
from torchvision.transforms import Compose, Resize, ToTensor
import torch

import argparse


transforms = Compose([
        Resize((256, 256)),
        ToTensor()
    ])


def collate_fn(batch):
    return {
        'image': torch.stack([transforms(sample['image']) for sample in batch]),
        'label': [sample['label'] for sample in batch]
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Streaming data reading profiling')
    parser.add_argument('--local', type=str, default='/tmp/streaming_read')
    parser.add_argument('--remote', type=str, default='s3://mlp-data-2024/streaming')
    parser.add_argument('--batch_size', type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset = StreamingDataset(local=args.local, remote=args.remote, shuffle=True, batch_size=args.batch_size)
    sample = dataset[3]
    img = sample['image']
    ann = sample['label']

    dataloader = DataLoader(dataset, collate_fn=collate_fn)
    for i, sample in enumerate(dataloader):
        print(i, sample['image'].shape, sample['label'])
        if i == 10:
            break
