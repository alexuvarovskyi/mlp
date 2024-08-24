from torch.utils.data import DataLoader
from streaming import StreamingDataset
from torchvision.transforms import Compose, Resize, ToTensor
import torch


transforms = Compose([
        Resize((256, 256)),
        ToTensor()
    ])


def collate_fn(batch):
    return {
        'image': torch.stack([transforms(sample['image']) for sample in batch]),
        'label': [sample['label'] for sample in batch]
    }




if __name__ == "__main__":
    remote = 's3://mlp-data-2024/streaming'
    local = '/Users/alexuvarovskiy/Documents/mlp/profiling/streaming_read____'

    dataset = StreamingDataset(local=local, remote=remote, shuffle=True, batch_size=4)
    sample = dataset[3]
    img = sample['image']
    ann = sample['label']

    dataloader = DataLoader(dataset, collate_fn=collate_fn)
    for i, sample in enumerate(dataloader):
        print(i, sample['image'].shape, sample['label'])
        if i == 10:
            break
    
    ### Process the dataset