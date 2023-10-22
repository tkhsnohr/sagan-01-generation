from glob import glob
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def get_image_paths(dir):
    image_paths = glob(f"{dir}/**/*.[jp][pn]g", recursive=True)
    
    return image_paths

class MNISTDataset(Dataset):
    def __init__(self, image_paths: list[str], image_size: int = 64) -> None:
        self.image_paths = image_paths
        self.image_size = image_size
        
        load_size = int(1.1 * self.image_size)
        osize = [load_size, load_size]
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])


    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = self.transform(image)

        return image