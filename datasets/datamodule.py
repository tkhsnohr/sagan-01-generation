from torch.utils.data import DataLoader
import lightning as L

from datasets.mnist_dataset import get_image_paths, MNISTDataset

class MNISTDataModule(L.LightningDataModule):
    def __init__(self, dir: str, batch_size: int = 32, num_workers: int = 3):
        super().__init__()
        self.dir = dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None) -> None:
        image_paths = get_image_paths(self.dir)
        self.train_dataset = MNISTDataset(image_paths)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )