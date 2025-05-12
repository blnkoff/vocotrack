import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from lightning.pytorch import LightningDataModule
from dataset import Dataset

NUM_WORKERS = os.cpu_count() // 2


class DataModule(LightningDataModule):
    __slots__ = (
        '_generator',
        '_data_dir',
        '_num_workers',
        '_batch_size',
        'train_dataset',
        'val_dataset',
        'test_dataset'
    )
    
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 32,
        num_workers: int = NUM_WORKERS,
        device: str = 'cpu',
        seed: int = 42
    ):
        super().__init__()
        generator = torch.Generator(device).manual_seed(seed)
        self._generator = generator
        
        self._data_dir = data_dir
        self._num_workers = num_workers
        self._batch_size = batch_size
        
        self.train_dataset: DataLoader | None = None
        self.val_dataset: DataLoader | None = None
        self.test_dataset: DataLoader | None = None
        
    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            data = torch.load(self._data_dir / 'train.pt')
            
            train_dataset, val_dataset = random_split(Dataset(data), lengths=(0.8, 0.2), generator=self._generator)
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
        elif stage == 'test':
            data = torch.load(self._data_dir / 'test.pt')
            test_dataset = Dataset(data)
            self.test_dataset = test_dataset
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers
        )
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers
        )
    