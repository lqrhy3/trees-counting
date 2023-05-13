import albumentations as albu
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from src.data.eopatch_dataset import EOPatchDataset


class EOPatchDataModule(pl.LightningDataModule):
    def __init__(
            self,
            eopatches_dir: str,
            train_transform: albu.Compose,
            val_transform: albu.Compose,
            batch_size: int,
            num_workers: int,
            pin_memory: bool
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.eopatches_dir = eopatches_dir
        self.train_transform = train_transform
        self.val_transform = val_transform

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str) -> None:
        if self.train_dataset is None and self.val_dataset is None:
            train_dataset = EOPatchDataset(self.eopatches_dir, split='train', transform=self.train_transform)
            self.train_dataset = train_dataset

            val_dataset = EOPatchDataset(self.eopatches_dir, split='val', transform=self.val_transform)
            self.val_dataset = val_dataset

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True
        )

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True
        )

if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    load_dotenv()

    train_transform = None# albu.Compose([
        # albu.Normalize(max_pixel_value=1.),
    # ])
    val_transform = None # albu.Compose([
        # albu.Normalize(max_pixel_value=1.),
    # ])

    datamodule = EOPatchDataModule(
        eopatches_dir=os.environ['EOPATCHES_DIR'],
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=1,
        num_workers=0,
        pin_memory=False
    )
    datamodule.setup('fit')
    for i in range(len(datamodule.train_dataset)):
        sample = datamodule.train_dataset[i]
        data = sample['data']
        density_map = sample['density_map']

        print(data.min(), data.max())
        print(density_map.min(), density_map.max(), density_map.sum())

        plt.subplot(1, 2, 1)
        plt.imshow(data.permute(1, 2, 0) * 3.5)
        plt.subplot(1, 2, 2)
        plt.imshow(density_map.permute(1, 2, 0))
        plt.show()

