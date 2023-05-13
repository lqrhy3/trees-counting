import os
from typing import Optional

import numpy as np
import torch
from eolearn.core import EOPatch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import albumentations as albu

from src.utils.misc import get_eopatches_split_dir


class EOPatchDataset(Dataset):
    def __init__(
            self,
            eopatches_dir: str,
            split: Optional[str],
            transform: albu.Compose
    ):
        self.eopatches_dir = eopatches_dir
        self.transform = transform

        if split:
            eopatches_split_dir = get_eopatches_split_dir(eopatches_dir)
            with open(os.path.join(eopatches_split_dir, f'{split}.txt'), 'r') as split_file:
                eopatch_names = split_file.readlines()
                eopatch_names = [name.strip() for name in eopatch_names]

        else:
            eopatch_names = [
                x for x in os.listdir(eopatches_dir)
                if os.path.isdir(os.path.join(self.eopatches_dir, x))
            ]

        self.eopatch_names = eopatch_names

    def __getitem__(self, idx: int):
        eopatch_name = self.eopatch_names[idx]
        path_to_eopatch = os.path.join(self.eopatches_dir, eopatch_name)

        eopatch = EOPatch.load(path_to_eopatch, lazy_loading=True)

        data = eopatch.data['L2A_BANDS'][0][:, :, [3, 2, 1]]  # [H, W, 3]
        data = np.clip(data, 0., 1.)
        density_map = eopatch.data_timeless['TREES_DENSITY']  # [H, W, 1]
        del eopatch

        if self.transform is not None:
            transformed = self.transform(image=data, mask=density_map)
            data = transformed['image']
            density_map = transformed['mask']

        data = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1)
        density_map = torch.tensor(density_map, dtype=torch.float32).permute(2, 0, 1)

        sample = dict()
        sample['data'] = data
        sample['density_map'] = density_map
        return sample

    def __len__(self):
        return len(self.eopatch_names)


if __name__ == '__main__':
    d = EOPatchDataset('/home/lqrhy3/PycharmProjects/trees-counting/data/raw/eopatches')
    for i in range(3):
        sample = d[i]
        plt.subplot(1, 2, 1)
        plt.imshow(sample['data'])
        plt.subplot(1, 2, 2)
        plt.imshow(sample['density_map'])
        plt.show()
