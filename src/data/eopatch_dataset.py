import os
from typing import Optional, List

import numpy as np
import torch
from eolearn.core import EOPatch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import albumentations as albu

from src.utils.misc import get_eopatches_split_dir, denormalize_imagenet
from src.data.get_data import BAND_NAMES


class EOPatchDataset(Dataset):
    def __init__(
            self,
            eopatches_dir: str,
            split: Optional[str],
            band_names_to_take: List[str],
            to_take_ndvi: bool,
            scale_rgb_intensity: Optional[float] = None,
            scale_density: Optional[float] = None,
            mask_data: Optional[float] = None,
            transform: Optional[albu.Compose] = None,
            normalization_transform: Optional[albu.ImageOnlyTransform] = None
    ):
        self.eopatches_dir = eopatches_dir
        self.band_names_to_take = band_names_to_take
        self.to_take_ndvi = to_take_ndvi
        self.scale_rgb_intensity = scale_rgb_intensity
        self.scale_density = scale_density
        self.mask_data = mask_data

        self.transform = transform
        self.normalization_transform = normalization_transform

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
        self.band_idxs_to_take = [BAND_NAMES.index(band_name) for band_name in self.band_names_to_take]

    def __getitem__(self, idx: int):
        eopatch_name = self.eopatch_names[idx]
        path_to_eopatch = os.path.join(self.eopatches_dir, eopatch_name)

        eopatch = EOPatch.load(path_to_eopatch, lazy_loading=True)

        data = eopatch.data['L2A_BANDS'][0][:, :, self.band_idxs_to_take]  # [H, W, Nb]
        if self.scale_rgb_intensity:
            data[:, :, :3] *= self.scale_rgb_intensity

        data = np.clip(data, 0., 1.)
        if self.to_take_ndvi:
            ndvi = eopatch.data['NDVI'][0]
            data = np.concatenate([data, ndvi], axis=-1)  # [H, W, Nb+1]

        density_map = eopatch.data_timeless['TREES_DENSITY']  # [H, W, 1]
        if self.scale_density:
            density_map *= self.scale_density

        street_mask = eopatch.mask_timeless['STREET_MASK']  # [H, W, 1]
        del eopatch

        if self.transform is not None:
            transformed = self.transform(image=data, masks=[density_map, street_mask])
            data = transformed['image']
            density_map = transformed['masks'][0]
            street_mask = transformed['masks'][1]

        if self.mask_data:
            data *= street_mask

        if self.normalization_transform is not None:
            transformed = self.normalization_transform(image=data)
            data = transformed['image']

        tree_count = density_map.sum()

        data = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1)
        density_map = torch.tensor(density_map, dtype=torch.float32).permute(2, 0, 1)
        tree_count = torch.tensor(tree_count, dtype=torch.float32)
        street_mask = torch.tensor(street_mask, dtype=torch.bool).permute(2, 0, 1)

        sample = dict()
        sample['data'] = data
        sample['density_map'] = density_map
        sample['tree_count'] = tree_count
        sample['street_mask'] = street_mask
        return sample

    def __len__(self):
        return len(self.eopatch_names)


if __name__ == '__main__':
    from omegaconf import OmegaConf
    from dotenv import load_dotenv
    import hydra

    load_dotenv()
    cfg = OmegaConf.load('../configs/experiment/run_5_yama.yaml')
    d = hydra.utils.instantiate(cfg['train_dataset'])

    # for i in [0, 0, 0, 0, 0, 0, 0]:
    for i in range(10):
        sample = d[i]
        data = sample['data']
        # print(data[3].min(), data[3].mean(), data[3].max())
        print(data[0].min(), data[0].mean(), data[0].max())
        # continue
        plt.subplot(1, 2, 1)
        plt.imshow(denormalize_imagenet(sample['data'][:3]).permute(1, 2, 0) * 2.5)
        # plt.imshow(sample['data'][:3].permute(1, 2, 0) * 2.5)
        plt.subplot(1, 2, 2)
        plt.imshow(np.clip(sample['density_map'][0, :, :], 0, 1))
        plt.show()
