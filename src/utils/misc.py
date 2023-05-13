import os
import random
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import torch
from matplotlib import pyplot as plt
from torch.optim import Optimizer


IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]


def read_df_with_coordinates(
        path_to_csv: str,
        x_name: str = 'longitude',
        y_name: str = 'latitude',
        crs: pyproj.CRS = pyproj.CRS('GCS_WGS_1984')
):
    df = pd.read_csv(path_to_csv)
    geometry = gpd.points_from_xy(df[x_name], df[y_name], crs=crs)
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    return gdf


def get_eopatches_split_dir(eopatches_dir: str):
    eopatches_dir_base = os.path.join(*os.path.split(eopatches_dir)[:-1])
    eopatches_dir_name = os.path.split(eopatches_dir)[-1]
    eopatches_split_dir = os.path.join(eopatches_dir_base, f'{eopatches_dir_name}_split')
    return eopatches_split_dir


def denormalize_imagenet(image: torch.Tensor):
    mean = torch.tensor(IMAGENET_MEAN, device=image.device)  # [3,]
    std = torch.tensor(IMAGENET_STD, device=image.device)  # [3,]

    if image.ndim == 4:
        assert image.shape[1] == 3
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    elif image.ndim == 3:
        assert image.shape[0] == 3
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)

    image = image * std + mean
    return image


def save_checkpoint(
        model: torch.nn.Module,
        epoch: int,
        artefacts_dir: str,
        optimizer: Optional[Optimizer] = None,
        scheduler=None,
        save_only_one: bool = False
):
    state_dict = model.state_dict()
    save_dict = {'epoch': epoch, "state_dict": state_dict}
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()

    checkpoints_dir = os.path.join(artefacts_dir, 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    checkpoint_pth = os.path.join(checkpoints_dir, f'epoch_{epoch:04d}.pt')

    other_checkpoint_names = []
    if save_only_one:
        other_checkpoint_names = os.listdir(checkpoints_dir)

    torch.save(save_dict, checkpoint_pth)
    if other_checkpoint_names:
        for other_checkpoint_name in other_checkpoint_names:
            os.remove(os.path.join(checkpoints_dir, other_checkpoint_name))


def load_checkpoint(model, path_to_checkpoint, device, optimizer=None):
    state_dict = torch.load(path_to_checkpoint, map_location=device)
    model.load_state_dict(state_dict['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state_dict['optimizer'])


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


def get_class(obj):
    return obj.__class__


def plot_and_save_bbox_splits(boundaries_gdf, bbox_list, bbox_gdf):
    fig, ax = plt.subplots(figsize=(60, 60))
    boundaries_gdf.plot(ax=ax, facecolor="w", edgecolor="b", alpha=0.8)
    bbox_gdf.plot(ax=ax, facecolor="w", edgecolor="r", alpha=0.5)
    for idx, bbox in enumerate(bbox_list):
        geo = bbox.geometry
        ax.text(geo.centroid.x, geo.centroid.y, idx, ha="center", va="center", fontsize=6)

    plt.show()
