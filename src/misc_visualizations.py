import os
import random
import sys
from typing import Optional, List

import cv2
import numpy as np
import pyrootutils
import rasterio
from eolearn.core import EOPatch
from matplotlib import pyplot as plt

import geopandas as gpd
from sentinelhub import UtmZoneSplitter
from sentinelsat import read_geojson
from shapely import Polygon
from shapely.geometry import shape
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from src.utils.misc import get_eopatches_split_dir
from rasterio.features import rasterize

pyrootutils.setup_root(__file__, indicator='.project-root', project_root_env_var=True, dotenv=True, pythonpath=True)


BBOX_SIZE = 128
RESOLUTION = 10

plt.rcParams['figure.dpi'] = 140


def get_rgb(eopatch, rgb=None, scale=2.5):
    if rgb is None:
        rgb = [3, 2, 1]
    rgb = np.clip(eopatch.data['L2A_BANDS'][0][:, :, rgb] * scale, 0, 1)
    return rgb


def get_custom_street_mask(eopatch, kernel_size):
    street_geometry = eopatch.mask_timeless['STREET_GEOMETRY']  # [H, W, 1]
    street_geometry = (street_geometry * 255).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    street_mask = cv2.dilate(street_geometry, kernel, iterations=1)
    street_mask = (street_mask / 255).astype(int)
    return street_mask[:, :, None]


def plot_and_save_bbox_splits():
    boundaries_gdf = gpd.read_file(os.environ['PATH_TO_NY_BOROUGH_BOUNDARIES'])
    boundaries_geojson = read_geojson(os.environ['PATH_TO_NY_BOROUGH_BOUNDARIES'])
    boundary_geometries = [shape(boundaries_geojson['features'][i]['geometry']) for i in range(len(boundaries_gdf))]
    bbox_splitter = UtmZoneSplitter(
        boundary_geometries, crs=boundaries_gdf.crs, bbox_size=BBOX_SIZE * RESOLUTION
    )
    bbox_list = np.array([bbox.transform(boundaries_gdf.crs) for bbox in bbox_splitter.get_bbox_list()])
    geometry = [Polygon(bbox.get_polygon()) for bbox in bbox_list]
    bbox_gdf = gpd.GeoDataFrame(crs=boundaries_gdf.crs,geometry=geometry)

    eopatches_split_dir = get_eopatches_split_dir(os.environ['EOPATCHES_DIR'])
    with open(os.path.join(eopatches_split_dir, f'val.txt'), 'r') as split_file:
        eopatch_names = split_file.readlines()
        eopatch_names = [name.strip() for name in eopatch_names]
        val_idxs_1 = [int(name.split('_')[1]) for name in eopatch_names]

    val_idxs_2 = [idx + random.randint(1, 10) for idx in val_idxs_1]
    print(val_idxs_1)
    print(val_idxs_2)
    exit(0)


    boundaries_gdf['dummy'] = 'Границы Нью-Йорка'

    fig, ax = plt.subplots(figsize=(24, 24))
    # boundaries_gdf.plot(ax=ax, column='dummy', categorical=True, legend=True, facecolor='none', edgecolor='b', alpha=0.7)
    boundaries_gdf.plot(ax=ax, facecolor='none', edgecolor='b', alpha=0.8, linewidth=1.3)


    legend_elements = [Line2D([0], [0], color='b', lw=4, label='Граница Нью-Йорка', alpha=0.8, linewidth=1.3),
                       Patch(facecolor='none', edgecolor='r', label='Тренировочная часть', linewidth=1.3),
                       Patch(facecolor='g', edgecolor='r', label='Валидационная часть №1', alpha=0.5, linewidth=1.3),
                       Patch(facecolor='y', edgecolor='r', label='Валидационная часть №2', alpha=0.5, linewidth=1.3)]

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.05, 0.9), fontsize=24)

    bbox_gdf.plot(ax=ax, facecolor="w", edgecolor="r", alpha=0.5, linewidth=1)
    for idx, bbox in enumerate(bbox_list):
        geo = bbox.geometry
        ax.text(geo.centroid.x, geo.centroid.y, idx, ha="center", va="center", fontsize=15)
    bbox_gdf[bbox_gdf.index.isin(val_idxs_1)].plot(ax=ax, facecolor='g', edgecolor='r', alpha=0.3)
    bbox_gdf[bbox_gdf.index.isin(val_idxs_2)].plot(ax=ax, facecolor='y', edgecolor='r', alpha=0.3)
    # plt.title('Разбиение области на тренировочную и валидационные части.', fontsize=)
    plt.axis('off')
    plt.savefig(os.path.join(os.environ['VISUALIZATIONS_DIR'], 'train_val_split.png'), bbox_inches='tight', dpi=80)
    # plt.show()


def plot_eopatch_rgb_nir_ndvi(eopatch_names: List[str], title: Optional[str] = None):
    eopatch_1 = EOPatch.load(os.path.join(os.environ['EOPATCHES_DIR'], eopatch_names[0]), lazy_loading=True)
    rgb_1 = np.clip(eopatch_1.data['L2A_BANDS'][0][:, :, [3, 2, 1]] * 2.5, 0, 1)
    nir_1 = eopatch_1.data['L2A_BANDS'][0][:, :, [7, 3, 2]]
    nir_1[:, :, 0] *= 1.5
    nir_1[:, :, 1:] *= 2.5
    nir_1 = np.clip(nir_1, 0, 1)

    ndvi_1 = eopatch_1.data['NDVI'][0]

    eopatch_2 = EOPatch.load(os.path.join(os.environ['EOPATCHES_DIR'], eopatch_names[1]), lazy_loading=True)
    rgb_2 = np.clip(eopatch_2.data['L2A_BANDS'][0][:, :, [3, 2, 1]] * 2.5, 0, 1)
    nir_2 = eopatch_2.data['L2A_BANDS'][0][:, :, [7, 3, 2]]
    nir_2[:, :, 0] *= 1.5
    nir_2[:, :, 1:] *= 2.5
    nir_2 = np.clip(nir_2, 0, 1)

    ndvi_2 = eopatch_2.data['NDVI'][0]

    plt.subplot(2, 3, 1)
    plt.imshow(rgb_1)
    plt.axis('off')
    plt.gca().set_title('R, G, B')

    plt.subplot(2, 3, 2)
    plt.imshow(nir_1)
    plt.axis('off')
    plt.gca().set_title('NIR, R, G')

    plt.subplot(2, 3, 3)
    plt.imshow(ndvi_1, cmap='YlGn', vmin=-0.3, vmax=0.9)
    plt.axis('off')
    plt.gca().set_title('NDVI')

    plt.subplot(2, 3, 4)
    plt.imshow(rgb_2)
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(nir_2)
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(ndvi_2, cmap='YlGn', vmin=-0.3, vmax=0.9)
    plt.axis('off')

    plt.savefig(
        os.path.join(os.environ['VISUALIZATIONS_DIR'], 'rgb_nir_ndvi.png'),
        bbox_inches='tight',
        dpi=100
    )
    plt.show()


def plot_tree_annotations_and_density(eopatch_names: List[str]):
    eopatch_1 = EOPatch.load(os.path.join(os.environ['EOPATCHES_DIR'], eopatch_names[0]), lazy_loading=True)
    annotations_1 = eopatch_1.mask_timeless['TREES_ANNOTATIONS']
    density_1 = eopatch_1.data_timeless['TREES_DENSITY']

    eopatch_2 = EOPatch.load(os.path.join(os.environ['EOPATCHES_DIR'], eopatch_names[1]), lazy_loading=True)
    annotations_2 = eopatch_2.mask_timeless['TREES_ANNOTATIONS']
    density_2 = eopatch_2.data_timeless['TREES_DENSITY']

    vmax = np.max(np.stack([annotations_1, annotations_2]))
    print(vmax)

    plt.subplot(2, 2, 1)
    plt.imshow(annotations_1, vmin=0, vmax=vmax)
    ax1 = plt.gca()
    ax1.set_title('Карта аннотаций', fontsize=10)
    ax1.set_axis_off()
    plt.subplot(2, 2, 2)
    plt.imshow(density_1, vmin=0, vmax=vmax)
    ax2 = plt.gca()
    ax2.set_title('Карта плотностей', fontsize=10)
    ax2.set_axis_off()

    plt.subplot(2, 2, 3)
    plt.imshow(annotations_2, vmin=0, vmax=vmax)
    ax3 = plt.gca()
    ax3.set_axis_off()
    plt.subplot(2, 2, 4)
    plt.imshow(density_2, vmin=0, vmax=vmax)
    ax4 = plt.gca()
    ax4.set_axis_off()

    plt.colorbar(ax=[ax1, ax2, ax3, ax4])

    plt.savefig(
        os.path.join(os.environ['VISUALIZATIONS_DIR'], 'tree_annotations_and_density.png'),
        bbox_inches='tight',
        dpi=100
    )
    plt.show()


def plot_street_annotations_mask_rgb(eopatch_names: List[str]):
    eopatch_1 = EOPatch.load(os.path.join(os.environ['EOPATCHES_DIR'], eopatch_names[0]), lazy_loading=True)
    rgb_1 = get_rgb(eopatch_1)
    annotations_1 = eopatch_1.mask_timeless['STREET_GEOMETRY']
    # mask_1 = eopatch_1.mask_timeless['STREET_MASK']
    mask_1 = get_custom_street_mask(eopatch_1, kernel_size=3)

    eopatch_2 = EOPatch.load(os.path.join(os.environ['EOPATCHES_DIR'], eopatch_names[1]), lazy_loading=True)
    rgb_2 = get_rgb(eopatch_2)
    annotations_2 = eopatch_2.mask_timeless['STREET_GEOMETRY']
    # mask_2 = eopatch_2.mask_timeless['STREET_MASK']
    mask_2 = get_custom_street_mask(eopatch_2, kernel_size=3)

    plt.subplot(2, 3, 1)
    plt.imshow(annotations_1)
    ax1 = plt.gca()
    ax1.set_title('Геометрия улиц', fontsize=10)
    ax1.set_axis_off()
    plt.subplot(2, 3, 2)
    plt.imshow(mask_1)
    ax2 = plt.gca()
    ax2.set_title('Дилатированная геометрия', fontsize=8)
    ax2.set_axis_off()

    plt.subplot(2, 3, 3)
    plt.imshow(mask_1 * rgb_1)
    ax2 = plt.gca()
    ax2.set_title('Маскированные данные', fontsize=8)
    ax2.set_axis_off()

    plt.subplot(2, 3, 4)
    plt.imshow(annotations_2)
    ax3 = plt.gca()
    ax3.set_axis_off()
    plt.subplot(2, 3, 5)
    plt.imshow(mask_2)
    ax4 = plt.gca()
    ax4.set_axis_off()

    plt.subplot(2, 3, 6)
    plt.imshow(mask_2 * rgb_2)
    ax2 = plt.gca()
    ax2.set_axis_off()

    plt.savefig(
        os.path.join(os.environ['VISUALIZATIONS_DIR'], 'street_geometry_mask_rgb.png'),
        bbox_inches='tight',
        dpi=300
    )
    plt.show()

def plot_whole_ny_eopatch():
    eopatch = EOPatch.load('/home/lqrhy3/PycharmProjects/trees-counting/data/raw/whole_ny_eopatch/eopatch')
    rgb = get_rgb(eopatch, rgb=[0, 1, 2])

    transform = rasterio.transform.from_bounds(
        eopatch.bbox.min_x, eopatch.bbox.min_y, eopatch.bbox.max_x, eopatch.bbox.max_y,
        width=rgb.shape[1], height=rgb.shape[0]
    )

    boundaries_gdf = gpd.read_file(os.environ['PATH_TO_NY_BOROUGH_BOUNDARIES'])
    boundary = rasterize(
        shapes=[boundaries_gdf.loc[i].geometry.boundary for i in range(len(boundaries_gdf))],
        out_shape=rgb.shape[:2],
        transform=transform
    )

    rgb[boundary == 1] = [23 / 255, 156 / 255, 28 / 255]
    plt.imshow(rgb)
    # plt.axis('off')
    # plt.xlim([eopatch.bbox.min_x, eopatch.bbox.max_x])
    # plt.ylim([eopatch.bbox.min_y, eopatch.bbox.max_y])
    plt.grid()
    plt.savefig(
        os.path.join(os.environ['VISUALIZATIONS_DIR'], 'whole_ny_rgb.png'),
        bbox_inches='tight',
        dpi=140
    )
    plt.show()
    plt.show()


def plot_sigma_vs_mse_mape():
    sigmas = ['$\sigma=0,$\nk=1', '$\sigma=0.75,$\nk=3', '$\sigma=1.5,$\nk=5', '$\sigma=2.25,$\nk=(7)']
    mses = [0.0256, 0.0121, 0.0140, 0.0125]
    mapes = [16.4, 14.6, 15.3, 14.9]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(mses, marker='x')
    plt.grid()
    plt.xticks([0, 1, 2, 3], sigmas)
    plt.ylim(0, 0.05)
    plt.gca().set_title('MSE')
    plt.subplot(1, 2, 2)
    plt.plot(mapes, marker='x', color='orange')
    plt.xticks([0, 1, 2, 3], sigmas)
    plt.ylim(12, 20)
    plt.grid()
    plt.gca().set_title('MAPE (в %)')
    plt.savefig(
        os.path.join(os.environ['VISUALIZATIONS_DIR'], 'sigma_vs_mse_mape.png'),
        bbox_inches='tight',
        dpi=80
    )

    plt.show()


if __name__ == '__main__':
    pass
    plot_and_save_bbox_splits()
    # plot_eopatch_rgb('eopatch_0216', title='Патч №216')
    # plot_eopatch_rgb('eopatch_0019', title='Патч №19')
    # plot_tree_annotations_and_density('eopatch_0019')
    # plot_tree_annotations_and_density('eopatch_0283')
    # plot_tree_annotations_and_density(['eopatch_0100', 'eopatch_0322'])
    # plot_eopatch_rgb_nir_ndvi(['eopatch_0100', 'eopatch_0322'])
    # plot_street_annotations_mask_rgb(['eopatch_0100', 'eopatch_0322'])
    # plot_whole_ny_eopatch()
    # plot_sigma_vs_mse_mape()