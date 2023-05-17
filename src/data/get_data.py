import datetime
import os
from typing import List, Union, Tuple

import geopandas as gpd
import numpy as np
import pyrootutils
from eolearn.core import EOTask, EOPatch, FeatureType, SaveTask, OverwritePermission, linearly_connect_tasks, EOWorkflow, EOExecutor
from eolearn.features import LinearInterpolationTask
from matplotlib import pyplot as plt
from sentinelhub import UtmZoneSplitter, DataCollection, get_area_dates, MosaickingOrder
from sentinelsat import read_geojson
from shapely.geometry import shape
from eolearn.io import SentinelHubInputTask

RESOLUTION = 10
BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
BBOX_SIZE = 128


class AddValidityMaskTask(EOTask):
    def __init__(self, mask_name: str = 'IS_VALID'):
        self.mask_name = mask_name

    def execute(self, eopatch: EOPatch):
        mask = eopatch.mask['IS_DATA'].astype(bool) & (~eopatch.mask['CLM'].astype(bool))
        eopatch[FeatureType.MASK, self.mask_name] = mask
        return eopatch


class AddValidCoverageScalarTask(EOTask):
    def __init__(self, scalar_name: str = 'CC'):
        self.scalar_name = scalar_name

    def execute(self, eopatch):
        num_valid_pixels = np.sum(eopatch.mask['IS_VALID'], axis=(1, 2, 3)) / np.prod(eopatch.mask['IS_VALID'].shape[1:])
        num_valid_pixels = num_valid_pixels[:, None]  # [T, 1]
        eopatch[FeatureType.SCALAR, self.scalar_name] = num_valid_pixels
        return eopatch


def main(
        bbox_size: int,
        band_names: List[str],
        maxcc: float,
        time_interval: Union[str, tuple],
        eopatches_dir: str,
        max_threads: int
):
    bboxes = get_bboxes(bbox_size=bbox_size)
    workflow_nodes = compose_workflow_nodes(
        band_names=band_names,
        maxcc=maxcc,
        max_threads=max_threads,
        eopatches_dir=eopatches_dir,
    )

    input_node = workflow_nodes[0]
    save_node = workflow_nodes[-1]
    execution_args = []
    for idx, bbox in enumerate(bboxes):
        execution_args.append(
            {
                input_node: {'bbox': bbox, 'time_interval': time_interval},
                save_node: {'eopatch_folder': f'eopatch_{str(idx).zfill(4)}'},
            }
        )

    # Execute the workflow
    executor = EOExecutor(EOWorkflow(workflow_nodes), execution_args, save_logs=True)
    executor.run(workers=4)

    executor.make_report()

    failed_ids = executor.get_failed_executions()
    if failed_ids:
        raise RuntimeError(
            f"Execution failed EOPatches with IDs:\n{failed_ids}\n"
            f"For more info check report at {executor.get_report_path()}"
        )


def get_bboxes(bbox_size: int):
    boundaries_gdf = gpd.read_file(os.environ['PATH_TO_NY_BOROUGH_BOUNDARIES'])
    boundaries_geojson = read_geojson(os.environ['PATH_TO_NY_BOROUGH_BOUNDARIES'])
    boundary_geometries = [shape(boundaries_geojson['features'][i]['geometry']) for i in range(len(boundaries_gdf))]
    bbox_splitter = UtmZoneSplitter(
        boundary_geometries, crs=boundaries_gdf.crs, bbox_size=bbox_size * RESOLUTION
    )
    bbox_list = np.array([bbox.transform(boundaries_gdf.crs) for bbox in bbox_splitter.get_bbox_list()])
    return bbox_list


def compose_workflow_nodes(
        band_names: List[str],
        maxcc: float,
        max_threads: int,
        eopatches_dir: str,
):
    input_task = SentinelHubInputTask(
        data_collection=DataCollection.SENTINEL2_L2A,
        bands=band_names,
        bands_feature=(FeatureType.DATA, 'L2A_BANDS'),
        additional_data=[(FeatureType.MASK, 'dataMask', 'IS_DATA'), (FeatureType.MASK, 'CLM')],
        resolution=RESOLUTION,
        maxcc=maxcc,
        single_scene=True,
        mosaicking_order=MosaickingOrder.LEAST_CC,
        max_threads=max_threads
    )

    add_validity_mask_task = AddValidityMaskTask(mask_name='IS_VALID')
    save_task = SaveTask(eopatches_dir, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    workflow_nodes = linearly_connect_tasks(
        input_task,
        add_validity_mask_task,
        save_task
    )

    return workflow_nodes


if __name__ == '__main__':
    pyrootutils.setup_root(__file__, project_root_env_var=True, dotenv=True, pythonpath=True)

    maxcc = 0.5
    time_interval = ('2017-06-01', '2017-09-20')
    eopatches_dir = os.environ['EOPATCHES_DIR']
    max_threads = 3

    main(
        bbox_size=BBOX_SIZE,
        band_names=BAND_NAMES,
        maxcc=maxcc,
        time_interval=time_interval,
        eopatches_dir=eopatches_dir,
        max_threads=max_threads
    )

    # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 25))

    # for i, name in enumerate(os.listdir(eopatches_dir)):
    #     eopatch_path = os.path.join(eopatches_dir, name)
    #     # eopatch_path = os.path.join(eopatches_dir, f"eopatch_{idx}")
    #     eopatch = EOPatch.load(eopatch_path, lazy_loading=True)
    #
    #     # ax = axs[i // 2][i % 2]
    #     # ax.imshow(np.clip(eopatch.data["L2A_BANDS"][0][..., [3, 2, 1]] * 3.5, 0, 1))
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(np.clip(eopatch.data["L2A_BANDS"][2][..., [3, 2, 1]] * 3.5, 0, 1))
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(eopatch.mask['CLM'][2])
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(eopatch.mask['IS_DATA'][2])
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_aspect("auto")
        # del eopatch
        # plt.show()

    # fig.subplots_adjust(wspace=0, hspace=0)
    # plt.show()

