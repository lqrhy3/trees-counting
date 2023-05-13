import os
from typing import Union

import geopandas as gpd
from eolearn.core import LoadTask, FeatureType, EOPatch, SaveTask, OverwritePermission, linearly_connect_tasks, \
    EOExecutor, EOWorkflow, EOTask
from eolearn.geometry.transformations import VectorToRasterTask
from pyrootutils import pyrootutils
from rasterio.enums import MergeAlg
from scipy.ndimage import gaussian_filter

from src.utils.misc import read_df_with_coordinates


class AddTreesDensityDataTask(EOTask):
    def __init__(self, sigma: Union[int, float], radius: int):
        self.sigma = sigma
        self.radius = radius

    def execute(self, eopatch: EOPatch):
        trees_annotations = eopatch[FeatureType.MASK_TIMELESS, 'TREES_ANNOTATIONS']  # [H, W, 1]
        trees_density = gaussian_filter(
            trees_annotations.astype(float),
            mode='constant',
            sigma=self.sigma,
            radius=self.radius
        )
        eopatch[FeatureType.DATA_TIMELESS, 'TREES_DENSITY'] = trees_density
        return eopatch


class AddNumberOfTreesScalarTask(EOTask):
    def execute(self, eopatch: EOPatch,):
        num_trees = eopatch.mask_timeless['TREES_ANNOTATIONS'].sum()
        eopatch[FeatureType.SCALAR_TIMELESS, 'NUM_TREES'] = num_trees
        return eopatch


def main(
        eopatches_dir: str,
        sigma: Union[int, float],
        radius: int
):
    eopatch_names = os.listdir(eopatches_dir)
    trees_gdf = read_df_with_coordinates(os.environ['PATH_TO_TREE_LABELS'])

    workflow_nodes = compose_workflow_nodes(
        eopatches_dir=eopatches_dir,
        trees_gdf=trees_gdf,
        sigma=sigma,
        radius=radius
    )

    load_node = workflow_nodes[0]
    save_node = workflow_nodes[-1]
    execution_args = []
    for eopatch_name in eopatch_names:
        execution_args.append(
            {
                load_node: {'eopatch_folder': eopatch_name},
                save_node: {'eopatch_folder': eopatch_name},
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


def compose_workflow_nodes(
        eopatches_dir: str,
        trees_gdf: gpd.GeoDataFrame,
        sigma: Union[int, float],
        radius: int
):
    load_task = LoadTask(path=eopatches_dir)
    add_trees_annotations_task = VectorToRasterTask(
        vector_input=trees_gdf,
        raster_feature=(FeatureType.MASK_TIMELESS, 'TREES_ANNOTATIONS'),
        values=1,
        raster_shape=(128, 128),
        raster_dtype=int,
        no_data_value=0,
        merge_alg=MergeAlg.add
    )
    add_trees_density_task = AddTreesDensityDataTask(sigma=sigma, radius=radius)
    add_num_trees_task = AddNumberOfTreesScalarTask()
    save_task = SaveTask(path=eopatches_dir, overwrite_permission=OverwritePermission.OVERWRITE_FEATURES)

    tasks = [load_task, add_trees_annotations_task, add_trees_density_task, add_num_trees_task, save_task]
    workflow_nodes = linearly_connect_tasks(*tasks)
    return workflow_nodes


if __name__ == '__main__':
    pyrootutils.setup_root(__file__, project_root_env_var=True, dotenv=True, pythonpath=True)

    eopatches_dir = '/home/lqrhy3/PycharmProjects/trees-counting/data/raw/eopatches'
    sigma = 0.75
    radius = 1

    main(
        eopatches_dir=eopatches_dir,
        sigma=sigma,
        radius=radius
    )
    # load_task = LoadTask(eopatches_dir, lazy_loading=True)
    # for eopatch_name in os.listdir(eopatches_dir):
    #     eopatch = load_task.execute(eopatch_folder=eopatch_name)

        # plt.figure(figsize=(20, 20))
        # plt.subplot(1, 3, 1)
        # plt.imshow(np.clip(eopatch.data['L2A_BANDS'][0][..., [3, 2, 1]] * 3.5, 0, 1))
        # plt.subplot(1, 3, 2)
        # plt.imshow(eopatch.mask_timeless['TREES_ANNOTATIONS'][:, :, 0])
        # plt.subplot(1, 3, 3)
        # plt.imshow(eopatch.data_timeless['TREES_DENSITY'][:, :, 0])
        # plt.show()

