import os
import geopandas as gpd
import pyproj
import pyrootutils
from dotenv import load_dotenv
from eolearn.core import FeatureType, SaveTask, OverwritePermission, linearly_connect_tasks, EOExecutor, EOWorkflow
from eolearn.io import SentinelHubInputTask
from sentinelhub import DataCollection, MosaickingOrder
from sentinelhub.geometry import BBox

RESOLUTION = 100


def main(eopatch_dir, time_interval):
    boundaries_gdf = gpd.read_file(os.environ['PATH_TO_NY_BOROUGH_BOUNDARIES'])
    bbox = boundaries_gdf.total_bounds.tolist()
    bbox = BBox(bbox, crs=boundaries_gdf.crs)
    del boundaries_gdf

    input_task = SentinelHubInputTask(
        data_collection=DataCollection.SENTINEL2_L2A,
        bands=['B04', 'B03', 'B02'],
        bands_feature=(FeatureType.DATA, 'L2A_BANDS'),
        resolution=100,
        maxcc=0.2,
        single_scene=True,
        mosaicking_order=MosaickingOrder.LEAST_CC,
        max_threads=4
    )
    save_task = SaveTask(eopatch_dir, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    workflow_nodes = linearly_connect_tasks(
        input_task,
        save_task
    )

    execution_args = [{
        workflow_nodes[0]: {'bbox': bbox, 'time_interval': time_interval},
        workflow_nodes[-1]: {'eopatch_folder': 'eopatch'}
    }]
    executor = EOExecutor(EOWorkflow(workflow_nodes), execution_args, save_logs=True)
    executor.run(workers=4)


if __name__ == '__main__':
    pyrootutils.setup_root(__file__, project_root_env_var=True, dotenv=True, pythonpath=True)
    eopatch_dir = '/home/lqrhy3/PycharmProjects/trees-counting/data/raw/whole_ny_eopatch'
    time_interval = ('2017-05-01', '2017-09-20')
    main(eopatch_dir=eopatch_dir, time_interval=time_interval)

