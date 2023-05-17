import os

import numpy as np
import pyrootutils
from eolearn.core import EOPatch
from src.data.get_data import BAND_NAMES


def main(eopatches_dir: str):
    pixel_sum = np.zeros((len(BAND_NAMES,)), dtype=float)
    pixel_sum_sq = np.zeros((len(BAND_NAMES),), dtype=float)

    for eopatch_name in os.listdir(eopatches_dir):
        eopatch_pth = os.path.join(eopatches_dir, eopatch_name)
        eopatch = EOPatch.load(eopatch_pth)
        data = eopatch.data['L2A_BANDS'][0]  # [H, W, Nb]



if __name__ == '__main__':
    pyrootutils.setup_root(__file__, project_root_env_var=True, dotenv=True, pythonpath=True)

    eopatches_dir = os.environ['EOPATCHES_DIR']
    main(eopatches_dir)