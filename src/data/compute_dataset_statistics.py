import os

import numpy as np
import pyrootutils
from eolearn.core import EOPatch
from tqdm import tqdm
from src.data.get_data import BAND_NAMES


def main(eopatches_dir: str):
    pixel_sum = np.zeros((len(BAND_NAMES) + 1,), dtype=float)
    pixel_sum_sq = np.zeros((len(BAND_NAMES) + 1,), dtype=float)
    pixel_count = 0

    for eopatch_name in tqdm(os.listdir(eopatches_dir)):
        eopatch_pth = os.path.join(eopatches_dir, eopatch_name)
        eopatch = EOPatch.load(eopatch_pth)
        data = eopatch.data['L2A_BANDS'][0]  # [H, W, Nb]
        ndvi = eopatch.data['NDVI'][0]
        data = np.concatenate([data, ndvi], axis=-1)

        pixel_sum += data.sum(axis=(0, 1))
        pixel_sum_sq += np.square(data).sum(axis=(0, 1))
        pixel_count += data.shape[0] * data.shape[1]

    mean = pixel_sum / pixel_count
    std = np.sqrt((pixel_sum_sq / pixel_count - np.square(mean)))
    print(mean, std)


if __name__ == '__main__':
    pyrootutils.setup_root(__file__, project_root_env_var=True, dotenv=True, pythonpath=True)

    eopatches_dir = os.environ['EOPATCHES_DIR']
    main(eopatches_dir)
