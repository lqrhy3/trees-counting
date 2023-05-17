import os
from enum import Enum

import pyrootutils

from src.utils.misc import get_eopatches_split_dir


class SplitType(Enum):
    EVERY_NTH = 1


def main(eopatches_dir: str, val_size: float, split_type: SplitType):
    eopatch_names = [
            x for x in os.listdir(eopatches_dir)
            if os.path.isdir(os.path.join(eopatches_dir, x))
        ]
    eopatch_names = sorted(eopatch_names)

    num_eopathces = len(eopatch_names)

    if split_type == SplitType.EVERY_NTH:
        num_val_eopatches = int(num_eopathces * val_size)
        every_nth = num_eopathces // num_val_eopatches

        train_eopatch_names = [eopatch_names[i] for i in range(num_eopathces) if i % every_nth != 0]
        val_eopatch_names = eopatch_names[::every_nth]
    else:
        raise RuntimeError('Unknown split type.')

    eopatches_split_dir = get_eopatches_split_dir(eopatches_dir)
    os.makedirs(eopatches_split_dir, exist_ok=True)

    with open(os.path.join(eopatches_split_dir, 'train.txt'), 'w') as train_split_file:
        train_split_file.writelines(
            list(map(lambda x: f'{x}\n', train_eopatch_names))
        )

    with open(os.path.join(eopatches_split_dir, 'val.txt'), 'w') as train_split_file:
        train_split_file.writelines(
            list(map(lambda x: f'{x}\n', val_eopatch_names))
        )

    return train_eopatch_names, val_eopatch_names


if __name__ == '__main__':
    pyrootutils.setup_root(__file__, project_root_env_var=True, dotenv=True, pythonpath=True)

    eopathces_dir = os.environ['EOPATCHES_DIR']
    val_size = 0.15
    split_type = SplitType.EVERY_NTH

    main(
        eopatches_dir=eopathces_dir,
        val_size=val_size,
        split_type=split_type
    )
