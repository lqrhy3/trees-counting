import logging
import os
import sys

import hydra.utils
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, Union

import pyrootutils
import typer
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError

from src.data.eopatch_dataset import EOPatchDataset
from src.utils.misc import seed_everything, load_checkpoint, denormalize_imagenet


def main(config_name: str = 'eval.yaml'):
    pyrootutils.setup_root(
        __file__,
        indicator='.project-root',
        project_root_env_var=True,
        dotenv=True,
        pythonpath=True,
        cwd=False
    )

    cfg_pth = os.path.join(os.environ['PROJECT_ROOT'], 'src', 'configs', 'evaluation', config_name)
    cfg = OmegaConf.load(cfg_pth)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    eval_artefacts_dir = os.path.join(cfg['artefacts_dir'], 'eval_artefacts', os.path.splitext(cfg['checkpoint_name'])[0])
    os.makedirs(eval_artefacts_dir, exist_ok=True)
    os.makedirs(os.path.join(eval_artefacts_dir, 'images'), exist_ok=True)

    cfg['eval_artefacts_dir'] = eval_artefacts_dir
    cfg['path_to_checkpoint'] = os.path.join(cfg['artefacts_dir'], 'checkpoints', cfg['checkpoint_name'])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(eval_artefacts_dir, 'log.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

    train_cfg_pth = os.path.join(cfg['artefacts_dir'], 'train_config.yaml')
    train_cfg = OmegaConf.load(train_cfg_pth)
    train_cfg = OmegaConf.to_container(train_cfg)

    if cfg.get('model') is not None:
        logging.info('Model topology will be overloaded by eval config.')
    else:
        logging.info('Model topology will be derived from train config.')
        assert train_cfg is not None
        cfg['model'] = train_cfg['model']

    if cfg.get('val_transform') is not None:
        logging.info('Transforms will be overloaded by eval config.')
    else:
        logging.info('Transforms will be derived from train config.')
        assert train_cfg is not None
        cfg['transform'] = train_cfg['val_transform']

    seed_everything(42)
    run(cfg)


def run(cfg: Union[Dict, DictConfig]):
    device = torch.device(cfg['device'])
    model = hydra.utils.instantiate(cfg['model'])
    load_checkpoint(model, cfg['path_to_checkpoint'], device)
    model.eval()

    transform = hydra.utils.instantiate(cfg['transform'])
    dataset = EOPatchDataset(
        os.environ['EOPATCHES_DIR'],
        split='val',
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=cfg['pin_memory']
    )

    mse_metric = MeanSquaredError().to(device)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i > 5:
                break
            eopatch_name = dataset.eopatch_names[i]
            data, targets = batch['data'], batch['density_map']
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            mse_val = mse_metric(outputs, targets)
            logging.info(f'{eopatch_name}, mse: {mse_val:.4f}')
            plot_and_save_prediction(
                batch,
                outputs,
                save_dir=os.path.join(cfg['eval_artefacts_dir'], 'images'),
                eopatch_name=eopatch_name,
                show=False
            )

        mse_val = mse_metric.compute().item()
        logging.info(f'overall mse: {mse_val:.4f}')


def plot_and_save_prediction(batch, outputs, save_dir, eopatch_name, show):
    data = batch['data'].cpu()
    data = np.clip(denormalize_imagenet(data) * 2.5, 0, 1)
    data = data[0].permute(1, 2, 0).numpy()

    targets = batch['density_map'].cpu()
    targets = targets[0].permute(1, 2, 0).numpy()
    outputs = outputs.cpu()
    outputs = outputs[0].permute(1, 2, 0).numpy()

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(outputs, vmin=0., vmax=1.)
    axs[0].set_title('predicted density', fontsize=8)
    axs[0].set_axis_off()

    axs[1].imshow(data)
    axs[1].set_title('data', fontsize=8)
    axs[1].set_axis_off()

    axs[2].imshow(targets, vmin=0., vmax=1.)
    axs[2].set_title('target density', fontsize=8)
    axs[2].set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{eopatch_name}.png'), bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    typer.run(main)
