import logging
import math
import os
import shutil
import sys
import time
from datetime import datetime

import hydra
import pyrootutils
import torch
import typer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError

from src.data.eopatch_dataset import EOPatchDataset
from src.utils.composite_metric import CompositeMetric
from src.utils.misc import seed_everything, move_batch_to_device
from src.utils.wandb_logger import WandBLogger
from src.utils.model_checkpointer import ModelCheckpointer


def run(cfg):
    num_epochs = cfg['num_epochs']
    val_interval = cfg['val_interval']
    artefacts_dir = cfg['artefacts_dir']

    if 'train_dataset' in cfg:
        train_dataset = hydra.utils.instantiate(cfg['train_dataset'])
    else:
        train_transform = hydra.utils.instantiate(cfg['train_transform'])
        train_dataset = EOPatchDataset(
            eopatches_dir=os.environ['EOPATCHES_DIR'],
            split='train',
            band_names_to_take=cfg['band_names_to_take'],
            to_take_ndvi=cfg['to_take_ndvi'],
            scale_rgb_intensity=cfg['scale_rgb_intensity'],
            mask_data=cfg['mask_data'],
            transform=train_transform
        )

    if 'val_dataset' in cfg:
        val_dataset = hydra.utils.instantiate(cfg['val_dataset'])
    else:
        val_transform = hydra.utils.instantiate(cfg['val_transform'])
        val_dataset = EOPatchDataset(
            eopatches_dir=os.environ['EOPATCHES_DIR'],
            split='val',
            band_names_to_take=cfg['band_names_to_take'],
            to_take_ndvi=cfg['to_take_ndvi'],
            scale_rgb_intensity=cfg['scale_rgb_intensity'],
            mask_data=cfg['mask_data'],
            transform=val_transform
        )

    batch_size = cfg['batch_size']
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=cfg['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=cfg['pin_memory']
    )

    device = torch.device(cfg['device'])

    model = hydra.utils.instantiate(cfg['model'])
    model = model.to(device)

    optimizer_partial = hydra.utils.instantiate(cfg['optimizer'])
    optimizer = optimizer_partial(model.parameters())

    scheduler = hydra.utils.instantiate(cfg['scheduler'])

    loss_function = hydra.utils.instantiate(cfg['loss'])

    num_images_to_log = cfg['num_images_to_log']
    wandb_logger = WandBLogger(cfg=cfg, model=model, save_config=True, num_images_to_log=num_images_to_log)

    train_composite_metric = CompositeMetric(
        density_metrics={
            'train/mse': MeanSquaredError().to(device)
        },
        tree_count_metrics={
            'train/cnt_mae': MeanAbsoluteError().to(device),
            'train/cnt_mape': MeanAbsolutePercentageError().to(device)
        }
    )

    val_composite_metric = CompositeMetric(
        density_metrics={
            'val/mse': MeanSquaredError().to(device)
        },
        tree_count_metrics={
            'val/cnt_mae': MeanAbsoluteError().to(device),
            'val/cnt_mape': MeanAbsolutePercentageError().to(device)
        }
    )

    model_checkpointer = ModelCheckpointer(
        checkpoints_dir=os.path.join(artefacts_dir, 'checkpoints'),
        save_top_k=cfg['save_top_k_checkpoints'],
        save_last=cfg['save_last_checkpoint']
    )
    best_metric = float('inf')
    best_metric_epoch = -1

    for epoch in range(num_epochs):
        epoch_start = time.time()
        logging.info("-" * 10)
        logging.info(f"epoch {epoch + 1}/{num_epochs}")

        model.train()
        epoch_loss = 0

        train_loader_iterator = iter(train_loader)

        for step in range(1, len(train_loader) + 1):
            step_start = time.time()

            batch = next(train_loader_iterator)
            batch = move_batch_to_device(batch, device)
            inputs, targets = (
                batch['data'],
                batch['density_map']
            )

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, batch)
            loss.backward()
            optimizer.step()

            if cfg['loss'].get('use_street_mask'):
                outputs = outputs.detach() * batch['street_mask']

            pred_tree_counts = outputs.sum(dim=(1, 2, 3))
            tgt_tree_counts = batch['tree_count']
            train_composite_metric(outputs, targets, pred_tree_counts, tgt_tree_counts)

            epoch_loss += loss.item()
            epoch_len = math.ceil(len(train_dataset) / train_loader.batch_size)
            logging.info(
                f'[{step}/{epoch_len}] train_loss: {loss.item():.4f},'
                f' bandwidth: {(time.time() - step_start) / train_loader.batch_size:.4f}'
            )
            wandb_logger.log_scalar('train/loss', loss.item())
            if (epoch + 1) % val_interval == 0:
                if len(train_loader) - step < num_images_to_log // batch_size + 1:
                    wandb_logger.log_prediction_as_image('image/train', outputs, pred_tree_counts, batch)

        if scheduler is not None:
            scheduler.step()

        epoch_loss /= step
        density_metric_values, tree_count_metric_values = train_composite_metric.compute()
        train_composite_metric.reset()
        wandb_logger.reset_image_logging()

        logging.info(f'epoch {epoch + 1} average loss: {epoch_loss:.4f}, MAPE: {tree_count_metric_values["train/cnt_mape"]:.4f}')
        wandb_logger.log_scalar('train/epoch_loss', epoch_loss)
        wandb_logger.log({**density_metric_values, **tree_count_metric_values})

        current_lr = scheduler.get_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        wandb_logger.log_scalar('train/lr', current_lr)
        wandb_logger.log_scalar('epoch', epoch)

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loader_iterator = iter(val_loader)

                for i in range(len(val_loader)):
                    val_batch = next(val_loader_iterator)
                    val_batch = move_batch_to_device(val_batch, device)
                    val_inputs, val_targets = (
                        val_batch['data'],
                        val_batch['density_map'],
                    )

                    val_outputs = model(val_inputs)
                    if cfg['loss'].get('use_street_mask'):
                        val_outputs *= val_batch['street_mask']

                    val_pred_tree_counts = val_outputs.sum(dim=(1, 2, 3))
                    val_tgt_tree_counts = val_batch['tree_count']
                    val_composite_metric(val_outputs, val_targets, val_pred_tree_counts, val_tgt_tree_counts)
                    wandb_logger.log_prediction_as_image('image/val', val_outputs, val_pred_tree_counts, val_batch)

                wandb_logger.reset_image_logging()
                val_density_metric_values, val_tree_count_metric_values = val_composite_metric.compute()
                val_composite_metric.reset()
                # val_mse_metric.reset()

                val_mse = val_density_metric_values['val/mse']
                if val_mse < best_metric:
                    best_metric = val_mse
                    best_metric_epoch = epoch + 1

                model_checkpointer(val_mse, epoch, model, optimizer, scheduler)

                logging.info(
                    f'current epoch: {epoch + 1},'
                    f' current mse: {val_mse:.4f},'
                    f' best mse: {best_metric:.4f}'
                    f' at epoch: {best_metric_epoch}'
                    f' val MAPE: {val_tree_count_metric_values["val/cnt_mape"]}'
                )
                # wandb_logger.log_scalar('val/mse', val_mse)
                wandb_logger.log({**val_density_metric_values, **val_tree_count_metric_values})

        logging.info(
            f'time consuming of epoch {epoch + 1} is:'
            f' {(time.time() - epoch_start):.4f}'
        )
    wandb_logger.finish()


def main(config_name: str = typer.Option('test.yaml')):
    pyrootutils.setup_root(
        __file__,
        indicator='.project-root',
        project_root_env_var=True,
        dotenv=True,
        pythonpath=True,
        cwd=False
    )

    cfg_pth = os.path.join(os.environ['PROJECT_ROOT'], 'src', 'configs', 'experiment', config_name)
    cfg = OmegaConf.load(cfg_pth)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    run_name = os.path.splitext(config_name)[0]
    artefacts_dir = os.path.join(os.environ['PROJECT_ROOT'], 'artefacts', run_name)
    if os.path.exists(artefacts_dir):
        print(f'Run with name "{run_name}" already exists. Do you want to erase it? [y/N]')
        to_erase = input().lower()
        if to_erase in ['y', 'yes']:
            shutil.rmtree(artefacts_dir)
        else:
            now = datetime.now().strftime("%m-%d_%H-%M-%S")
            artefacts_dir = '_'.join([artefacts_dir, now])

    os.makedirs(os.path.join(artefacts_dir, 'checkpoints'))

    cfg['artefacts_dir'] = artefacts_dir
    cfg['run_name'] = run_name
    OmegaConf.save(cfg, os.path.join(artefacts_dir, 'train_config.yaml'))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(cfg['artefacts_dir'], 'log.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f'Run artefacts will be saved to {cfg["artefacts_dir"]} directory.')

    seed_everything(seed=42)
    run(cfg)


if __name__ == '__main__':
    typer.run(main)
