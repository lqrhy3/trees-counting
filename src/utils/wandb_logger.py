import os
from typing import Optional, Dict

import cv2
import numpy as np
import torch
import wandb
from torch import nn

from src.utils.misc import denormalize_imagenet


class WandBLogger:
    def __init__(
            self,
            cfg: dict,
            model: nn.Module,
            save_config: bool,
            num_images_to_log: int
    ):
        self.cfg = cfg
        self.num_images_to_log = num_images_to_log

        self.run = wandb.init(
            dir=os.path.split(cfg['artefacts_dir'])[0],
            project='trees-counting',
            name=self.cfg['run_name'],
            config=self.cfg,
            resume='allow',
            # mode='offline'
        )

        wandb.watch(model, log='all', log_freq=100, log_graph=True)
        if save_config:
            cfg['wandb_dir'] = self.run.dir
            wandb.save(os.path.join(cfg['artefacts_dir'], 'train_config.yaml'), policy='now')

        self._num_logged_images = 0

    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs)

    def log_scalar(self, name: str, value: float, commit: Optional[bool] = None):
        self.log({name: value}, commit=commit)

    def log_prediction_as_image(self, name: str, outputs: torch.Tensor, batch: Dict[str, torch.Tensor]):
        num_samples = self.num_images_to_log - self._num_logged_images

        outputs = outputs[:num_samples].detach().cpu().numpy()
        data = batch['data'][:num_samples].detach().cpu()
        data = denormalize_imagenet(data).numpy()
        tgt_density_maps = batch['density_map'][:num_samples].detach().cpu().numpy()

        self._num_logged_images += len(outputs)

        for i in range(len(outputs)):
            tgt_density_map_i = tgt_density_maps[i, 0]
            max_density_value = tgt_density_map_i.max()

            output_i = outputs[i, 0]
            output_i = (output_i / max_density_value * 255).astype(np.uint8)
            output_i = cv2.applyColorMap(output_i, cv2.COLORMAP_VIRIDIS)
            output_i = cv2.cvtColor(output_i, cv2.COLOR_BGR2RGB)

            tgt_density_map_i = (tgt_density_map_i / max_density_value * 255).astype(np.uint8)
            tgt_density_map_i = cv2.applyColorMap(tgt_density_map_i, cv2.COLORMAP_VIRIDIS)
            tgt_density_map_i = cv2.cvtColor(tgt_density_map_i, cv2.COLOR_BGR2RGB)

            data_i = np.clip(data[i].transpose((1, 2, 0)) * 2.5 * 255, 0, 255).astype(np.uint8)

            image = np.concatenate([output_i, data_i, tgt_density_map_i], axis=1)
            image = wandb.Image(
                image,
                caption='prediction vs target'
            )

            self.log({name: image})

    def finish(self):
        self.run.finish()

    def reset_image_logging(self):
        self._num_logged_images = 0
