from typing import Any, Dict, Callable

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric
from torch.optim import Optimizer


class UNetModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        criterion: Callable,
        optimizer_class: Any,
        optimizer_kwargs: Dict[str, Any],
        scheduler_class: Any,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=['net'])

        self.net = net
        self.criterion = criterion

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def model_step(self, batch: Dict):
        data, density_maps = batch['data'], batch['density_map']
        outputs = self.forward(data)
        loss = self.criterion(outputs, density_maps)
        return loss, outputs, density_maps

    def training_step(self, batch: Any, batch_idx: int):
        loss, outputs, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log('train/loss', self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, outputs, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_class(params=self.parameters(), **self.hparams.optimizer_kwargs)
        if self.hparams.scheduler_class is not None:
            scheduler = self.hparams.scheduler_class(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {'optimizer': optimizer}
