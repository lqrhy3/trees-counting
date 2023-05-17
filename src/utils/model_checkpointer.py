import glob
import logging
import os.path
from typing import List, Tuple
from src.utils.misc import save_checkpoint


class ModelCheckpointer:
    def __init__(self, checkpoints_dir: str, save_top_k: int, save_last: bool):
        self.checkpoints_dir = checkpoints_dir
        self.save_top_k = save_top_k
        self.save_last = save_last

        self._top_k_checkpoints: List[Tuple[float, str]] = []

        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def __call__(self, value, epoch, model, optimizer, scheduler):
        if self._to_skip():
            return

        if len(self._top_k_checkpoints) < self.save_top_k:
            checkpoint_name = f'best_epoch_{epoch:04d}_value_{value:.4f}.pth'
            self._top_k_checkpoints.append((value, checkpoint_name))
            self._top_k_checkpoints.sort(key=lambda key: key[0])

            save_checkpoint(
                model=model,
                epoch=epoch,
                checkpoints_dir=self.checkpoints_dir,
                checkpoint_name=checkpoint_name,
                optimizer=optimizer,
                scheduler=scheduler,
                save_only_one=False
            )

            logging.info(f'saved new top-{self.save_top_k} best metric model.')

        elif value < max(self._top_k_checkpoints, key=lambda key: key[0])[0]:
            checkpoint_name = f'best_epoch_{epoch:04d}_value_{value:.4f}.pth'
            _, checkpoint_name_to_delete = self._top_k_checkpoints.pop()
            os.remove(os.path.join(self.checkpoints_dir, checkpoint_name_to_delete))

            self._top_k_checkpoints.append((value, checkpoint_name))
            self._top_k_checkpoints.sort(key=lambda key: key[0])

            save_checkpoint(
                model=model,
                epoch=epoch,
                checkpoints_dir=self.checkpoints_dir,
                checkpoint_name=checkpoint_name,
                optimizer=optimizer,
                scheduler=scheduler,
                save_only_one=False
            )

            logging.info(f'saved new top-{self.save_top_k} best metric model.')

        if self.save_last:
            checkpoint_pths_to_delete = glob.glob(os.path.join(self.checkpoints_dir, 'last_*'))
            for checkpoint_pth_to_delete in checkpoint_pths_to_delete:
                os.remove(checkpoint_pth_to_delete)

            checkpoint_name = f'last_epoch_{epoch:04d}.pth'
            save_checkpoint(
                model=model,
                epoch=epoch,
                checkpoints_dir=self.checkpoints_dir,
                checkpoint_name=checkpoint_name,
                optimizer=optimizer,
                scheduler=scheduler,
                save_only_one=False
            )

    def _to_skip(self):
        return self.save_top_k == 0 and not self.save_last
