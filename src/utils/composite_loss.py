from typing import List, Optional, Union, Dict

from torch import nn, Tensor


class CompositeLoss(nn.Module):
    def __init__(
            self,
            losses: List[nn.Module],
            weights: Optional[Union[int, float]] = None,
            use_street_mask: bool = False
    ):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights if weights is not None else [1.] * len(losses)
        self.use_street_mask = use_street_mask

    def forward(self, outputs: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        targets = batch['density_map']
        street_mask = None
        if self.use_street_mask:
            street_mask = batch['street_mask']

        total_loss_value = 0.
        for weight, loss in zip(self.weights, self.losses):
            loss_value = loss(outputs, targets)
            if street_mask is not None:
                loss_value *= street_mask
            total_loss_value += weight * loss_value.sum()

        return total_loss_value
