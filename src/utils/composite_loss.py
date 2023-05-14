from typing import List, Optional, Union

from torch import nn, Tensor

nn.MSELoss


class CompositeLoss(nn.Module):
    def __init__(self, losses: List[nn.Module], weights: Optional[Union[int, float]] = None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights if weights is not None else [1.] * len(losses)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss_value = 0.
        for weight, loss in zip(self.weights, self.losses):
            loss_value += weight * loss(input, target)

        return loss_value
