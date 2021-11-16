import torch
from torch import nn


loss_dict = {
    'mse': MSELoss
}


class MSELoss(nn.Module):
    """Mean square error loss module"""
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, preds, targets):
        loss = self.loss(preds['rgb_coarse'], targets)
        if 'rgb_fine' in preds:
            loss += self.loss(preds['rgb_fine'], targets)

        return loss
