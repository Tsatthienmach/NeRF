import torch
from torch import nn


class MSELoss(nn.Module):
    """Mean square error loss module"""
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, preds, targets):
        coarse_loss = self.loss(preds['rgb_coarse'], targets)
        losses = {
            'coarse': coarse_loss,
            'total': coarse_loss
        }
        if 'rgb_fine' in preds:
            fine_loss = self.loss(preds['rgb_fine'], targets)
            losses['fine'] = fine_loss
            losses['total'] = losses['total'] + fine_loss

        return losses


loss_dict = {
    'mse': MSELoss
}
