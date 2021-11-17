import torch
from abc import ABC
from .base_metric import BaseMetric
from skimage.metrics import structural_similarity as ssim


class MSE(BaseMetric, ABC):
    """Mean square error metric"""

    def __init__(self):
        self.preds = []
        self.gts = []

    def reset(self):
        self.preds = []
        self.gts = []

    def update(self, pred, gt):
        """Add prediction and ground-truth

        Args:
            pred (tensor | TODO): prediction
            gt (tensor | TODO): ground-truth
        """
        self.preds.append(pred)
        self.gts.append(gt)

    def compute(self):
        mses = [self.formula(self.preds[i], self.gts[i]) for i in range(len(self.preds))]
        return torch.mean(torch.Tensor(mses))

    @staticmethod
    def formula(x, y):
        return torch.mean((x-y) ** 2)


class PSNR(BaseMetric):
    """Peak signal-to-noise ratio metric"""
    def __init__(self):
        self.mses = []

    def reset(self):
        self.mses = []

    def update(self, pred, gt):
        self.mses.append(
            MSE.formula(pred, gt)
        )

    def compute(self):
        psnrs = [-10 * torch.log10(mse) for mse in self.mses]
        return torch.mean(torch.Tensor(psnrs))


class SSIM(BaseMetric):
    """Structural similarity index measure is used for measuring
    the similarity between two images."""
    def __init__(self):
        self.preds = []
        self.gts = []

    def reset(self):
        self.preds = []
        self.gts = []

    def update(self, pred, gt):
        self.preds.append(pred)
        self.gts.append(gt)

    def compute(self):
        ssims = [ssim(self.gts[i], self.preds[i]) for i in range(len(self.preds))]
        return torch.mean(torch.Tensor(ssims))
