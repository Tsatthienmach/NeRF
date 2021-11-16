import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


class Writer:
    """Tensorboard logging writer.
    Tasks:
        - save loss every epoch
        - save PSNR
        - save SSIM
        - save validation images

    Args:
        save_dir (str): saving direction
        exp_name (str): experiment name
        sfx (str): suffix of the experiment
        i_image (int): save images every i_image epoch
    """

    def __init__(self, exp_name, save_dir='logs', sfx='', i_image=5):
        self.i_image = i_image
        sfx = f'_{sfx}' if sfx else ''
        self.log_dir = os.path.join(save_dir, f'{exp_name}{sfx}')
        self.writer = SummaryWriter(log_dir=f'{self.log_dir}/tensorboard')

    def save_loss(self, loss, epoch, prefix):
        self.writer.add_scalar(f'Loss/{prefix}', loss, epoch)

    def save_metrics(self, metrics, epoch, prefix):
        """Save multiple metric results.

        Args:
            metrics (dict): Ex. {'psnr': 43.3, 'ssim': 22.2}
            epoch (int): current epoch
            prefix (str): train/val
        """
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f'Metric/{prefix}/{metric_name}',
                                   metric_value,
                                   epoch)

    def save_imgs(self, pred_imgs, gt_imgs, epoch, sfx=''):
        """Save validation images.

        Args:
            pred_imgs (tensor | Bs, 3, H, W): predicted images
            gt_imgs (tensor | Bs, 3, H, W): ground truth images
            epoch (int): current epoch
            sfx (str): suffix for special cases
        """
        if epoch % self.i_image == 0:
            merged_images = torch.cat([gt_imgs, pred_imgs], dim=-2)
            self.writer.add_images('Results', merged_images, epoch)

        if sfx != '':
            merged_images = torch.cat([gt_imgs, pred_imgs], dim=-2)
            self.writer.add_images(f'Results/{sfx}', merged_images, epoch)
