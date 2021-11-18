import os
import torch
from torch.utils.tensorboard import SummaryWriter


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

    def save_loss(self, loss, epoch, pfx):
        self.writer.add_scalar(f'Loss/{pfx}', loss, epoch)

    def save_metrics(self, metrics, epoch, pfx):
        """Save multiple metric results.

        Args:
            metrics (dict): Ex. {'psnr': 43.3, 'ssim': 22.2}
            epoch (int): current epoch
            pfx (str): train/val
        """
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f'Metric/{pfx}/{metric_name}',
                                   metric_value,
                                   epoch)

    def save_imgs(self, pred_imgs, gt_imgs, epoch, sfx='', data_format='NHWC'):
        """Save validation images.

        Args:
            pred_imgs (tensor | Bs, H, W, C): predicted images
            gt_imgs (tensor | Bs, H, W, C): ground truth images
            epoch (int): current epoch
            sfx (str): suffix for special cases
            data_format (str): Image data format specification of the form
                NCHW, NHWC, CHW, HWC, HW, WH, etc
        """
        if epoch % self.i_image == 0:
            merged_images = torch.cat([gt_imgs, pred_imgs], dim=-3)
            self.writer.add_images('Results', merged_images, epoch,
                                   dataformats=data_format)

        if sfx != '':
            merged_images = torch.cat([gt_imgs, pred_imgs], dim=-3)
            self.writer.add_images(f'Results/{sfx}', merged_images, epoch,
                                   dataformats=data_format)

    def save_depths(self, depths, epoch, sfx=''):
        """Save depth of predictions

        Args:
            depths (tensor | BxHxW)
            epoch (int): current epoch
            sfx (str): suffix
        """
        if epoch % self.i_image == 0:
            self.writer.add_images('Depths', depths.unsqueeze(3), epoch,
                                   dataformats='NHWC')

        if sfx != '':
            self.writer.add_images(f'Depth/{sfx}', depths.unsqueeze(3),
                                   epoch, dataformats='NHWC')
