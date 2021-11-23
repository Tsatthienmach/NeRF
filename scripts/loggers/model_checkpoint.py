import os
import torch


class ModelCheckPoint:
    """Model checkpoint manipulator.
    Tasks:
        - save weights
        - save optimizer
        - save learning rate scheduler
        - save global epoch

    Args:
        save_dir (str): saving direction
        exp_name (str): experiment name
        sfx (str): suffix of the experiment
        i_save (int): save checkpoint every i_save epoch

    """

    def __init__(self, exp_name, save_dir='logs', sfx='', i_save=1):
        sfx = f'_{sfx}' if sfx else ''
        self.save_dir = os.path.join(save_dir, f'{exp_name}{sfx}')
        self.i_save = i_save
        os.makedirs(self.save_dir, exist_ok=True)

    def save(self, models, optimizer, lr_scheduler, best_psnr, epoch,
             test_info, sfx='', addition={}):
        """Saving function.

        Args:
            models: dictionary that contains models
            optimizer: training optimizer
            lr_scheduler: learning rate scheduler
            best_psnr: the best PSNR achievement
            epoch: current epoch
            sfx (str): more comment on saving file
            test_info (dict): Info that helps create inference pose
            addition (dict): save more information
        """
        if epoch % self.i_save == 0:
            torch.save(
                self.to_dict(
                    models, optimizer, lr_scheduler, best_psnr, test_info,
                    addition, epoch
                ), f'{self.save_dir}/checkpoint.pth'
            )

        sfx = f'_{sfx}' if sfx else ''
        if sfx != "":
            torch.save(
                self.to_dict(
                    models, optimizer, lr_scheduler, best_psnr, test_info,
                    addition, epoch
                ), f'{self.save_dir}/checkpoint{sfx}.pth'
            )

    @staticmethod
    def to_dict(models, optimizer, lr_scheduler, best_psnr, test_info,
                addition, epoch):
        saved_dict = {
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'test_info': test_info,
            'best_psnr': best_psnr,
            'models': [model.state_dict() for model in models],
            'addition': addition
        }
        return saved_dict
