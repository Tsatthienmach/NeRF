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
        ave_dir (str): saving direction
        exp_name (str): experiment name
        sfx (str): suffix of the experiment
        i_save (int): save checkpoint every i_save epoch

    """

    def __init__(self, exp_name, save_dir='logs', sfx='', i_save=5):
        sfx = f'_{sfx}' if sfx else ''
        self.save_dir = os.path.join(save_dir, f'{exp_name}{sfx}')
        self.i_save = i_save
        os.makedirs(self.save_dir, exist_ok=True)

    def save(self, models, optimizer, lr_scheduler, best_psnr, epoch, sfx=''):
        """Saving function.

        Args:
            models: dictionary that contains models
            optimizer: training optimizer
            lr_scheduler: learning rate scheduler
            best_psnr: the best PSNR achievement
            epoch: current epoch
            sfx (str): more comment on saving file
        """
        if epoch % self.i_save == 0:
            torch.save(
                self.to_dict(
                    models, optimizer, lr_scheduler, best_psnr, epoch
                ),
                f'{self.save_dir}/checkpoint.pth'
            )

        sfx = f'_{sfx}' if sfx else ''
        if epoch == -1 and sfx != "":
            torch.save(
                self.to_dict(
                    models, optimizer, lr_scheduler, best_psnr, epoch
                ),
                f'{self.save_dir}/checkpoint{sfx}.pth'
            )

    @staticmethod
    def to_dict(models, optimizer, lr_scheduler, best_psnr, epoch):
        saved_dict = {
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'best_psnr': best_psnr
        }
        for model_name, model in models.items():
            saved_dict[model_name] = model.state_dict()

        return saved_dict

    @staticmethod
    def load(path, models, optimizer, lr_scheduler):
        loaded_dict = torch.load(path)
        optimizer = optimizer.load_state_dict(loaded_dict.pop('optimizer'))
        lr_scheduler = lr_scheduler.load_state_dict(
            loaded_dict.pop('lr_scheduler')
        )
        epoch = loaded_dict.pop('epoch')
        best_psnr = loaded_dict.pop('best_psnr')
        for k, v in loaded_dict.items():
            models[k] = v

        return models, optimizer, lr_scheduler, epoch, best_psnr
