import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from .utils.rendering import render_rays


class Trainer:
    """TODO
    Tasks:
        - train a epoch
        - validate
        - test after n epochs

    Args:
        train_set (DataLoader): training set
        val_set (DataLoader): validation set
        test_set (DataLoader): testing set
        embedders (dict): dictionary that contains position and direction
            embedders
        models (dict): dictionary that contains trained models
        loss: loss module
        metrics (dict): dictionary that contains metrics
        optimizer: training optimizer
        lr_scheduler: learning rate scheduler
        write: writer module for logging training information
        model_ckpt: model checkpoint module for saving/loading model
        load_weight (bool): If True, load pretrained checkpoints
        chunk (int): TODO
        epochs (int): number of training loops
        device: training device
    """

    def __init__(self,
                 train_set,
                 val_set,
                 test_set,
                 embedders,
                 models,
                 loss,
                 metrics,
                 optimizer,
                 lr_scheduler,
                 writer,
                 device,
                 model_ckpt=None,
                 N_samples=64,
                 N_importance=64,
                 chunk=1024 * 32,
                 epochs=100,
                 perturb=1.0,
                 noise_std=1.0,
                 white_bg=False,
                 use_disp=False,
                 load_weight=False):
        self.embedders = embedders
        self.models = models
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.writer = writer
        self.model_ckpt = model_ckpt
        self.load_weight = load_weight
        self.chunk = chunk
        self.epochs = epochs
        self._current_epoch = 0
        self.N_samples = N_samples
        self.N_importance = N_importance
        self.perturb = perturb
        self.noise_std = noise_std
        self.white_bg = white_bg
        self.use_disp = use_disp
        self.device = device
        self.best_psnr = 0.
        # Init trainer

    def forward(self, rays, val_tqdm=None, test_mode=False):
        bs = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, bs, self.chunk):
            chunk_ray = rays[i:i + self.chunk]
            rendered_ray_chunks = render_rays(
                self.models, self.embedders, chunk_ray, self.N_samples,
                self.use_disp, self.perturb, self.noise_std, self.N_importance,
                self.chunk, self.white_bg, test_mode=test_mode
            )
            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

            if val_tqdm:
                val_tqdm.set_description(f'Rays: {i + self.chunk}/{bs}')

        for k, v in results.items():
            results[k] = torch.cat(v, dim=0)

        return results

    def train_one_epoch(self, epoch):
        psnr_metric = self.metrics['psnr']
        psnr_metric.reset()
        data_tqdm = tqdm(self.train_set)
        self.train()
        for b_idx, batch in enumerate(data_tqdm):
            rays, rgbs = self.decode_batch(batch)
            rays = rays.to(self.device)
            rgbs = rgbs.to(self.device)
            self.optimizer.zero_grad()
            results = self.forward(rays)
            losses = self.loss(results, rgbs)
            loss = losses['total']
            loss.backward()
            self.optimizer.step()
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            data_tqdm.set_description(
                f"Loss: C: {losses['coarse']:.3f}" +
                f"| F: {losses['fine']:.3f}" +
                f"| {loss:.3f}"
            )
            with torch.no_grad():
                psnr_metric.update(
                    pred=results[f'rgb_{typ}'].cpu(),
                    gt=rgbs.cpu()
                )

            if b_idx > 1:
                break

        psnr = psnr_metric.compute()
        self.writer.save_loss(np.mean(psnr_metric.mses), epoch, pfx='train')
        self.writer.save_metrics({'psnr': psnr}, epoch, pfx='train')
        self.model_ckpt.save(self.models, self.optimizer, self.lr_scheduler,
                             psnr, epoch, sfx='')
        if psnr > self.best_psnr:
            self.best_psnr = psnr
            self.model_ckpt.save(self.models, self.optimizer,
                                 self.lr_scheduler, psnr, epoch=-1,
                                 sfx='best_psnr')

    def validate(self, epoch):
        self.eval()
        for metric_name, metric in self.metrics.items():
            vars()[f'{metric_name}_metric'] = metric
            vars()[f'{metric_name}_metric'].reset()

        data_tqdm = tqdm(self.val_set)
        for b_idx, batch in enumerate(data_tqdm):
            b_rays, b_rgbs = self.decode_batch(batch)
            pred_rgbs = []
            for i, rays in enumerate(b_rays):
                with torch.no_grad():
                    results = self.forward(rays.to(self.device),
                                           val_tqdm=data_tqdm)

                typ = 'fine' if 'rgb_fine' in results else 'coarse'
                img = results[f'rgb_{typ}'].view(b_rgbs[i].shape).cpu()
                pred_rgbs.append(img)
                for metric_name in self.metrics.keys():
                    vars()[f'{metric_name}_metric'].update(img, b_rgbs[i])

            metric_results = {}
            for metric_name in self.metrics.keys():
                metric_results[metric_name] = \
                    vars()[f'{metric_name}_metric'].compute()

            self.writer.save_metrics(metric_results, epoch, pfx='val')
            self.writer.save_loss(np.mean(vars()['psnr_metric'].mses),
                                  epoch, pfx='val')
            self.writer.save_imgs(torch.stack(pred_rgbs, dim=0),
                                  b_rgbs.cpu(),
                                  epoch, data_format='NHWC')

    def train(self):
        for _, model in self.models.items():
            model.train()

    def eval(self):
        for _, model in self.models.items():
            model.eval()

    def test(self):
        pass

    def fit(self):
        for e in range(self._current_epoch, self.epochs):
            print(f'--------------- {e} ---------------')
            self.train_one_epoch(e)
            self.validate(e)

    def load_weight(self):
        self._current_epoch = 0  # TODO

    @staticmethod
    def decode_batch(batch):
        rays = batch['rays']
        rgbs = batch['rgbs']
        return rays, rgbs
