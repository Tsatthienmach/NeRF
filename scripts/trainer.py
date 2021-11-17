import torch
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
        self.train_set = tqdm(train_set)
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
        # Init trainer

    def forward(self, rays):
        bs = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, bs, self.chunk):
            chunk_ray = rays[i:i+self.chunk]
            rendered_ray_chunks = render_rays(
                self.models, self.embedders, chunk_ray, self.N_samples,
                self.use_disp, self.perturb, self.noise_std, self.N_importance,
                self.chunk, self.white_bg, test_mode=False
            )
            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, dim=0)

        return results

    def train_one_epoch(self, epoch):
        self.metrics['psnr'].reset()
        for b_idx, batch in enumerate(self.train_set):
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
            with torch.no_grad():
                self.metrics['psnr'].update(
                    pred=results[f'rgb_{typ}'],
                    gt=rgbs
                )

        print(f'---> coarse: {losses["coarse"]} | fine: {losses["fine"]} | {loss}')
        print('---->', self.metrics['psnr'].compute())


    def validate_one_epoch(self):
        pass

    def test(self):
        pass

    def fit(self):
        for e in range(self._current_epoch, self.epochs):
            self.train_one_epoch(e)

    def load_weight(self):
        self._current_epoch = 0  # TODO

    @staticmethod
    def decode_batch(batch):
        rays = batch['rays']
        rgbs = batch['rgbs']
        return rays, rgbs
