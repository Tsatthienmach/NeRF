import torch


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
                 model_ckpt,
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
        # Init trainer

    def train_one_epoch(self):
        pass

    def validate_one_epoch(self):
        pass

    def train(self):
        pass

    def validate(self):
        pass

    def test(self):
        pass
