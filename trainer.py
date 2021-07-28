# general python
from mmap import MAP_EXECUTABLE
import os
import yaml
from typing import List

# general ml
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# pytorch lightning
from pytorch_lightning import LightningModule, seed_everything, Trainer
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from pytorch_lightning.utilities.cli import LightningCLI

# data
from data import LitDataModuleDP, CIFAR10DataModule, CIFAR10DataModuleDP

# model 
from models import get_model

# metrics & logging
from pytorch_lightning.loggers import WandbLogger, wandb
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint, model_checkpoint

# differential privacy
from deepee import (PrivacyWrapper, PrivacyWatchdog, UniformDataLoader,
                     ModelSurgeon, SurgicalProcedures, watchdog)

class LitModelDP(LightningModule):
    def __init__(
        self, 
        name: str = "resnet18",
        pretrained: bool = False,
        num_classes: int = 10,
        lr: float = 0.05, 
        batch_size: int = 32,
        dp: bool = False, 
        L2_clip: float = 1.0,
        noise_multiplier: float = 1.0,
        target_epsilon: float = 10.0,
        abort: bool = False,
        target_delta: float = 1e-5,
        fallback_to_rdp: bool = False
    ):
        """
        Wrapper Module to extend normal models defined in models.py to DP

        Args:
            name: selection of the model
            pretrained: pertrained or not
            num_classes: number of classes
            lr: init learning rate for optimizer
            batch_size: set through data subclass in configs
            dp: whether to train with dp or not
            L2_clip, noise_multiplier: for deepee.PrivacyWrapper
            target_epsilon, abort
            target_delta, callback_to_rdp: for deepee.PrivacyWatchdog
        """
        super().__init__()

        # save all arguments to this model to self.hparams (also saved in logs)
        self.save_hyperparameters()
       
       # TODO: can be flexibly plugged in in models.
        # select standard torchvision model
        self.model = get_model(name, pretrained, num_classes)
       
        # DeePee: model init.
        if self.hparams.dp:
            # create deepee watchdog to be used in model instantiation
            # self.watchdog = PrivacyWatchdog(
            #     # possible because before trainer.fit() calls self._run(model) 
            #     # the datamodule is attached
            #     self.trainer.datamodule.train_dataloader(),
            #     target_epsilon=10.0,
            #     abort=False,
            #     target_delta=1e-5,
            #     fallback_to_rdp=False,
            # )

            # # BatchNorm to GroupNorm
            # surgeon = ModelSurgeon(SurgicalProcedures.BN_to_BN_nostats)
            # self.model = surgeon.operate(self.model) 
            # self.model = PrivacyWrapper(
            #     self.model, 
            #     batch_size=self.hparams.batch_size, 
            #     L2_clip=1.0, 
            #     noise_multipler=1.0, 
            #     # we define self.watchdog in LightningCLI.before_fit()
            #     watchdog=self.watchdog,
            # )

            # DeePee: disable automatic optimization, to be able to add noise
            self.automatic_optimization = False

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = F.log_softmax(self.model(x), dim=1)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)

        if self.hparams.dp:
            # DeePee: adding the DP procedure
            opt = self.optimizers()
            sch = self.lr_schedulers()
            opt.zero_grad()
            # manual_backward automatically applies scaling, etc.
            self.manual_backward(loss)
            self.model.clip_and_accumulate()
            self.model.noise_gradient()
            self.log('spend_epsilon', self.model.current_epsilon)

            opt.step()
            sch.step()
            self.model.prepare_next_batch()

        return loss

    # utility function (no pl-hook)
    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')

    def configure_optimizers(self):
        # DeePee: we want params from wrapped mdoel
        # self.paramters() -> self.model.wrapped_model.parameters()
        # NOTE: check if self.wrapped_model.parameters() would also work
        optimizer = torch.optim.SGD(
            self.model.wrapped_model.parameters() if self.hparams.dp else self.model.parameters(), 
            lr=self.hparams.lr, 
            # momentum=0.9, 
            # weight_decay=5e-4
        )
        steps_per_epoch = 45000 // self.hparams.batch_size
        scheduler_dict = {
            'scheduler': OneCycleLR(optimizer, 0.1, epochs=self.trainer.max_epochs, steps_per_epoch=steps_per_epoch),
            'interval': 'step',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}


# custom LightningCLI to include some deepee function calls
class LightningCLI_Custom(LightningCLI):
    # override instantiation of the datamodule to create a deepee watchdog
    # because instantiate_datamodule() is called before instantiate_model() in instantiate_classes()
    # see: https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/utilities/cli.html
    # def instantiate_datamodule(self) -> None:
    #     """Instantiates the datamodule using self.config_init['data'] if given"""
    #     if self.datamodule_class is None:
    #         self.datamodule = None
    #     elif self.subclass_mode_data:
    #         self.datamodule = self.config_init['data']
    #     else:
    #         self.datamodule = self.datamodule_class(**self.config_init.get('data', {}))
        
    #     # create deepee watchdog to be used in model instantiation
    #     self.watchdog = PrivacyWatchdog(
    #         self.datamodule.train_dataloader(),
    #         target_epsilon=10.0,
    #         abort=False,
    #         target_delta=1e-5,
    #         fallback_to_rdp=False,
    #     )

    def add_arguments_to_parser(self, parser):
        """Hook to run some code before arguments are parsed"""
        # that way the batch_size has to be only stated once in the data section 
        parser.link_arguments('data.batch_size', 'model.batch_size')
        # parser.link_arguments('model.dp', 'data.dp')

    def before_fit(self):
        """Hook to run some code before fit is started"""
        # possible because self.datamodule and self.model are instantiated beforehand
        # in LightningCLI.instantiate_trainer(self) -- see docs
        
        # NOTE self.datamodule.prepare_data() hook might be handy in the future
        # TODO: why do I have to call them explictly here -- in docs not mentioned (not found in .trainerfit())
        self.datamodule.prepare_data()
        self.datamodule.setup() 
        self.model.datamodule = self.datamodule

        if self.model.hparams.dp:
            watchdog = PrivacyWatchdog(
                    self.datamodule.train_dataloader(),
                    target_epsilon=self.model.hparams.target_epsilon,
                    abort=self.model.hparams.abort,
                    target_delta=self.model.hparams.target_delta,
                    fallback_to_rdp=self.model.hparams.fallback_to_rdp,
            )

            # TODO: surgical procedures have to be done automatically depending on model
            # we alter the model to make it DP-compatible - we call self.model.model 
            # because our LitModel defines the model itself as an attr self.model. 
            # This way we can wrap the actual model in the PrivacyWraper without loosing 
            # compatibility with the trainer.fit() (that calls some functions on the model).
            surgeon = ModelSurgeon(SurgicalProcedures.BN_to_BN_nostats)
            self.model.model = surgeon.operate(self.model.model) 
            self.model.model = PrivacyWrapper(
                base_model=self.model.model, 
                num_replicas=self.model.hparams.batch_size,
                L2_clip=self.model.hparams.L2_clip, 
                noise_multiplier=self.model.hparams.noise_multiplier, 
                watchdog=watchdog,
            )

            # self.fit_kwargs is passed to self.trainer.fit() in LightningCLI.fit(self)
            self.fit_kwargs.update({
                'model': self.model
            })

cli = LightningCLI_Custom(LitModelDP, CIFAR10DataModuleDP)
