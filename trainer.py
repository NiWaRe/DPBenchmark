# general python
from mmap import MAP_EXECUTABLE
import os
import yaml
from argparse import Namespace
from typing import Union, Any, Dict

# general ml
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# pytorch lightning
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.utilities.cli import LightningCLI, LightningArgumentParser
from pytorch_lightning.callbacks import Callback
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn

# data
from data import LitDataModuleDP, CIFAR10DataModule, CIFAR10DataModuleDP

# model 
from models import get_model

# metrics & logging
from wandb import Artifact
from pytorch_lightning.loggers import WandbLogger, wandb
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint, model_checkpoint

# differential privacy
from deepee import (PrivacyWrapper, PrivacyWatchdog, UniformDataLoader,
                     ModelSurgeon, SurgicalProcedures, watchdog)

### MODEL ###
class LitModelDP(LightningModule):
    def __init__(
        self, 
        model_name: str = "resnet18",
        data_name: str = "cifar10",
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
            model_name: selection of the model
            data_name: also pass in what data to train on to possibly adapt model 
                       and to also save data as part of hparams for wandb.
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
       
        # select standard torchvision model
        self.model = get_model(model_name, pretrained, num_classes)
       
        # DeePee: model init.
        if self.hparams.dp:
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

### ARG PARSING ###
# custom LightningCLI to include some deepee function calls
class LightningCLI_Custom(LightningCLI):
    
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

            # in addition to the params saved through the model, save some others from trainer
            important_keys_trainer = ['gpus', 'max_epochs', 'deterministic']
            self.trainer.logger.experiment.config.update(
                {
                    important_key:self.config['trainer'][important_key] 
                    for important_key in important_keys_trainer
                }
            )
            # the rest is stored as part of the SaveConfigCallbackWandB
            # (too big to store every metric as part of the above config)
        
            # TODO: does this work?
            self.trainer.logger.experiment.watch(self.model)
    
# TODO: better option: directly include as meta data to saved, trained model-weights
# rather overwrite: pytorch_lightning.callbacks.ModelCheckpoint
# if done --> pass in 'None' for Trainer.save_config_callback 
# https://docs.wandb.ai/guides/artifacts/api
# custom save_config_callback to log all configs as artifact to wandb
class SaveConfigCallbackWandB(Callback):
    """Saves a LightningCLI config to the log_dir when training starts"""

    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Union[Namespace, Dict[str, Any]],
        # NOTE: wandb stores its config file at the same path 'config.yaml'
        config_filename: str = 'config_pl_cli.yaml'
    ) -> None:
        self.parser = parser
        self.config = config
        self.config_filename = config_filename

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        log_dir = trainer.log_dir or trainer.default_root_dir
        config_path = os.path.join(log_dir, self.config_filename)
        self.parser.save(self.config, config_path, skip_none=False)
        # save config file as artifact
        artifact = Artifact('complete_config', type='params')
        artifact.add_file(config_path)
        trainer.logger.experiment.log_artifact(artifact)

### TRAINER ###
# for now standard class "Trainer" is used (by default)

cli = LightningCLI_Custom(
    model_class=LitModelDP, 
    datamodule_class=CIFAR10DataModuleDP, 
    save_config_callback=SaveConfigCallbackWandB,
)

# TODO: cli object can be manipulated (store artifacts to wandb, etc.) + embed in sweep
