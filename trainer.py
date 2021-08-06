# general python
from mmap import MAP_EXECUTABLE
import os
import deepee
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.optim import lr_scheduler
import yaml
from argparse import Namespace
from typing import Callable, ClassVar, Union, Any, Dict, Type

# general ml
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Optimizer

# pytorch lightning
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.utilities.cli import LightningCLI, LightningArgumentParser
from pytorch_lightning.callbacks import Callback
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from pytorch_lightning.core.datamodule import LightningDataModule

# data
from data import LitDataModuleDP, CIFAR10DataModule, CIFAR10DataModuleDP, MNISTDataModuleDP

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
# TODO: model_surgeon, optimizer, configs checken
class LitModelDP(LightningModule):
    def __init__(
        self, 
        model_name: str = "resnet18",
        model_surgeon: ModelSurgeon = None,
        data_name: str = "cifar10",
        datamodule_class: Type[LightningDataModule] = None,
        pretrained: bool = False,
        num_classes: int = 10,
        optimizer: str = "sgd",
        opt_kwargs: str = '{"lr":0.05}', 
        lr_scheduler: bool = False,
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
            model_surgeon: passed in deepee.ModelSurgeon to make model compatible with DP
            data_name: also pass in what data to train on to possibly adapt model 
                       and to also save data as part of hparams for wandb.
            pretrained: pertrained or not
            num_classes: number of classes
            optimizer: pass in the optimizer to be used
            lr: init learning rate for optimizer
            lr_scheduler: whether to use a lr_scheduler or not
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
        self.model = get_model(model_name, pretrained, num_classes, data_name)
        # operate - alter the model to be DP compatible if needed
        if model_surgeon: 
            self.model = model_surgeon.operate(self.model)

        # DeePee: model init.
        if dp:
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
            opt.zero_grad()
            # manual_backward automatically applies scaling, etc.
            self.manual_backward(loss)
            self.model.clip_and_accumulate()
            self.model.noise_gradient()
            opt.step()
            self.log('spend_epsilon', self.model.current_epsilon)

            # learning rate scheduler if configured so
            if self.hparams.lr_scheduler:
                self.lr_schedulers().step()

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
        optims = {}
        # DeePee: we want params from wrapped mdoel
        # self.paramters() -> self.model.wrapped_model.parameters()
        if self.hparams.optimizer=='sgd':
            optimizer = torch.optim.SGD(
                self.model.wrapped_model.parameters() if self.hparams.dp else self.model.parameters(), 
                # TODO: change 
                lr=0.05,
            )
        elif self.hparams.optimizer=='adam':
            optimizer = torch.optim.Adam(
                self.model.wrapped_model.parameters() if self.hparams.dp else self.model.parameters(), 
                # TODO: change
                lr=0.05,
            )
        optims.update({'optimizer': optimizer})

        # learning rate scheduler
        if self.hparams.lr_scheduler:
            steps_per_epoch = 45000 // self.hparams.batch_size
            scheduler_dict = {
                'scheduler': OneCycleLR(optimizer, 0.1, epochs=self.trainer.max_epochs, steps_per_epoch=steps_per_epoch),
                'interval': 'step',
            }
            optims.update({'lr_scheduler': scheduler_dict})

        return optims

### ARG PARSING ###
# custom LightningCLI to include some deepee function calls
class LightningCLI_Custom(LightningCLI):
    
    def add_arguments_to_parser(self, parser):
        """Hook to run some code before arguments are parsed"""
        # that way the batch_size has to be only stated once in the data section 
        parser.link_arguments('data.batch_size', 'model.batch_size')
        # parser.link_arguments('model.dp', 'data.dp')
        # TEMP: data config alternative
        # parser.link_arguments('datamodule_class', 'model.datamodule_class')

    def before_fit(self):
        """Hook to run some code before fit is started"""
        # possible because self.datamodule and self.model are instantiated beforehand
        # in LightningCLI.instantiate_trainer(self) -- see docs
        
        # NOTE self.datamodule.prepare_data() hook might be handy in the future
        # TODO: why do I have to call them explictly here -- in docs not mentioned (not found in .trainerfit())
        self.datamodule.prepare_data()
        self.datamodule.setup() 
        
        # TODO: why did i write this here????
        # self.model.datamodule = self.datamodule

        if self.model.hparams.dp:
            watchdog = PrivacyWatchdog(
                self.datamodule.train_dataloader(),
                target_epsilon=self.model.hparams.target_epsilon,
                abort=self.model.hparams.abort,
                target_delta=self.model.hparams.target_delta,
                fallback_to_rdp=self.model.hparams.fallback_to_rdp,
            )

            # We call self.model.model because our LitModel defines the model itself as 
            # an attr self.model. This way we can wrap the actual model in the PrivacyWraper 
            # without loosing compatibility with the trainer.fit() 
            # (that calls some functions on the model).
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
        
            # TODO: track gradients, etc. --> exceeds maximum metric data size per step
            #self.trainer.logger.experiment.watch(self.model)
    
# TODO: better option: directly include as meta data to saved, trained model-weights
# rather overwrite: pytorch_lightning.callbacks.ModelCheckpoint
# if done --> pass in 'None' for Trainer.save_config_callback 
# https://docs.wandb.ai/guides/artifacts/api
# custom save_config_callback to log all configs as artifact to wandb
class SaveConfigCallbackWandB(Callback):
    """Saves a LightningCLI config and model architecture print 
       to the log_dir when training starts"""

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

    ## SAVE FILES AND STORE ARTIFACTS ##
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        log_dir = trainer.log_dir or trainer.default_root_dir
        unique_config_name = trainer.logger.experiment.name + "_" + self.config_filename
        config_path = os.path.join(
            log_dir, 
            unique_config_name,
        )
        # save the configs as a file at config_path
        self.parser.save(self.config, config_path, skip_none=False)
        # save config file as artifact
        c_artifact = Artifact(unique_config_name, type='params')
        c_artifact.add_file(config_path)
        trainer.logger.experiment.log_artifact(c_artifact)

        # save model architecture in a textfile
        unique_model_name = trainer.logger.experiment.name + '_model_architecture.txt'
        model_arch_path = os.path.join(
            log_dir, 
            unique_model_name,
        )
        with open(model_arch_path, 'w') as f: 
            f.write(trainer.model.model.__str__())
        # save model archicture as txt file
        m_artifact = Artifact(unique_model_name, type='params')
        m_artifact.add_file(model_arch_path)
        trainer.logger.experiment.log_artifact(m_artifact)

### TRAINER ###
# for now standard class "Trainer" is used (by default)

cli = LightningCLI_Custom(
    model_class=LitModelDP, 
    # TODO: think about making model selection and data selection more consistent
    # --> try to use LitDataModuleDP as wrapper class
    datamodule_class=CIFAR10DataModuleDP, 
    save_config_callback=SaveConfigCallbackWandB,
)

# TODO: cli object can be manipulated (store artifacts to wandb, etc.) + embed in sweep
