# general python
from mmap import MAP_EXECUTABLE
import os
import yaml
from argparse import Namespace
from typing import Callable, ClassVar, Union, Any, Dict, Type, List
import numpy as np

# general ml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
import pytorch_lightning as pl
from scipy import stats
import pandas as pd

# pytorch lightning
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.utilities.cli import LightningCLI, LightningArgumentParser
from pytorch_lightning.callbacks import Callback
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.utilities.types import STEP_OUTPUT

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
from utils import get_grad_norm

# differential privacy
from deepee import (PrivacyWrapper, PrivacyWatchdog, UniformDataLoader,
                     ModelSurgeon, SurgicalProcedures, watchdog)

from opacus import PrivacyEngine, privacy_engine
from opacus.dp_model_inspector import DPModelInspector
from opacus.utils import module_modification

# interpretability 
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)

# visualization
import matplotlib
import matplotlib.pyplot as plt

### MODEL ###
class LitModelDP(LightningModule):
    def __init__(
        self, 
        model_name: str = "resnet18",
        kernel_size: int = 3,
        conv_layers: List[int] = [64, 32, 16],
        model_surgeon: ModelSurgeon = None,
        data_name: str = "cifar10",
        datamodule_class: Type[LightningDataModule] = None,
        pretrained: bool = False,
        num_classes: int = 10,
        optimizer: str = "sgd",
        opt_kwargs: dict = {"lr":0.05}, 
        lr_scheduler: bool = False,
        batch_size: int = 32,
        virtual_batch_size: int = 32,
        dp: bool = False, 
        dp_tool: str = "opacus",
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
            ## For SimpleConvNet ##
            kernel_size: kernel size for conv layers
            n_conv_layers: array consisting of numbers of feature maps
            ##
            model_surgeon: passed in deepee.ModelSurgeon to make model compatible with DP [deepee]
            data_name: also pass in what data to train on to possibly adapt model 
                       and to also save data as part of hparams for wandb.
            pretrained: pertrained or not
            num_classes: number of classes
            optimizer: pass in the optimizer to be used
            lr: init learning rate for optimizer
            lr_scheduler: whether to use a lr_scheduler or not
            batch_size: set through data subclass in configs
            virtual_batch_size: has to be divisible by batch_size. [opacus]
            dp_tool: whether to use 'opacus' or 'deepee' as DP tool
            dp: whether to train with dp or not
            L2_clip, noise_multiplier,target_epsilon, target_delta: [deepee, opacus]
            abort, callback_to_rdp: [deepee]
        """
        super().__init__()

        # save all arguments to this model to self.hparams (also saved in logs)
        self.save_hyperparameters()
       
        # select standard torchvision model
        self.model = get_model(
            model_name, 
            pretrained, 
            num_classes, 
            data_name, 
            kernel_size,
            conv_layers
        )
        # TODO: only temp., change later
        # operate - alter the model to be DP compatible if needed
        if dp and model_name == "resnet18":
            if dp_tool == "deepee": 
                self.model = model_surgeon.operate(self.model)
            elif dp_tool == "opacus": 
                self.model = module_modification.convert_batchnorm_modules(self.model)

        # disable auto. backward to be able to add noise and track 
        # the global grad norm (also in the non-dp case, lightning 
        # only does per param grad tracking)
        self.automatic_optimization = False

        # Opacus: init attribute 'privacy_engine'
        if dp and dp_tool == "opacus": 
            self.privacy_engine = None

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = F.log_softmax(self.model(x), dim=1)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)

        # also do non-dp backward manually to track global grad
        if not self.hparams.dp:
            opt = self.optimizers()
            opt.zero_grad()

            self.manual_backward(loss)
            # log global grad norm 
            self.log('global grad norm', get_grad_norm(
                    self.model.wrapped_model
                    if self.hparams.dp and self.hparams.dp_tool=='deepee'
                    else self.model, 
                )
            )
            opt.step()
        else:
            # DeePee: adding the DP procedure
            opt = self.optimizers()
            opt.zero_grad()
            # manual_backward automatically applies scaling, etc.
            self.manual_backward(loss)
            # log global grad norm 
            self.log('global grad norm', get_grad_norm(
                    self.model.wrapped_model
                    if self.hparams.dp and self.hparams.dp_tool=='deepee'
                    else self.model, 
                )
            )

            if self.hparams.dp_tool == "opacus":
                # NOTE: virtual steps can be taken to do calculations with less memory consumption. 
                # Hence allowing a bigger batch_size. To do twice as big batch_sizes with only a marginal 
                # more memory consuption do an opt.step() and an opt.virtual_step() alternating. 
                # opt.step() does a real step and opt.virtual_step() only aggregates gradients which are
                # then aggregated to the next opt.step(). Use self.hparams.n_accumulation_steps
                opt.step()
                # TODO: do smt with best_alpha
                spent_epsilon, best_alpha = opt.privacy_engine.get_privacy_spent(
                    self.hparams.target_delta,
                )
                self.log('spent_epsilon', spent_epsilon)
                # stop if epsilon too big 
                if spent_epsilon > self.hparams.target_epsilon: 
                    # tell trainer to stop -- copied from built-in EarlyStopping Callback (see docs)
                    # stop every ddp process if any world process decides to stop 
                    should_stop = self.trainer.training_type_plugin.reduce_boolean_decision(True)
                    self.trainer.should_stop = self.trainer.should_stop or should_stop

            elif self.hparams.dp_tool == "deepee":
                self.model.clip_and_accumulate()
                self.model.noise_gradient()
                opt.step()
                self.model.prepare_next_batch()
                # self.model.current_epsilon is a property of the class
                self.log('spent_epsilon', self.model.current_epsilon)

            # learning rate scheduler if configured so
            if self.hparams.lr_scheduler:
                self.lr_schedulers().step()

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
                self.model.wrapped_model.parameters() 
                if self.hparams.dp and self.hparams.dp_tool=='deepee'
                else self.model.parameters(), 
                **self.hparams.opt_kwargs,
            )
        elif self.hparams.optimizer=='adam':
            optimizer = torch.optim.Adam(
                self.model.wrapped_model.parameters() 
                if self.hparams.dp and self.hparams.dp_tool=='deepee'
                else self.model.parameters(), 
                **self.hparams.opt_kwargs,
            )

        if self.hparams.dp_tool=='opacus' and self.hparams.dp:
            self.privacy_engine.attach(optimizer)

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
    
    # NOTE self.datamodule.prepare_data() hook might be handy in the future

    def add_arguments_to_parser(self, parser):
        """Hook to run some code before arguments are parsed"""
        # that way the batch_size has to be only stated once in the data section 
        parser.link_arguments('model.batch_size', 'data.batch_size')
        parser.link_arguments('model.dp_tool', 'data.dp_tool')
        # parser.link_arguments('model.dp', 'data.dp')
        # TEMP: data config alternative
        # parser.link_arguments('datamodule_class', 'model.datamodule_class')

    def before_fit(self):
        """Hook to run some code before fit is started"""
        # possible because self.datamodule and self.model are instantiated beforehand
        # in LightningCLI.instantiate_trainer(self) -- see docs

        # TODO: why do I have to call them explictly here 
        #       -- in docs not mentioned (not found in .trainerfit())
        self.datamodule.prepare_data()
        self.datamodule.setup() 
        
        # TODO: why did i write this here????
        self.model.datamodule = self.datamodule

        if self.model.hparams.dp:
            if self.model.hparams.dp_tool == "deepee":
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
            elif self.model.hparams.dp_tool == "opacus":
                # NOTE: for now the adding to the optimizers is in model.configure_optimizers()
                # because at this point model.configure_optimizers() wasn't called yet. 
                # That's also why we save n_accumulation_steps as a model parameter.
                sample_rate = self.datamodule.batch_size/len(self.datamodule.dataset_train)
                if self.model.hparams.virtual_batch_size >= self.model.hparams.batch_size: 
                    self.model.n_accumulation_steps = int(
                        self.model.hparams.virtual_batch_size/self.model.hparams.batch_size
                    )
                else: 
                    self.model.n_accumulation_steps = 1 # neutral
                    print("Virtual batch size has to be bigger than real batch size!")

                # NOTE: For multiple GPU support: see PL code. 
                # For now we only consider shifting to cuda, if there's at least one GPU ('gpus' > 0)
                self.model.privacy_engine = PrivacyEngine(
                    self.model.model,
                    sample_rate=sample_rate * self.model.n_accumulation_steps,
                    target_delta = self.model.hparams.target_delta,
                    target_epsilon=self.model.hparams.target_epsilon,
                    # NOTE: either 'epochs' and 'target_epsilon' or 'noise_multiplier' can be specified.
                    # If noise_multiplier is not specified, (target_epsilon, target_delta, epochs) 
                    # is used to calculate it.
                    # epochs=self.trainer.max_epochs,
                    noise_multiplier=self.model.hparams.noise_multiplier,
                    max_grad_norm=self.model.hparams.L2_clip,
                ).to("cuda:0" if self.trainer.gpus else "cpu")
                print(f"Noise Multiplier: {self.model.privacy_engine.noise_multiplier}")

            else: 
                print("Use either 'opacus' or 'deepee' as DP tool.")

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
        
        # track gradients, etc.
        self.trainer.logger.experiment.watch(self.model)

    def after_fit(self) -> None:
        # TODO: detach PrivacyEngine from optimizer - necessary?
        #self.model.privacy_engine.detach()

        # TODO: this could also be done before the run 
        # TODO: made only for resnet18 at the moment
        # TODO: probably include as extra module, script where files are uploaded to 
        # the respective wandb run afterwards.
        if self.model.hparams.model_name == "resnet18" and False:
            # Check last layer of ResNet18 for now
            cond = LayerConductance(self.model.model, self.model.model.fc)
            # TODO: for now simply get first image in dataset
            #test_input_tensor, _ = self.datamodule.dataset_train[0]
            test_input_tensor = torch.rand([64, 1, 3, 3])
            cond_vals = cond.attribute(test_input_tensor,target=1)
            cond_vals = cond_vals.detach().numpy()
            visualize_importances(
                range(10),
                np.mean(cond_vals, axis=0),
                title="Average Neuron Importances", 
                axis_title="Neurons"
            )
    
class OpacusEpsilonEarlyStopping(Callback): 
   
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the train batch ends."""
        current_epsilon = trainer.model.privacy_engine.get_privacy_spent()
        if current_epsilon > self.model.privacy_engine.target_epsilon: 
            # tell trainer to stop -- copied from built-in EarlyStopping Callback
            # stop every ddp process if any world process decides to stop 
            should_stop = trainer.training_type_plugin.reduce_boolean_decision(True)
            trainer.should_stop = trainer.should_stop or should_stop
            if should_stop:
                self.stopped_epoch = trainer.current_epoch

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

### INTERPRETABILITY ###
# TODO: change this function to create a plot on wandb (should be possible natively)
# NOTE: see work in after_fit() function
# from: https://captum.ai/tutorials/Titanic_Basic_Interpret
# Helper method to print importances and visualize distribution
def visualize_importances(
    feature_names, 
    importances, 
    title="Average Feature Importances", 
    plot=True, 
    axis_title="Features"
):
    print(title)
    for i in range(len(feature_names)):
        print(feature_names[i], ": ", '%.3f'%(importances[i]))
    x_pos = (np.arange(len(feature_names)))
    if plot:
        plt.figure(figsize=(12,6))
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, feature_names, wrap=True)
        plt.xlabel(axis_title)
        plt.title(title)

cli = LightningCLI_Custom(
    model_class=LitModelDP, 
    # TODO: think about making model selection and data selection more consistent
    # --> try to use LitDataModuleDP as wrapper class
    datamodule_class=CIFAR10DataModuleDP, 
    save_config_callback=SaveConfigCallbackWandB,
)

# TODO: cli object can be manipulated (store artifacts to wandb, etc.) + embed in sweep