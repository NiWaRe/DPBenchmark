# general python
from mmap import MAP_EXECUTABLE
import os
import yaml

# general ml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

# data 
from data import (
    FashionMNISTDataModule, 
    FashionMNISTDataModule_DP, 
    CIFAR10DataModule, 
    CIFAR10DataModule_DP, 
    ImagenetDataModule, 
    ImagenetDataModule_DP,
)
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

#

# model & training
from pytorch_lightning import LightningModule, seed_everything, Trainer
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn

# metrics & logging
from pytorch_lightning.loggers import WandbLogger, wandb
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint, model_checkpoint

# differential privacy
from deepee import (PrivacyWrapper, PrivacyWatchdog, UniformDataLoader,
                     ModelSurgeon, SurgicalProcedures)

# class args: 
#     PATH_DATASETS = os.environ.get('PATH_DATASETS', '.')
#     AVAIL_GPUS = min(1, torch.cuda.device_count())
#     BATCH_SIZE = 32 if AVAIL_GPUS else 64
#     NUM_WORKERS = int(os.cpu_count() / 2)
#     DP_ON = False
#     PRETRAINED = False

# dynamic configs
dynamic_configs = dict(
    avail_gpus=min(1, torch.cuda.device_count()),
    # BATCH_SIZE = 32 if AVAIL_GPUS else 64,
    num_workers = int(os.cpu_count() / 2),
    path_datasets = os.environ.get('PATH_DATASETS', '.')
)

# static and dynamic configs 
configs = dict(
    yaml='./configs-base.yaml', 
    params=dynamic_configs,
)

# for reproduceability
seed_everything(configs['seed'])



train_transforms = torchvision.transforms.Compose([
    # torchvision.transforms.RandomCrop(32, padding=4),
    # torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    cifar10_normalization(),
])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    cifar10_normalization(),
])

cifar10_dm = CIFAR10DataModule_DP(
    data_dir=configs['path_datasets'],
    batch_size=configs['batch_size'],
    num_workers=configs['num_workers'],
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms,
)

# DeePee: create privacy watchdog 
# call setup() explicitly, otherwise will only be called when training starts
cifar10_dm.setup()

if configs['dp_on']:
    watchdog = PrivacyWatchdog(
        cifar10_dm.train_dataloader(),
        target_epsilon=10.0,
        abort=False,
        target_delta=1e-5,
        fallback_to_rdp=False,
    )

def create_model():
    model = torchvision.models.resnet18(pretrained=configs['pretrained'], num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

class LitResnet(LightningModule):

    def __init__(self, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.log('hparams_initial', self.hparams_initial)
        #self.model = create_model()
        if configs['model'] == 'resnet18':
            self.model = torchvision.models.resnet18(pretrained=configs['pretrained'], num_classes=10)
        else: 
            print("*******\nModel doesn't exist\n********")
        #self.model = torchvision.models.vgg16(pretrained=False, num_classes=10)
        #self.model = torchvision.models.inception_v3(pretrained=False, num_classes=10)

        # DeePee: model init.
        if configs['dp_on']:
            # BatchNorm to GroupNorm
            surgeon = ModelSurgeon(SurgicalProcedures.BN_to_BN_nostats)
            self.model = surgeon.operate(self.model) 
            self.model = PrivacyWrapper(self.model, configs['batch_size'], 1.0, 1.0, watchdog=watchdog)

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

        if configs['dp_on']:
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
            self.model.wrapped_model.parameters(), 
            lr=self.hparams.lr, 
            # momentum=0.9, 
            # weight_decay=5e-4
        )
        self.log("hparams", self.hparams)
        steps_per_epoch = 45000 // configs['batch_size']
        scheduler_dict = {
            'scheduler': OneCycleLR(optimizer, 0.1, epochs=self.trainer.max_epochs, steps_per_epoch=steps_per_epoch),
            'interval': 'step',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}

model = LitResnet(lr=0.05)
model.datamodule = cifar10_dm

# create ModelCheckpoint callback which auto. saves the model checking some metric
# in this case the 'val_loss' metric which is also logged (see above)
mc = ModelCheckpoint(monitor='val_loss')

# configs = dynamic_configs
# with open('configs-base.yaml') as f: 
#     params = yaml.safe_load(f)
#     configs.update(params)

#logger=TensorBoardLogger('lightning_logs/', name='resnet')

# NOTE: all additional params are automatically passed to wandb.init(...) in WandbLogger.init()
# the wandb object can accessed with logger.experiment.function_to_call()
logger = WandbLogger(
    project='raw_performances', 
    name='cifar10_resnet18', 
    log_model=True,
    # **kwargs
    config=configs,
)

logger.watch(model)

trainer = Trainer(
    progress_bar_refresh_rate=10,
    max_epochs=30,
    gpus=configs['avail_gpus'],
    logger=logger,
    callbacks=[LearningRateMonitor(logging_interval='step'), mc],
)

# train and test
trainer.fit(model, cifar10_dm)
trainer.test(model, datamodule=cifar10_dm)