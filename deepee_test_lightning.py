import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

from pl_bolts.datamodules import CIFAR10DataModule
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split
from torchvision import transforms

from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn

from torchmetrics.functional import accuracy

from deepee import (PrivacyWrapper, PrivacyWatchdog, UniformDataLoader,
                     ModelSurgeon, SurgicalProcedures)
from typing import Optional, Union, Any

seed_everything(7)

PATH_DATASETS = os.environ.get('PATH_DATASETS', '.')
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)

# DeePee: change dataloader to be able to create the privacy watchdog
class CIFAR10DataModule_DP(CIFAR10DataModule):
    def __init__(
        self, 
        data_dir, 
        batch_size, 
        num_workers, 
        train_transforms, 
        test_transforms, 
        val_transforms) -> None:
        super().__init__(
            data_dir=data_dir, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            train_transforms=train_transforms,
            test_transforms=test_transforms,
            val_transforms=val_transforms)

    # def __init__(self, data_dir, batch_size, num_workers, train_transforms, test_transforms, val_transforms):
    #     super().__init__()
    #     self.data_dir = data_dir
    #     self.batch_size = batch_size
    #     self.train_transforms = train_transforms
    #     self.test_transforms = test_transforms

    #     self.dims = (3, 32, 32)
    #     self.num_classes = 10

    # def prepare_data(self):
    #     # download
    #     CIFAR10(self.data_dir, train=True, download=True)
    #     CIFAR10(self.data_dir, train=False, download=True)

    # def setup(self, stage=None):

    #     # Assign train/val datasets for use in dataloaders
    #     if stage == 'fit' or stage is None:
    #         cifar_full = CIFAR10(self.data_dir, train=True, transform=self.train_transforms)
    #         self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

    #     # Assign test dataset for use in dataloader(s)
    #     if stage == 'test' or stage is None:
    #         self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.test_transforms)

    def train_dataloader(self):
        return UniformDataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return UniformDataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return UniformDataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)

# NOTE DeePee: train transforms?
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
    data_dir=PATH_DATASETS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms,
)

# DeePee: create privacy watchdog
cifar10_dm.setup()
watchdog = PrivacyWatchdog(
    cifar10_dm.train_dataloader(),
    target_epsilon=1.0,
    abort=False,
    target_delta=1e-5,
    fallback_to_rdp=False,
)

def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

# From DeePee tutorial
class args:
    batch_size = 200
    test_batch_size = 200
    log_interval = 1000
    num_epochs = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
device = args.device

class LitResnet(LightningModule):

    def __init__(self, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model()

        # DeePee: model init.
        # BatchNorm to GroupNorm
        surgeon = ModelSurgeon(SurgicalProcedures.BN_to_GN)
        self.model = surgeon.operate(self.model) 
        self.model = PrivacyWrapper(self.model, BATCH_SIZE, 1.0, 1.0, watchdog=watchdog)

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

        # DeePee: adding the DP procedure
        opt = self.optimizers()
        opt.zero_grad()
        # manual_backward automatically applies scaling, etc.
        self.manual_backward(loss)
        self.model.clip_and_accumulate()
        self.model.noise_gradient()
        opt.step()
        self.model.prepare_next_batch()

        return loss

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
        steps_per_epoch = 45000 // BATCH_SIZE
        scheduler_dict = {
            'scheduler': OneCycleLR(optimizer, 0.1, epochs=self.trainer.max_epochs, steps_per_epoch=steps_per_epoch),
            'interval': 'step',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}

model = LitResnet(lr=0.05)
model.datamodule = cifar10_dm

trainer = Trainer(
    progress_bar_refresh_rate=10,
    max_epochs=30,
    gpus=AVAIL_GPUS,
    logger=TensorBoardLogger('lightning_logs/', name='resnet'),
    callbacks=[LearningRateMonitor(logging_interval='step')],
)

trainer.fit(model, cifar10_dm)
trainer.test(model, datamodule=cifar10_dm)

# #################
# ## CLEAN FILES ##
# #################

# class args:
#     batch_size = 200
#     test_batch_size = 200
#     log_interval = 1000
#     num_epochs = 5
#     device = "cuda" if torch.cuda.is_available() else "cpu"
# device = args.device

# class SimpleNet(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(784, 256)
#         self.bn1 = nn.BatchNorm1d(256)
#         self.fc2 = nn.Linear(256, 64)
#         self.fc3 = nn.Linear(64, 10)

#     def forward(self, x):
#         x = torch.flatten(x, 1)
#         x = torch.sigmoid(self.fc1(x))
#         x = self.bn1(x)
#         x = torch.sigmoid(self.fc2(x))
#         x = self.fc3(x)
#         return x
    
#     def configure_optimizers(self):
#         # optimizer = torch.optim.SGD(model.wrapped_model.parameters(), lr=0.1)
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer

#     def training_step(self, batch, batch_idx):
#         # for epoch in range(args.num_epochs):
#         #     model.train()
#         #     for batch_idx, (data, target) in enumerate(train_loader):
#         #         data, target = data.to(device), target.to(device)
#         #         optimizer.zero_grad()
#         #         output = model(data)
#         #         loss = torch.nn.CrossEntropyLoss()(output, target)
#         #         loss.backward()
#         #         model.clip_and_accumulate()
#         #         model.noise_gradient()
#         #         optimizer.step()
#         #         model.prepare_next_batch()
#         #         if batch_idx % args.log_interval == 0:
#         #             print(
#         #                 "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
#         #                     epoch,
#         #                     batch_idx * len(data),
#         #                     len(train_loader.dataset),
#         #                     100.0 * batch_idx / len(train_loader),
#         #                     loss.item(),
#         #                 )
#         #             )
#         x, y = batch
#         y_hat = self(x)
#         loss = F.cross_entropy(y_hat, y)
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         # model.eval()
#         # test_loss = 0
#         # correct = 0
#         # with torch.no_grad():
#         #     for data, target in test_loader:
#         #         data, target = data.to(device), target.to(device)
#         #         output = model(data)
#         #         test_loss += torch.nn.CrossEntropyLoss(reduction="sum")(
#         #             output, target
#         #         ).item()  # sum up batch loss
#         #         pred = output.argmax(
#         #             dim=1, keepdim=True
#         #         )  # get the index of the max log-probability
#         #         correct += pred.eq(target.view_as(pred)).sum().item()

#         # test_loss /= len(test_loader.dataset)

#         # print(
#         #     "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
#         #         test_loss,
#         #         correct,
#         #         len(test_loader.dataset),
#         #         100.0 * correct / len(test_loader.dataset),
#         #     )
#         # )

#         x, y = batch
#         y_hat = self(x)
#         val_loss = F.cross_entropy(y_hat, y)
#         return val_loss



# class MNISTDataModule(pl.LightningDataModule): 

#     def prepare_data(self): 
#         pass

#     def train_dataloader(self):
#         pass

#     def val_dataloader(self): 
#         pass

#     def test_dataloader(self):
#         pass

# train_loader = UniformDataLoader(
#     datasets.MNIST(
#         "./data",
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.1307,), (0.3081,)),
#             ]
#         ),
#     ),
#     batch_size=args.batch_size,
# )
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "./data",
#         train=False,
#         transform=transforms.Compose(
#             [
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.1307,), (0.3081,)),
#             ]
#         ),
#     ),
#     batch_size=args.test_batch_size,
#     shuffle=True,
# ) 

# watchdog = PrivacyWatchdog(
#     train_loader,
#     target_epsilon=1.0,
#     abort=False,
#     target_delta=1e-5,
#     fallback_to_rdp=False,
# )

# # BatchNorm to GroupNorm
# surgeon = ModelSurgeon(SurgicalProcedures.BN_to_GN)
# model = surgeon.operate(SimpleNet()) 

# model = PrivacyWrapper(model, args.batch_size, 1.0, 1.0, watchdog=watchdog).to(
#     args.device
# )
# optimizer = torch.optim.SGD(model.wrapped_model.parameters(), lr=0.1)


# # Train
# for epoch in range(args.num_epochs):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = torch.nn.CrossEntropyLoss()(output, target)
#         loss.backward()
#         model.clip_and_accumulate()
#         model.noise_gradient()
#         optimizer.step()
#         model.prepare_next_batch()
#         if batch_idx % args.log_interval == 0:
#             print(
#                 "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
#                     epoch,
#                     batch_idx * len(data),
#                     len(train_loader.dataset),
#                     100.0 * batch_idx / len(train_loader),
#                     loss.item(),
#                 )
#             )

#     # Test
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += torch.nn.CrossEntropyLoss(reduction="sum")(
#                 output, target
#             ).item()  # sum up batch loss
#             pred = output.argmax(
#                 dim=1, keepdim=True
#             )  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)

#     print(
#         "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
#             test_loss,
#             correct,
#             len(test_loader.dataset),
#             100.0 * correct / len(test_loader.dataset),
#         )
#     )