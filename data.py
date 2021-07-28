# general python 
from typing import Optional, Union, Any, List

# data
from pl_bolts.datamodules.vision_datamodule import VisionDataModule

from typing import Any, Callable, Optional, Sequence, Union

from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_bolts.datasets import TrialCIFAR10
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
    from torchvision.datasets import (
        CIFAR10, 
        FashionMNIST, 
        MNIST, 
        ImageNet
    )
else:  # pragma: no cover
    warn_missing_pkg('torchvision')
    CIFAR10 = None
    FashionMNIST = None
    MNIST = None
    ImageNet = None

######

from pl_bolts.datamodules import (
    FashionMNISTDataModule, 
    CIFAR10DataModule, 
    ImagenetDataModule, 
    MNISTDataModule
)
from pytorch_lightning.callbacks import ModelCheckpoint, model_checkpoint
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

# differential privacy
from deepee import (PrivacyWrapper, PrivacyWatchdog, UniformDataLoader,
                     ModelSurgeon, SurgicalProcedures)

### Standard Lightning Bolt Data Modules ###

# TODO: The PrivacyWrapper also supports normal DataLoaders -- don't the DataModules have that type?

# DP Extended Base DataModule Wrapper
class LitDataModuleDP(MNISTDataModule, FashionMNISTDataModule, CIFAR10DataModule, ImagenetDataModule):
    """
    This is based on the preconfigured torchvision datamodules (e.g. CIFAR10DataModule)
    """
    def __init__(
        self,
        name: str = "cifar10",
        dp: bool = True,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 16,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            name: What torchvision dataset to select (mnist, fashion_mnist, cifar10 or imagenet)
            data_dir: Where to save/load the data
            val_split: Percent (float) or number (int) of samples to use for the validation split
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
            dp: Whether this should be compatible with DeePee or not
        """

        dataset_cls = None
        if name=="mnist":
            dataset_cls = MNISTDataModule
        elif name=="fashion_mnist": 
            dataset_cls = FashionMNISTDataModule
        elif name=="cifar10":
            dataset_cls = CIFAR10DataModule
        elif name=="imagenet": 
            dataset_cls = ImagenetDataModule

        dataset_cls.__init__(  # type: ignore[misc]
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            *args,
            **kwargs,
        )

        self.dp = dp
        self.name = name

    # DeePee: overload dataloaders to be able to create the privacy watchdog
    # TODO: if UniformDataLoader would accept *args, **kwargs 
    #       I could simply overload VisionDataModule._data_loader
    def train_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, UniformDataLoader]:
        if self.dp:
            return UniformDataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers)
        else: 
            # standard from class VisionDataModule
            return self._data_loader(self.dataset_train, shuffle=self.shuffle)
    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader], UniformDataLoader]:
        if self.dp:
            return UniformDataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)
        else: 
            # standard from class VisionDataModule
            return self._data_loader(self.dataset_val)
    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader], UniformDataLoader]:
        if self.dp:
            return UniformDataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)
        else: 
            # standard from class VisionDataModule
            return self._data_loader(self.dataset_test)


# DP Extended Base DataModule Wrapper
# class LitDataModuleDP(VisionDataModule):
#     """
#     This is based on the preconfigured torchvision datamodules (e.g. CIFAR10DataModule)
#     """
#     def __init__(
#         self,
#         name: str = "cifar10",
#         data_dir: Optional[str] = None,
#         val_split: Union[int, float] = 0.2,
#         num_workers: int = 16,
#         normalize: bool = False,
#         batch_size: int = 32,
#         seed: int = 42,
#         shuffle: bool = False,
#         pin_memory: bool = False,
#         drop_last: bool = False,
#         dp: bool = True,
#         *args: Any,
#         **kwargs: Any,
#     ) -> None:
#         """
#         Args:
#             name: What torchvision dataset to select (cifar10, fashion_mnist, imagenet or mnist)
#             data_dir: Where to save/load the data
#             val_split: Percent (float) or number (int) of samples to use for the validation split
#             num_workers: How many workers to use for loading data
#             normalize: If true applies image normalize
#             batch_size: How many samples per batch to load
#             seed: Random seed to be used for train/val/test splits
#             shuffle: If true shuffles the train data every epoch
#             pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
#                         returning them
#             drop_last: If true drops the last incomplete batch
#             dp: Whether this should be compatible with DeePee or not
#         """
#         super().__init__(  # type: ignore[misc]
#             name=name,
#             data_dir=data_dir,
#             val_split=val_split,
#             num_workers=num_workers,
#             normalize=normalize,
#             batch_size=batch_size,
#             seed=seed,
#             shuffle=shuffle,
#             pin_memory=pin_memory,
#             drop_last=drop_last,
#             dp=dp,
#             *args,
#             **kwargs,
#         )
#         if name=="cifar10":
#             self.dataset_cls = CIFAR10
#             self.dims = (3, 32, 32)
#         elif name=="fashion_mnist":
#             self.dataset_cls = FashionMNIST
#             self.dims = (1, 28, 28)
#         elif name=="imagenet":
#             pass
#         elif name=="mnist":
#             self.dataset_cls = MNIST
#             self.dims = (1, 28, 28)

#     @property
#     def num_samples(self) -> int:
#         train_len = None
#         if self.name=="cifar10":
#             train_len, _ = self._get_splits(len_dataset=50_000)
#         elif self.name=="fashion_mnist":
#             pass
#         elif self.name=="imagenet":
#             pass
#         elif self.name=="mnist":
#             pass
#         return train_len

#     @property
#     def num_classes(self) -> int:
#         """
#         Return:
#             10
#         """
#         if self.name=="cifar10" or self.name=="fashion_mnist":
#             return 10
#         elif self.name=="imagenet":
#             return 1000
#         elif self.name=="mnist":
#             pass

#     def default_transforms(self) -> Callable:
#         transforms = None
#         if self.name=="cifar10":
#             if self.normalize:
#                 transforms = transform_lib.Compose([transform_lib.ToTensor(), cifar10_normalization()])
#             else:
#                 transforms = transform_lib.Compose([transform_lib.ToTensor()])
#         elif self.name=="fashion_mnist":
#             if self.normalize:
#                 transforms = transform_lib.Compose([
#                     transform_lib.ToTensor(), transform_lib.Normalize(mean=(0.5, ), std=(0.5, ))
#                 ])
#             else:
#                 transforms = transform_lib.Compose([transform_lib.ToTensor()])
#         elif self.name=="imagenet":
#             pass
#         elif self.name=="mnist":
#             pass
        
#         return transforms
    
#     # DeePee: overload dataloader to be able to create the privacy watchdog
#     def train_dataloader(self):
#         return UniformDataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers)
#     def val_dataloader(self):
#         return UniformDataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)
#     def test_dataloader(self):
#         return UniformDataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)

# NOTE: all non-DP DataModules can directly be used from above import
# NOTE: the DataModules are slightly adapted to be used with deepee
class FashionMNISTDataModule_DP(FashionMNISTDataModule):
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

    # DeePee: change dataloader to be able to create the privacy watchdog
    def train_dataloader(self):
        return UniformDataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers)
    def val_dataloader(self):
        return UniformDataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)
    def test_dataloader(self):
        return UniformDataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)

class CIFAR10DataModule_DP(CIFAR10DataModule):
    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 16,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(  # type: ignore[misc]
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            *args,
            **kwargs,
        )

    # DeePee: change dataloader to be able to create the privacy watchdog
    def train_dataloader(self):
        return UniformDataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers)
    def val_dataloader(self):
        return UniformDataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)
    def test_dataloader(self):
        return UniformDataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)

class ImagenetDataModule_DP(ImagenetDataModule):
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

    # DeePee: change dataloader to be able to create the privacy watchdog
    def train_dataloader(self):
        return UniformDataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers)
    def val_dataloader(self):
        return UniformDataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)
    def test_dataloader(self):
        return UniformDataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)

### Custom DataModules ###
# TODO: to be added. 