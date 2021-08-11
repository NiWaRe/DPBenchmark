# general python 
from typing import Optional, Union, Any, List

# data
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from torch.utils.data import DataLoader

from pl_bolts.datamodules import (
    FashionMNISTDataModule, 
    CIFAR10DataModule, 
    ImagenetDataModule, 
    MNISTDataModule
)

# differential privacy
from deepee import (PrivacyWrapper, PrivacyWatchdog, UniformDataLoader,
                     ModelSurgeon, SurgicalProcedures)

from opacus.utils.uniform_sampler import UniformWithReplacementSampler

### Standard Lightning Bolt Data Modules ###
# TODO: The PrivacyWrapper also supports normal DataLoaders -- don't the DataModules have that type?
# TODO: Instantiating just one of the classes doens't work (says data_dir isn't given -- although it is?)
# TODO: This doesn't work with LightningArgumentParser bc objects that are created don't have "dp" and "name"
# as fixed arguments and thus ArgumentParser doesn't find an "action". Why don't the "self.dp = dp", doesn't
# tell the ArgumentParser that "dp" and "name" are needed in child object?
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

        dataset_cls.__init__(  
            self,# type: ignore[misc]
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

class CIFAR10DataModuleDP(CIFAR10DataModule):
    """
    This is based on the preconfigured torchvision datamodules (e.g. CIFAR10DataModule)
    """
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
        dp_tool: str = "opacus",
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
            dp_tool: whether to use 'opacus' or 'deepee' as DP tool
        """

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
        self.dp_tool = dp_tool

    # DeePee: overload dataloaders to be able to create the privacy watchdog
    # TODO: if UniformDataLoader would accept *args, **kwargs 
    #       I could simply overload VisionDataModule._data_loader
    def train_dataloader(self) -> Union[UniformDataLoader, DataLoader]:
        dataloader = None
        if self.dp_tool == "opacus":
            dataloader = DataLoader(
                self.dataset_train, 
                num_workers=self.num_workers,
                batch_sampler=UniformWithReplacementSampler(
                    num_samples=len(self.dataset_train), 
                    sample_rate=self.batch_size/len(self.dataset_train)
                ),
            )
        elif self.dp_tool == "deepee":
            dataloader = UniformDataLoader(
                self.dataset_train, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers
            )
        else: 
            print("Please select either opacus or deepee as DP tool")
        return dataloader

    def val_dataloader(self) -> Union[UniformDataLoader, DataLoader]:
        dataloader = None
        if self.dp_tool == "opacus":
            dataloader = DataLoader(
                self.dataset_val, 
                num_workers=self.num_workers,
                batch_sampler=UniformWithReplacementSampler(
                    num_samples=len(self.dataset_val), 
                    sample_rate=self.batch_size/len(self.dataset_val)
                ),
            )
        elif self.dp_tool == "deepee":
            dataloader = UniformDataLoader(
                self.dataset_val, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers
            )
        else: 
            print("Please select either opacus or deepee as DP tool")
        return dataloader

    # TODO: test loader should stay the same, right?
    #def test_dataloader(self) -> Union[UniformDataLoader, DataLoader]:


class MNISTDataModuleDP(MNISTDataModule):
    """
    This is based on the preconfigured torchvision datamodules (e.g. CIFAR10DataModule)
    """
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
        dp_tool: str = "opacus",
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
            dp_tool: whether to use 'opacus' or 'deepee' as DP tool
        """

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
        self.dp_tool = dp_tool

    ## DeePee: overload dataloaders to be able to create the privacy watchdog
    # TODO: if UniformDataLoader would accept *args, **kwargs 
    #       I could simply overload VisionDataModule._data_loader
    def train_dataloader(self) -> Union[UniformDataLoader, DataLoader]:
        dataloader = None
        if self.dp_tool == "opacus":
            dataloader = DataLoader(
                self.dataset_train, 
                num_workers=self.num_workers,
                batch_sampler=UniformWithReplacementSampler(
                    num_samples=len(self.dataset_train), 
                    sample_rate=self.batch_size/len(self.dataset_train)
                ),
            )
        elif self.dp_tool == "deepee":
            dataloader = UniformDataLoader(
                self.dataset_train, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers
            )
        else: 
            print("Please select either opacus or deepee as DP tool")
        return dataloader

    def val_dataloader(self) -> Union[UniformDataLoader, DataLoader]:
        dataloader = None
        if self.dp_tool == "opacus":
            dataloader = DataLoader(
                self.dataset_val, 
                num_workers=self.num_workers,
                batch_sampler=UniformWithReplacementSampler(
                    num_samples=len(self.dataset_val), 
                    sample_rate=self.batch_size/len(self.dataset_val)
                ),
            )
        elif self.dp_tool == "deepee":
            dataloader = UniformDataLoader(
                self.dataset_val, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers
            )
        else: 
            print("Please select either opacus or deepee as DP tool")
        return dataloader

    # TODO: test loader should stay the same, right?
    #def test_dataloader(self) -> Union[UniformDataLoader, DataLoader]:


### Custom DataModules ###
# TODO: to be added. 