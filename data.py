# general python 
import os
from typing import Optional, Union, Callable, Any, List, Tuple

# data
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms 

# utils
from PIL import Image
import pandas as pd

from pl_bolts.datamodules import (
    FashionMNISTDataModule, 
    CIFAR10DataModule, 
    ImagenetDataModule, 
    MNISTDataModule
)

# differential privacy
from deepee import (PrivacyWrapper, PrivacyWatchdog, UniformDataLoader,
                     ModelSurgeon, SurgicalProcedures)

from opacus.data_loader import DPDataLoader

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
            return DPDataLoader.from_data_loader(
                DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers), 
                distributed=False,
            )
        else: 
            # standard from class VisionDataModule
            return self._data_loader(self.dataset_train, shuffle=self.shuffle)
    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader], UniformDataLoader]:
        # standard from class VisionDataModule
        return self._data_loader(self.dataset_val)
    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader], UniformDataLoader]:
        # standard from class VisionDataModule
        return self._data_loader(self.dataset_test)

class CIFAR10DataModuleDP(CIFAR10DataModule):
    """
    This is based on the preconfigured torchvision datamodules (e.g. CIFAR10DataModule)
    CIFAR10 = 60K, 32x32 colour images in 10 classes equally distributed. 50K training, 10K test.
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
            dataloader = DPDataLoader.from_data_loader(
                DataLoader(
                    self.dataset_train, 
                    num_workers=self.num_workers,
                    batch_size=self.batch_size,
                )
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


class MNISTDataModuleDP(MNISTDataModule):
    """
    This is based on the preconfigured torchvision datamodules (e.g. MNISTDataModule).
    MNIST = 70K 28x28 grayscale images. 60K training, 10K test. 
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
            dataloader = DPDataLoader.from_data_loader(
                DataLoader(
                    self.dataset_train, 
                    num_workers=self.num_workers,
                    batch_size=self.batch_size,
                )
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


### Custom DataModules ###

class ImageNetteDataClass(Dataset):
    """
        ImageNette Dataset. 
        We assume that the images are already downloaded at given path.
        https://github.com/fastai/imagenette
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        noisy_labels: str = "noisy_labels_0", # by default no noisy labels
        ):
        """
        Args:
            Same args as for MNISTDataset.
        """
        # NOTE: change dimension in DP module if other dataset is chosen here!
        self.root_dir = root+"/imagenette2/"

        # convert labels from string to ints (for internals of lightning)
        data_info = pd.read_csv(self.root_dir+"noisy_imagenette.csv")
        unique_labels = data_info[noisy_labels].unique().tolist()
        map = {unique_labels[i]:i for i in range(len(unique_labels))}
        data_info = data_info.assign(
            targets=[
                map[data_info[noisy_labels].iloc[i]]
                for i in range(len(data_info))
            ],
        )
        # extract only train dataset or validation dataset
        self.data_info = data_info[
            data_info.is_valid==False
            if train else
            data_info.is_valid==True
        ][["path", "targets"]]
    
        # add custom tranforms if wanted
        self.transform = transform
        self.target_transform = target_transform

        if download:
            print("No download functionality for now, please download manually and pass in path to data.")


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        # load in image as PIL.Image 
        img = Image.open(os.path.join(self.root_dir, self.data_info.iloc[index, 0])).convert('RGB')

        # load in target as str
        
        target = self.data_info.iloc[index, 1]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self) -> int:
        return len(self.data_info)

class ImageNetteDataModuleDP(VisionDataModule):
    """

    Specs:
        - 10 classes (1 per digit)
        - Each image is (3 x 224 x 224)

    Standard ImageNette, train, val splits and transforms
    """
    name = "imagenette"
    dataset_cls = ImageNetteDataClass
    dims = (3, 224, 224)

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

    @property
    def num_classes(self) -> int:
        """
        Return:
            10
        """
        return 10

    def default_transforms(self) -> Callable:
        if self.normalize:
            # we take the standard transformation for validation from the Lightning ImageNet Class
            imagenette_transforms = transforms.Compose(
                [
                    # NOTE: change dimension if other dataset!
                    transforms.Resize(224 + 32),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            imagenette_transforms = transforms.Compose([transforms.ToTensor()])
        
        return imagenette_transforms


    ## DeePee: overload dataloaders to be able to create the privacy watchdog
    # TODO: if UniformDataLoader would accept *args, **kwargs 
    #       I could simply overload VisionDataModule._data_loader
    def train_dataloader(self) -> Union[UniformDataLoader, DataLoader]:
        dataloader = None
        if self.dp_tool == "opacus":
            dataloader = DPDataLoader.from_data_loader(
                DataLoader(
                    self.dataset_train, 
                    num_workers=self.num_workers,
                    batch_size=self.batch_size,
                )
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

# Checklist new dataset
# 1. Create Dataset
# 2. Create DPModule based on Dataset
# 3. In trainer.py add extra (search for 'cifar10')
# 4. In models.py add extra (search for 'cifar10')