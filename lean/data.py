# general python 
import os
from typing import Optional, Union, Callable, Any, List, Tuple

# data
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms 

# utils
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# differential privacy
from opacus.data_loader import DPDataLoader

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

        # create a map from unique str targets to int targets
        self.map = {unique_labels[i]:i for i in range(len(unique_labels))}
        
        # create extra column 'targets' with new int targets
        data_info = data_info.assign(
            targets=[
                self.map[data_info[noisy_labels].iloc[i]]
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

############
# GET DATA #
############

# TODO: add validation set with train_test_split()
# TODO: add CIFAR10, as option
def etl_data(
        data_name: str,
        val_split: float,
    ): 
    """
        Function to get normalized dataset.
    """

    # we take the standard transformation for validation from the Lightning ImageNet Class
    image_dim = 224
    imagenette_transforms = transforms.Compose(
        [
            transforms.Resize(image_dim + 32),
            transforms.CenterCrop(image_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # first create two training datasets and then select the relevant portions
    train_dataset = ImageNetteDataClass(
        root = "/home/nico/DPBenchmark/data",
        train = True,
        transform = imagenette_transforms,
        target_transform = imagenette_transforms,
    )

    val_dataset = ImageNetteDataClass(
        root = "/home/nico/DPBenchmark/data",
        train = True,
        transform = imagenette_transforms,
        target_transform = imagenette_transforms,
    )

    # create shuffled indices for every sample in the train_dataset
    indices = list(range(len(train_dataset)))
    np.random.shuffle(indices)

    # take the first indices for the validation set
    val_indices = indices[
        :int(len(indices)*val_split)
    ]
    train_indices = indices[
        int(len(indices)*val_split):
    ]

    # only select the training and validation part
    train_dataset.data_info = train_dataset.data_info.iloc[train_indices]
    val_dataset.data_info = val_dataset.data_info.iloc[val_indices]

    # create test dataset
    test_dataset = ImageNetteDataClass(
        root = "/home/nico/DPBenchmark/data",
        train = False,
        transform = imagenette_transforms,
        target_transform = imagenette_transforms,
    )

    # check if length make sense
    print(f"Datasets created, train len: {len(train_dataset)}, \
            val len: {len(val_dataset)}, \
            test len: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset