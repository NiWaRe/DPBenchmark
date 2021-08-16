# torch 
from torch.utils import data
import torchvision
from torch import nn

# data
from data import (
    MNISTDataModuleDP,
    CIFAR10DataModuleDP
)

# differential privacy
from deepee import (PrivacyWrapper, PrivacyWatchdog, UniformDataLoader,
                     ModelSurgeon, SurgicalProcedures, watchdog)

class SimpleConvNet(nn.Module):
    def __init__(
        self, 
        in_channels, 
        img_dim, 
        interm_n_channels, 
        end_n_channels,
        kernel_size
        ):
        super(SimpleConvNet, self).__init__()
        self.interm_n_channels = interm_n_channels
        self.end_n_channels = end_n_channels
        self.kernel_size = kernel_size
        # NOTE: padding, stride and dilation are not changed
        # TODO: same padding implementation -- or explicit calculation
        # FOCUS ON FEATURES: number of conv2d, kernel_size, nr_feature_maps, grad-clip, noise-multiplier
        self.flat_dim = int(self.end_n_channels * (img_dim/4)**2)
        self.conv1 = nn.Conv2d(in_channels, interm_n_channels, kernel_size=kernel_size, stride=1, padding='same', bias=False)
        self.conv2 = nn.Conv2d(interm_n_channels, self.end_n_channels, kernel_size=kernel_size, stride=1, padding='same', bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(self.flat_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, self.flat_dim)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

def get_model(
    name, 
    pretrained, 
    num_classes, 
    data_name, 
    interm_n_channels: int = 6,
    end_n_channels: int = 16,
    kernel_size: int = 3,) -> nn.Module:
    model = None
    in_channels = None
    img_dim = None
    # adapt input channel number based on dataset
    if data_name=="cifar10":
        in_channels = 3
        img_dim = 32
    elif data_name=="mnist":
        in_channels = 1
        img_dim = 28
    # choose model
    if name=="simple_conv": 
        model = SimpleConvNet(in_channels, img_dim, interm_n_channels, end_n_channels, kernel_size)
    elif name=="resnet18":
        model = torchvision.models.resnet18(pretrained=pretrained, num_classes=num_classes)
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
    elif name=="vgg11":
        model = torchvision.models.vgg11(pretrained=pretrained, num_classes=num_classes)
        model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.features[2] = nn.Identity()

    # TODO: surgical procedures have to be done automatically depending on model.
    #model = ModelSurgeon(SurgicalProcedures.BN_to_GN).operate(model)
    return model

# standard torchvision models are created and adapted in get_model 
# TODO: think about how to adapt models dynamically based on choice of dataset (not relevant for now)

# TODO: custom models are created in the following