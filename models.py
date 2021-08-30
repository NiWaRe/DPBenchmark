# general python 
from typing import List

# general ml
import numpy as np

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
    """
    ConvNet with max flexibility.
    Number of conv blocks, number of channels per block, kernel size 
    can be changed easily. We mimic the 'same' padding available in TF. 
    """
    def __init__(
        self, 
        in_channels: int = 3,
        img_dim: int = 32, 
        kernel_size: int = 3,
        conv_layers: List[int] = [16, 32, 64],
        ):
        super(SimpleConvNet, self).__init__()
        self.img_dim = img_dim
        self.kernel_size = kernel_size

        # calculate input dimension to first lin. layer assuming 'same' padding
        final_img_dim = int(img_dim/(2**len(conv_layers)))
        self.final_flat_dim = final_img_dim**2 * conv_layers[-1]

        conv_layers = [in_channels, *conv_layers]
        self.conv_layers = conv_layers

        # conv layers
        # 'same' padding with fixed stride=1, dilation=1  
        if (kernel_size-1)%2 == 0:
            # symmetric padding (here '//' doesn't change value, only type)
            same_pad = (kernel_size-1)//2
        else: 
            # asymmetric padding necessary
            same_pad = (kernel_size-1)//2
            same_pad = (same_pad, same_pad+1, same_pad, same_pad+1)

        self.conv_layers = nn.Sequential(
            # nn.Sequential expects single elements to be passed in, not a list
            *np.concatenate([
                [
                    nn.Conv2d(conv_layers[i], conv_layers[i+1], kernel_size, padding=same_pad),
                    # divide img_dim by two - ceil in case non-integer
                    nn.MaxPool2d(kernel_size=2, stride=2), 
                    nn.ReLU()
                ] 
                for i in range(len(conv_layers)-1)
            ]).tolist()
        )
        
        # TODO: adaptive pooling layer, auf 1024, 512
        # dense final layer
        self.fc1 = nn.Linear(self.final_flat_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(-1, self.final_flat_dim)

        # check if batch_sizes the same
        assert(x.shape[0] == out.shape[0]) 
        out = self.relu1(self.fc1(out))
        out = self.relu2(self.fc2(out))
        out = self.fc3(out)
        return out

class DepthConvNet(nn.Module):
    """
    ConvNet focused on only varying the number of conv blocks (depth). 
    Everything else is set based on usual design choices. 
    - ReLU as activation functions
    - Quadratic increasing number of channels per block, starting with 8 (2^3).
    """
    def __init__(
        self, 
        in_channels: int = 3,
        img_dim: int = 32, 
        kernel_size: int = 3,
        depth: int = 1,
        ):
        super(DepthConvNet, self).__init__()
        self.img_dim = img_dim
        self.kernel_size = kernel_size
        self.depth = depth

        # we fix the number of downsampling (halfing)
        # assumning CIFAR10 32 input dim, results in output dim of 4. 
        max_halfs = 3
        # if depth+1 < 3 then put maxpool behind every block.
        downsample_every = int((depth+1)/max_halfs) if depth==2 else 1
        # create strides array to equally distribute the 
        # downsampling among conv blocks (dim has to be bigger than kernel size)
        strides = np.zeros(depth+1, dtype=int).tolist()
        for i in range(min(depth+1, max_halfs)):
            strides[i*downsample_every] = 1

        # we use 'same' padding with fixed stride=1, dilation=1  
        if (kernel_size-1)%2 == 0:
            # symmetric padding (here '//' doesn't change value, only type)
            same_pad = (kernel_size-1)//2
        else: 
            print("there is no asym padding in torch 1.8.0 - do own conv2d")
            # TODO: THERE'S NO ASYM PADDING! (ALSO CHANGE IN SIMPLECONV)
            # check comment from 'kylemcdonald' https://github.com/pytorch/pytorch/issues/3867
            # asymmetric padding necessary
            same_pad = (kernel_size-1)//2
            same_pad = (same_pad, same_pad+1, same_pad, same_pad+1)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size, padding=same_pad),
            nn.MaxPool2d(kernel_size=2, stride=2) 
            if strides[0] else 
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            # nn.Sequential expects single elements to be passed in, not a list
            *np.concatenate([
                [
                    nn.Conv2d(2**i, 2**(i+1), kernel_size, padding=same_pad),
                    nn.MaxPool2d(kernel_size=2, stride=2) 
                    if strides[i-2] else 
                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1), 
                    nn.ReLU()
                ] 
                for i in range(3, 3+depth)
            ]).tolist()
        )

        # +1 because of extra layer bridging to 8
        # max(4, ...) so that it doesn't drop to 0 (4 bc we set max halfs to 3)
        #self.output_dim = max(4, int(img_dim/(2**(depth+1))))
        self.output_dim = 4
        self.output_channel = 2**(3+depth)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(self.output_dim) # quadratic

        self.fc1 = nn.Linear(self.output_channel*self.output_dim**2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.conv_layers(x)
        out = self.adaptive_pool(out)
        out = out.view(batch_size, -1)
        out = self.relu1(self.fc1(out))
        out = self.relu2(self.fc2(out))
        out = self.fc3(out)
        return out

def get_model(
    name, 
    pretrained, 
    num_classes, 
    data_name, 
    kernel_size, 
    conv_layers,
    depth) -> nn.Module:
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
        model = SimpleConvNet(in_channels, img_dim, kernel_size, conv_layers)
    elif name=="depth_conv":
        model = DepthConvNet(in_channels, img_dim, kernel_size, depth)
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