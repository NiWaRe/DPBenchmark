# general python 
from typing import List
from copy import deepcopy

# general ml
import numpy as np
import timm

# torch 
from torch.utils import data
import torchvision
from torch import nn
import torch.nn.functional as F

# data
from data import (
    MNISTDataModuleDP,
    CIFAR10DataModuleDP
)

# differential privacy
from deepee import (PrivacyWrapper, PrivacyWatchdog, UniformDataLoader,
                     ModelSurgeon, SurgicalProcedures, watchdog)

# Utility Modules
class Lambda(nn.Module):
    """
    Module to encapsulate specific functions as modules. 
    Good to easily integrate squeezing or padding functions.
    """
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

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
            # TODO: THERE'S NO ASYM PADDING! (ALSO CHANGE IN SIMPLECONV)
            # check comment from 'kylemcdonald' https://github.com/pytorch/pytorch/issues/3867
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
        
        # NOTE: as alternative: adaptive pooling layer, auf 1024, 512
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

class StageConvNet(nn.Module):
    """
    ConvNet focused on only varying the number of conv stages (follwing 
    definition of ConvNet params in EfficientNet).
    Everything else is set based on usual design choices. 
    - ReLU as activation functions
    - Quadratic increasing number of channels per block, starting with 8 (2^3).
    """
    def __init__(
        self, 
        in_channels: int = 3,
        img_dim: int = 32, 
        kernel_size: int = 3,
        nr_stages: int = 1,
        ):
        super(StageConvNet, self).__init__()
        self.img_dim = img_dim
        self.kernel_size = kernel_size
        self.nr_stages = nr_stages

        # we fix the number of downsampling (halfing)
        # assumning CIFAR10 32 input dim, results in output dim of 4. 
        max_halfs = 3
        # if nr_stages+1 < 3 then put maxpool behind every block.
        downsample_every = int((nr_stages+1)/max_halfs) if nr_stages>=2 else 1
        # create strides array to equally distribute the 
        # downsampling among conv blocks (dim has to be bigger than kernel size)
        strides = np.zeros(nr_stages+1, dtype=int).tolist()
        for i in range(min(nr_stages+1, max_halfs)):
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
                for i in range(3, 3+nr_stages)
            ]).tolist()
        )

        # +1 because of extra layer bridging to 8
        # max(4, ...) so that it doesn't drop to 0 (4 bc we set max halfs to 3)
        #self.output_dim = max(4, int(img_dim/(2**(depth+1))))
        self.output_dim = 4
        self.output_channel = 2**(3+nr_stages)
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

class ENScalingBaseModel(nn.Module):
    """
    ConvNet that is based on stage 1 network of StageConvNet and where
    depth (factor multiplied number of equal size conv blocks) and width 
    (factor multiplied number of channels per conv block) can be changed seperately. 
    By default depth and width are 1 which results in the stage 1 network of StafeConvNet.
    Args: 
        depth - a factor multiplied with number of conv blocks per stage of base model
        width - a factor multiplied with number of channels per conv block of base model
    """
    def __init__(
        self, 
        in_channels: int = 3,
        depth: float = 1.0,
        width: float = 1.0,
        ):
        super(ENScalingBaseModel, self).__init__()
        self.depth = depth
        self.width = width

        ## STAGE 0 ##
        # the stage 1 base model has 8 channels in stage 0
        # the stage 1 base model has 1 conv block per stage (in both stage 0 and 1)
        width_stage_zero = int(width*8)
        depth_stage_zero = int(depth*1)
        self.stage_zero = nn.Sequential(
            nn.Conv2d(in_channels, width_stage_zero, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0) 
            if depth <= 1 else 
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            # nn.Sequential expects single elements to be passed in, not a list
            *np.concatenate([
                [
                    nn.Conv2d(width_stage_zero, width_stage_zero, kernel_size=3, padding=1),
                    # for last convblock add downsampling (halving in this case)
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0) 
                    if i == depth_stage_zero-1 else
                    # otherwise keep dimensions
                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1), 
                    nn.ReLU()
                ] if i > 0 else []
                # the if/else has to be added in case the depth_stage_zero == 1 
                # because in that case np.concatenate would receive '[]' as input 
                # which yields an error
                for i in range(depth_stage_zero)
            ]).tolist(),
        )

        ## STAGE 1 ##
        # the stage 1 base model has 16 channels in stage 1
        # the stage 1 base model has 1 conv block per stage (in both stage 0 and 1)
        width_stage_one = int(width*16)
        depth_stage_one = int(depth*1)
        self.stage_one = nn.Sequential(
            nn.Conv2d(width_stage_zero, width_stage_one, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0) 
            if depth <= 1 else 
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            # nn.Sequential expects single elements to be passed in, not a list
            *np.concatenate([
                [
                    nn.Conv2d(width_stage_one, width_stage_one, kernel_size=3, padding=1),
                    # for last convblock add downsampling (halving in this case)
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0) 
                    if i == depth_stage_one-1 else
                    # otherwise keep dimensions
                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1), 
                    nn.ReLU()
                ] if i > 0 else []
                # the if/else has to be added in case the depth_stage_one == 1 
                # because in that case np.concatenate would receive '[]' as input 
                # which yields an error
                for i in range(depth_stage_one)
            ]).tolist(), 
        )   

        ## Final FC Block ##
        # output_dim is fixed to 4 (even if 8 makes more sense for the stage 1 StageConvModel)
        output_dim = 4
        self.adaptive_pool = nn.AdaptiveAvgPool2d(4) 

        self.fc1 = nn.Linear(width_stage_one*output_dim**2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.stage_zero(x) 
        out = self.stage_one(out) 
        out = self.adaptive_pool(out)
        out = out.view(batch_size, -1)
        out = self.relu1(self.fc1(out))
        out = self.relu2(self.fc2(out))
        out = self.fc3(out)
        return out

# NOTE: Specific features have been designed for CIFAR10 Input specifically
# NOTE: Difference to Stage 1 Network: Downsampling and channel adaptation
#       happens in first Conv2d.
class ResidualBlock(nn.Module):
    """
    Inspired by residual learning of ResNet.
    Main focus is on using residual learning aspect, using identity skip connections
    (projected connections are not considered), specifics of original paper are not 
    considered in detail (whether downsampling happens in first or last ConvBlock 
    per Residual Block - if there's a downsampling at all).

    Specifically geared towards experiments for the stage 1 network of StageConvNet, 
    which is used as baseline in my manually crafted experiments. Downsampling means
    halving with same padding specifically for CIFAR10 input. 

    Args:
        in_channels: the number of channels (feature maps) of the incoming embedding
        out_channels: the number of channels after the first convolution
        halve_dim : whether to half the dimension in the last Conv2D
        after_conv_fc: norm or pooling after Conv2D layer (BatchNorm2d, MaxPool, Identity)
        skip_depth: how much blocks skip connection should jump,
            2 = default, 0 = no skip connections
    """

    def __init__(
        self, 
        in_channels : int, 
        out_channels : int,
        halve_dim : bool, 
        after_conv_fc : nn.Module,
        skip_depth : int = 2,
    ):
        super().__init__()
        # save for turning on and off skip connection
        self.skip_depth = skip_depth

        # also add this so that skip doesn't show up in model print
        if skip_depth: 
            # add strides in the skip connection and zeros for the new channels
            # if output dimension or output channels differ from respective input
            if halve_dim or in_channels != out_channels:
                self.skip = Lambda(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2] if halve_dim else x, # stride = 2 for halving
                        (0, 0, 0, 0, 0, out_channels - in_channels), 
                        mode="constant", value=0
                    )
                )
            else:
                self.skip = nn.Sequential()

        # for first layer either add downsampling layer or normal layer
        # in any case adapt channels 
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size = 2 if halve_dim else 3, 
                    stride = 2 if halve_dim else 1, 
                    padding = 0 if halve_dim else 1, 
                    bias=False
                )
            ]
        )
        
        # skip_depth decides on how many Conv2d layers one Residual Block has
        # -1 the one we just created above
        self.conv_layers.extend(
            [
                nn.Conv2d(
                    out_channels, 
                    out_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1, 
                    bias=False
                )
                for i in range(1, skip_depth)
            ]
        )

        # initiate the functions after Conv2d
        self.after_conv_fcs = nn.ModuleList(
            [
                deepcopy(after_conv_fc)
                for i in range(skip_depth)
            ] 
            # extra handling of case with no skip connections
            if skip_depth else 
            [
                deepcopy(after_conv_fc)
            ]
        )

        # initiative activation functions
        self.activation_fcs = nn.ModuleList(
            [
                nn.ReLU()
                for i in range(skip_depth)
            ]
            # extra handling of case with no skip connections
            if skip_depth else 
            [
                nn.ReLU()
            ]
        )

    def forward(self, input):
        # go through all triples expect last one
        out = input
        for i in range(len(self.conv_layers)-1):
            out = self.conv_layers[i](out)
            out = self.after_conv_fcs[i](out)
            out = self.activation_fcs[i](out)

        # add last triple and connect input
        out = self.conv_layers[-1](out)
        out = self.after_conv_fcs[-1](out) 
        if self.skip_depth: 
            out += self.skip(input)
        out = self.activation_fcs[-1](out)

        return out

class ResidualStack(nn.Module):
    """
    Helper module to stack the different residual blocks. 
    
    Args:
        in_channels: The number of channels (feature maps) of the incoming embedding
        out_channels: The number of channels after the first layer
        halve_dim : whether to half the dimension in the last Conv2D
        after_conv_fc: norm or pooling after Conv2D layer (BatchNorm2d, MaxPool, Identity)
        skip_depth: how much blocks skip connection should jump,
            2 = default, 0 = no skip connections
        num_blocks: Number of residual blocks
    """
    
    def __init__(
        self, 
        in_channels : int, 
        out_channels : int, 
        halve_dim : bool, 
        after_conv_fc : nn.Module,
        skip_depth : int,
        num_blocks : int, 
    ):
        super().__init__()

        # first block to get the right number of channels (from previous block to current)
        # and sample down if specified (specifically at the first layer in the ResidualBlock)
        self.residual_stack = nn.ModuleList(
            [
                ResidualBlock(in_channels, out_channels, halve_dim, after_conv_fc, skip_depth)
            ]
        )
        
        # EXTEND adds array as elements of existing array, APPEND adds array as new element of array 
        self.residual_stack.extend(
            [
                ResidualBlock(out_channels, out_channels, False, after_conv_fc, skip_depth) 
                for i in range(1, num_blocks)
            ]
        )
        
    def forward(self, input):
        out = input
        for layer in self.residual_stack:
            out = layer(out)
        return out

# NOTE: some differences to my manually crafted CNN (not just added skip connections)
    # - downsampling is done with adaption of channels in first ConvBlock 
    # - downsampling is done through Conv2d layer and not dedicated maxpool layer

class ENScalingResidualModel(nn.Module):
    """
    ConvNet that is based on stage 1 network of StageConvNet and where
    depth (factor multiplied number of equal size conv blocks) and width 
    (factor multiplied number of channels per conv block) can be changed seperately. 
    By default depth and width are 1 which results in the stage 1 network of StafeConvNet.
    In the contrary to the ENScalingBaseModel this model considers residual connections as
    where implemented in the ResNet (skip connections over two conv blocks with batchnorm
    without any pooling applied behind the conv layer.)
    Args: 
        halve_dim : whether to half the dimension in the last Conv2D
        after_conv_fc: norm or pooling after Conv2D layer (BatchNorm2d, MaxPool, Identity)
        skip_depth: how much blocks skip connection should jump,
            2 = default, 0 = no skip connections
        depth - a factor multiplied with number of conv blocks per stage of base model
        width - a factor multiplied with number of channels per conv block of base model
                := num_blocks (as defined in the scaling approach)
    """
    def __init__(
        self, 
        halve_dim : bool, 
        after_conv_fc_str : str,
        skip_depth : int = 2, 
        in_channels: int = 3,
        depth: float = 1.0,
        width: float = 1.0,
        ):
        super(ENScalingResidualModel, self).__init__()

        ## STAGE 0 ##
        # the stage 1 base model has 8 channels in stage 0
        # the stage 1 base model has 1 conv block per stage (in both stage 0 and 1)
        width_stage_zero = int(width*8)
        depth_stage_zero = int(depth*1)

        # instantiate after_conv_fc accoridng to params
        if after_conv_fc_str == 'batch_norm':
            after_conv_fc = nn.BatchNorm2d(width_stage_zero)
        elif after_conv_fc_str == 'group_norm':
            # based on original paper 32 should be the number of groups, 
            # but here we choose 8 because the number of channels in efficientNets
            # is only dividable by 8. If number of channels is smaller than that we 
            # take the number of channels, that's why we apply a min.

            # TODO: tune group size. 
            after_conv_fc = nn.GroupNorm(
                min(8, width_stage_zero), 
                width_stage_zero, 
                affine=True
            )
        elif after_conv_fc_str == 'max_pool': 
            # keep dimensions for CIFAR10 dimenions assuming a downsampling 
            # only through halving. 
            after_conv_fc = nn.MaxPool2d(
                kernel_size=3, 
                stride=1, 
                padding=1
            )
        elif after_conv_fc_str == 'identity': 
            after_conv_fc = nn.Identity()


        self.stage_zero = nn.Sequential(
            ResidualStack(
                in_channels,
                width_stage_zero,
                halve_dim,
                after_conv_fc, 
                skip_depth, 
                depth_stage_zero
            ),
        )

        ## STAGE 1 ##
        # the stage 1 base model has 16 channels in stage 1
        # the stage 1 base model has 1 conv block per stage (in both stage 0 and 1)
        width_stage_one = int(width*16)
        depth_stage_one = int(depth*1)

        # instantiate after_conv_fc accoridng to params
        if after_conv_fc_str == 'batch_norm':
            after_conv_fc = nn.BatchNorm2d(width_stage_one)
        elif after_conv_fc_str == 'group_norm':
            # based on original paper 32 should be the number of groups, 
            # but here we choose 8 because the number of channels in efficientNets
            # is only dividable by 8. If number of channels is smaller than that we 
            # take the number of channels, that's why we apply a min.

            # TODO: tune group size. 
            # NAS width values were restricted to multiples of 8
            after_conv_fc = nn.GroupNorm(
                min(8, width_stage_one), 
                width_stage_one, 
                affine=True
            )
        elif after_conv_fc_str == 'max_pool': 
            # keep dimensions for CIFAR10 dimenions assuming a downsampling 
            # only through halving. 
            after_conv_fc = nn.MaxPool2d(
                kernel_size=3, 
                stride=1, 
                padding=1
            )
        elif after_conv_fc_str == 'identity': 
            after_conv_fc = nn.Identity()

        self.stage_one = nn.Sequential(
            ResidualStack(
                width_stage_zero,
                width_stage_one,
                halve_dim,
                after_conv_fc, 
                skip_depth, 
                depth_stage_one
            ),
        )   

        ## Final FC Block ##
        # output_dim is fixed to 4 (even if 8 makes more sense for the stage 1 StageConvModel)
        output_dim = 4
        self.adaptive_pool = nn.AdaptiveAvgPool2d(4) 

        self.fc1 = nn.Linear(width_stage_one*output_dim**2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.stage_zero(x) 
        out = self.stage_one(out) 
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
    nr_stages, 
    depth, 
    width,
    halve_dim, 
    after_conv_fc_str, 
    skip_depth,
) -> nn.Module:
    model = None
    in_channels = None
    img_dim = None
    output_classes = None
    # adapt input channel number based on dataset
    if data_name=="cifar10":
        in_channels = 3
        img_dim = 32
        output_classes = 10
    elif data_name=="mnist":
        in_channels = 1
        img_dim = 28
        output_classes = 10
    # choose model
    if name=="simple_conv": 
        model = SimpleConvNet(
            in_channels, 
            img_dim, 
            kernel_size, 
            conv_layers
        )
    elif name=="stage_conv":
        model = StageConvNet(
            in_channels, 
            img_dim, 
            kernel_size, 
            nr_stages
        )
    elif name=="en_scaling_base_model":
        # img_dim is assumed to be 32 (but model works also with other img_dim)
        model = ENScalingBaseModel(
            in_channels, 
            depth, 
            width,
        )
    elif name=="en_scaling_residual_model":
        # img_dim is assumed to be 32 (but model works also with other img_dim)
        model = ENScalingResidualModel(
            halve_dim, 
            after_conv_fc_str, 
            skip_depth, 
            in_channels, 
            depth, 
            width
        )
    elif name=="resnet18":
        model = torchvision.models.resnet18(
            pretrained=pretrained, 
            num_classes=num_classes
        )
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif name=="vgg11":
        model = torchvision.models.vgg11(
            pretrained=pretrained, 
            num_classes=num_classes
        )
        model.features[0] = nn.Conv2d(
            in_channels, 
            64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        model.features[2] = nn.Identity()
    elif name=="inception_v4": 
        model = timm.create_model("inception_v4", pretrained=True)
        model.last_linear = nn.Linear(1536, output_classes)
    elif name=="densenet121": 
        model = timm.create_model("densenet121", pretrained=True)
        model.classifier = nn.Linear(1024, output_classes)
    elif name=="mobilenetv3_large_100": 
        model = timm.create_model("mobilenetv3_large_100", pretrained=True)
        model.classifier = nn.Linear(1280, output_classes)
    # by default a timm model is created
    else: 
        # TODO: try out different pretrainings 
        # TODO check that batchnorm and data normalization works fine.
        model = timm.create_model(name, pretrained=True)
        # freezing model params
        # for param in model.parameters():
        #     param.requires_grad = False
        # adding new classifier which is trained
        # NOTE: b0 was 1280, b3 was 1536, b5 was 2048, b7 was 2560
        model.classifier = nn.Linear(1536, output_classes)

    # TODO: surgical procedures have to be done automatically depending on model.
    #model = ModelSurgeon(SurgicalProcedures.BN_to_GN).operate(model)
    return model

# standard torchvision models are created and adapted in get_model 
# TODO: think about how to adapt models dynamically based on choice of dataset (not relevant for now)

# TODO: custom models are created in the following