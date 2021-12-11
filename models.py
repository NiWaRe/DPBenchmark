# general python 
from typing import List, Optional
from copy import deepcopy
from functools import partial
import sys, os

# general ml
import numpy as np
import timm

# torch 
import torch
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

# own utility functions and classes
from utils import Lambda, getAfterConvFc, getActivationFunction

# additional models
sys.path.append(os.path.join(os.path.dirname(__file__), 'local_models/CondenseNet'))
from local_models.CondenseNet.models import condensenet # models/CondenseNet/models/condensenet.py

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
        after_conv_fc_str: norm or pooling after Conv2D layer (BatchNorm2d, MaxPool, Identity)
        activation_fc_str: choose activation function
        skip_depth: how much blocks skip connection should jump,
            2 = default, 0 = no skip connections
        dsc: whether to use depthwise seperable convolutions or not
    """

    def __init__(
        self, 
        in_channels : int, 
        out_channels : int,
        halve_dim : bool, 
        after_conv_fc_str : str,
        activation_fc_str : str,
        skip_depth : int = 2,
        dsc : bool = False,  
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
                if not dsc else 
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, 
                        in_channels, 
                        groups=in_channels,
                        kernel_size = 2 if halve_dim else 3, 
                        stride = 2 if halve_dim else 1, 
                        padding = 0 if halve_dim else 1, 
                        bias=False, 
                    ), 
                    nn.Conv2d(
                        in_channels, 
                        out_channels, 
                        kernel_size=1, 
                        bias=False
                    )
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
                if not dsc else 
                nn.Sequential(
                    nn.Conv2d(
                        out_channels, 
                        out_channels, 
                        groups=out_channels,
                        kernel_size=3, 
                        stride=1, 
                        padding=1, 
                        bias=False
                    ), 
                    nn.Conv2d(
                        out_channels, 
                        out_channels, 
                        kernel_size=1, 
                        bias=False
                    )
                )
                for i in range(1, skip_depth)
            ]
        )

        # initiate the functions after Conv2d
        after_conv_fc = getAfterConvFc(
            after_conv_fc_str=after_conv_fc_str, 
            num_features=out_channels,
        )
        self.after_conv_fcs = nn.ModuleList(
            [
                deepcopy(after_conv_fc)
                for i in range(skip_depth)
            ] 
            # extra handling of case with no skip connections
            if skip_depth else 
            [
                after_conv_fc
            ]
        )

        # initiative activation functions
        self.activation_fcs = nn.ModuleList(
            [
                deepcopy(
                    getActivationFunction(activation_fc_str)
                )
                for i in range(skip_depth)
            ]
            # extra handling of case with no skip connections
            if skip_depth else 
            [
                getActivationFunction(activation_fc_str)
            ]
        )

        # dropout
        # TODO: remove comment to use dropout
        # self.dropout_layers = nn.ModuleList(
        #     [
        #         deepcopy(
        #             nn.Dropout2d(0.1)
        #         )
        #         for i in range(skip_depth)
        #     ]
        #     if skip_depth else
        #     [
        #         nn.Dropout2d(0.1)
        #     ]
        # )

    def forward(self, input):
        # go through all triples expect last one
        out = input
        for i in range(len(self.conv_layers)-1):
            out = self.conv_layers[i](out)
            out = self.after_conv_fcs[i](out)
            out = self.activation_fcs[i](out)
            # dropout in between conv blocks as done in WideResNet
            # TODO: remove comment to use
            #out = self.dropout_layers[i](out)

        # add last triple and connect input
        out = self.conv_layers[-1](out)
        out = self.after_conv_fcs[-1](out) 
        if self.skip_depth: 
            out += self.skip(input)
        out = self.activation_fcs[-1](out)
        # add the end of the block
        # out = self.dropout_layers[-1](out)

        return out

class ResidualStack(nn.Module):
    """
    Helper module to stack the different residual blocks. 
    
    Args:
        in_channels: The number of channels (feature maps) of the incoming embedding
        out_channels: The number of channels after the first layer
        halve_dim : whether to half the dimension in the last Conv2D
        after_conv_fc_str: norm or pooling after Conv2D layer (BatchNorm2d, MaxPool, Identity)
        activation_fc_str: choose what activation function to use
        skip_depth: how much blocks skip connection should jump,
            2 = default, 0 = no skip connections
        num_blocks: Number of residual blocks
        dense: whether dense connections are used (in this case: skip_depth=0, halve_dim=False) 
        dsc: whether to use depthwise seperable convolutions or not
    """
    
    def __init__(
        self, 
        in_channels : int, 
        out_channels : int, 
        halve_dim : bool, 
        after_conv_fc_str : str,
        activation_fc_str : str,
        skip_depth : int,
        num_blocks : int, 
        dense : bool,
        dsc : bool
    ):
        super().__init__()
        self.dense = dense

        # first block to get the right number of channels (from previous block to current)
        # and sample down if specified (specifically at the first layer in the ResidualBlock)
        self.residual_stack = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    halve_dim=halve_dim, 
                    after_conv_fc_str=after_conv_fc_str, 
                    activation_fc_str=activation_fc_str,
                    skip_depth=skip_depth, 
                    dsc=dsc,
                )
            ]
        )
        
        # EXTEND adds array as elements of existing array, APPEND adds array as new element of array 
        self.residual_stack.extend(
            [
                ResidualBlock(
                    in_channels=out_channels*i+in_channels if dense else out_channels, 
                    out_channels=out_channels, 
                    halve_dim=False, 
                    after_conv_fc_str=after_conv_fc_str, 
                    activation_fc_str=activation_fc_str,
                    skip_depth=skip_depth, 
                    dsc=dsc,
                ) 
                for i in range(1, num_blocks)
            ]
        )
        
    def forward(self, input):
        out = input
        for layer in self.residual_stack:
            temp = layer(out)
            if self.dense: 
                # concatenate at channel dimension
                out = torch.cat((out, temp), 1)
            else: 
                out = temp
        return out

# NOTE: some differences to my manually crafted CNN (not just added skip connections)
    # - downsampling is done with adaption of channels in first ConvBlock 
    # - downsampling is done through Conv2d layer and not dedicated maxpool layer

# TODO: make adaptive for MNIST (most assume dn RGB img with 3 channels)
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
        activation_fc_str: choose activation function
        skip_depth: how much blocks skip connection should jump,
            2 = default, 0 = no skip connections
        dense: whether dense connections are used (skip_depth=0, halve_dim=False in this case)
        depth: a factor multiplied with number of conv blocks per stage of base model
        width: a factor multiplied with number of channels per conv block of base model
                := num_blocks (as defined in the scaling approach)
        dsc: whether depthwise seperable convolutions are used or normal convolutions
    """
    def __init__(
        self, 
        halve_dim : bool, 
        after_conv_fc_str : str,
        activation_fc_str : str,
        skip_depth : int = 2, 
        dense : bool = False,
        in_channels: int = 3,
        depth: float = 1.0,
        width: float = 1.0,
        dsc: bool = False,
        ):
        super(ENScalingResidualModel, self).__init__()
        self.dense = dense
        if dense: 
            if halve_dim == True or skip_depth != 0: 
                print("You're using dense connections => halve_dim=False and skip_depth=0")
            halve_dim = False
            skip_depth = 0

        ## STAGE 0 ##
        # the stage 1 base model has 8 channels in stage 0
        # the stage 1 base model has 1 conv block per stage (in both stage 0 and 1)
        width_stage_zero = int(width*8)
        depth_stage_zero = int(depth*1)

        self.stage_zero = nn.Sequential(
            ResidualStack(
                in_channels=in_channels,
                out_channels=width_stage_zero,
                halve_dim=halve_dim,
                after_conv_fc_str=after_conv_fc_str, 
                activation_fc_str=activation_fc_str,
                skip_depth=skip_depth, 
                num_blocks=depth_stage_zero, 
                dense=dense, 
                dsc=dsc,
            ),
        )

        ## STAGE 1 ##
        # the stage 1 base model has 16 channels in stage 1
        # the stage 1 base model has 1 conv block per stage (in both stage 0 and 1)
        width_stage_one = int(width*16)
        depth_stage_one = int(depth*1)
        
        # DenseTransition if using DenseBlocks #
        if self.dense: 
            # recalculate the number of input features 
            # depth_stage_zero (total num of blocks) + input channels (3 for CIFAR10)
            width_stage_zero = width_stage_zero * depth_stage_zero + 3
            # same as original Tranistion Layers in DenseNet
            # features are halved through 1x1 Convs and AvgPool is used to halv the dims
            self.dense_transition = nn.Sequential(
                #getAfterConvFc(after_conv_fc_str, width_stage_zero), 
                nn.Conv2d(
                    width_stage_zero, 
                    width_stage_zero//2, 
                    kernel_size=1, 
                    stride=1, 
                    bias=False
                ),
                nn.AvgPool2d(
                    kernel_size=2, 
                    stride=2
                ), 
            )
            width_stage_zero = width_stage_zero // 2

        self.stage_one = nn.Sequential(
            ResidualStack(
                in_channels=width_stage_zero,
                out_channels=width_stage_one,
                halve_dim=halve_dim,
                after_conv_fc_str=after_conv_fc_str, 
                activation_fc_str=activation_fc_str,
                skip_depth=skip_depth, 
                num_blocks=depth_stage_one, 
                dense=dense, 
                dsc=dsc,
            ),
        )   

        if self.dense: 
            self.pre_final = nn.AvgPool2d(kernel_size=2, stride=2)
            self.width_stage_one = width_stage_one

        ## Final FC Block ##
        # output_dim is fixed to 4 (even if 8 makes more sense for the stage 1 StageConvModel)
        output_dim = 4
        self.adaptive_pool = nn.AdaptiveAvgPool2d(4) 

        self.fc1 = nn.Linear(width_stage_one*output_dim**2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu1 = nn.SELU()
        self.relu2 = nn.SELU()

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.stage_zero(x)
        # add dense transition layer if using dense connections; in this case
        # no dim or feature reduction will happen in the stages themselves
        if self.dense: 
            out = self.dense_transition(out)
        out = self.stage_one(out) 
        # as input to last FC layer only the output of the last conv_block 
        # should be considered in the dense connection case
        if self.dense: 
            # last pooling layer to downsampling (same as in DenseNet)
            out = self.pre_final(out)
            # only get output of last conv layer
            out = out[:, -self.width_stage_one:, :, :]
        out = self.adaptive_pool(out)
        out = out.view(batch_size, -1)
        out = self.relu1(self.fc1(out))
        out = self.relu2(self.fc2(out))
        out = self.fc3(out)
        return out

# Extra Arguments Class for CondenseNet
class Args: 
    def __init__(
        self, 
        data: str, 
        num_classes: int,
        stages: List[int], 
        growth: List[int], 
        group_1x1: int, 
        group_3x3: int, 
        bottleneck: int, 
        condense_factor: int, 
        dropout_rate: int,
    ) -> None:
        # data for dimension 
        self.data = data
        # num classes
        self.num_classes = num_classes
        # per layer depth
        self.stages = stages
        # per layer growth 
        self.growth = growth
        # 1x1 group convolution (default: 4)
        self.group_1x1 = group_1x1 
        # 3x3 group convolution (default: 4)
        self.group_3x3 = group_3x3
        # bottleneck (default: 4)
        self.bottleneck = bottleneck
        # condense factor (default: 4)
        self.condense_factor = condense_factor
        # drop out (default: 0)
        self.dropout_rate = dropout_rate

def get_model(
    model_name, 
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
    activation_fc_str,
    skip_depth,
    dense,
    dsc
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
    if model_name=="simple_conv": 
        model = SimpleConvNet(
            in_channels, 
            img_dim, 
            kernel_size, 
            conv_layers
        )
    elif model_name=="stage_conv":
        model = StageConvNet(
            in_channels, 
            img_dim, 
            kernel_size, 
            nr_stages
        )
    elif model_name=="en_scaling_base_model":
        # img_dim is assumed to be 32 (but model works also with other img_dim)
        model = ENScalingBaseModel(
            in_channels, 
            depth, 
            width,
        )
    elif model_name=="en_scaling_residual_model":
        # img_dim is assumed to be 32 (but model works also with other img_dim)
        model = ENScalingResidualModel(
            halve_dim=halve_dim, 
            after_conv_fc_str=after_conv_fc_str, 
            activation_fc_str=activation_fc_str,
            skip_depth=skip_depth, 
            dense=dense,
            in_channels=in_channels, 
            depth=depth, 
            width=width,
            dsc=dsc,
        )
    elif model_name=="resnet18":
        model = torchvision.models.resnet18(
            pretrained=pretrained
        )
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(512, output_classes)
    elif model_name=="resnet50": 
        model = timm.create_model("resnet50", pretrained=pretrained)
        model.fc = nn.Linear(2048, output_classes)
    elif model_name=="wide_resnet50_2": 
        model = timm.create_model("wide_resnet50_2", pretrained=pretrained)
        model.fc = nn.Linear(2048, output_classes)
    elif model_name=="vgg11":
        model = torchvision.models.vgg11(
            pretrained=pretrained
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
        model.classifier[6] = nn.Linear(4096, output_classes)
    elif model_name=="vgg11_bn":
        model = torchvision.models.vgg11_bn(
            pretrained=pretrained
        )
        model.features[0] = nn.Conv2d(
            in_channels, 
            64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        model.features[3] = nn.Identity()
        model.classifier[6] = nn.Linear(4096, output_classes)
    elif model_name=="googlenet":
        model = torchvision.models.googlenet(
            pretrained=pretrained,
        )
        model.fc = nn.Linear(1024, output_classes)
        # these outputs are not considered 
        # model.aux1.fc2 = nn.Linear(1024, 10)
        # model.aux2.fc2 = nn.Linear(1024, 10)
    elif model_name=="xception":
        model = timm.create_model('xception', pretrained=pretrained)
        model.fc = nn.Linear(2048, output_classes)
    elif model_name=="inception_v4": 
        model = timm.create_model("inception_v4", pretrained=pretrained)
        model.last_linear = nn.Linear(1536, output_classes)
    elif model_name=="densenet121": 
        model = timm.create_model("densenet121", pretrained=pretrained)
        model.classifier = nn.Linear(1024, output_classes)
    elif model_name=="densenet201": 
        model = timm.create_model("densenet201", pretrained=pretrained)
        model.classifier = nn.Linear(1920, output_classes)
    elif model_name=="mobilenetv3_large_100": 
        model = timm.create_model("mobilenetv3_large_100", pretrained=pretrained)
        model.classifier = nn.Linear(1280, output_classes)
    elif model_name=="mobilenet_v3_small": 
        model = torchvision.models.mobilenet_v3_small(
            pretrained=pretrained,
        )
        model.classifier[3] = nn.Linear(1024, output_classes)
    elif model_name=="vit_base_patch16_224":
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        model.head = nn.Linear(768, output_classes)
    elif model_name=="condense_net_v1": 
        # use standard configurations from official repo ShichenLiu/CondenseNet
        args = Args(
            # data for dimension 
            data = data_name,
            # num classes
            num_classes = output_classes,
            # per layer depth
            stages = [14, 14, 14],
            # per layer growth 
            growth = [8, 16, 32],
            # 1x1 group convolution (default: 4)
            # NOTE: opacus doesn't support Conv2d groups != 1 or in_channels
            group_1x1 = 1, 
            # 3x3 group convolution (default: 4)
            # NOTE: opacus doesn't support Conv2d groups != 1 or in_channels
            group_3x3 = 1,
            # bottleneck (default: 4)
            bottleneck = 4,
            # condense factor (default: 4)
            condense_factor = 4,
            # drop out (default: 0)
            dropout_rate = 0,
        )
        model = condensenet(args)
    # by default a timm model is created
    else: 
        # TODO: try out different pretrainings 
        # TODO check that batchnorm and data normalization works fine.
        model = timm.create_model(model_name, pretrained=True)
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