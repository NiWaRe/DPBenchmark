from typing import Union

import torch
from torch import nn, Size
from torch.nn.modules import activation

from deepee import SurgicalProcedures
from data import (
    MNISTDataModuleDP,
    CIFAR10DataModuleDP 
)

# TODO: think of more elegant way of making them available natively

# to make surgical procedures are available as part of a module,
# so that they can be passed to Callable types from the configs.yaml

def BN_to_IN(module: nn.modules.batchnorm._BatchNorm) -> nn.Module:
    return SurgicalProcedures.BN_to_IN(module)

def BN_to_BN_nostats(module: nn.modules.batchnorm._BatchNorm) -> nn.Module:
    return SurgicalProcedures.BN_to_BN_nostats(module)

def IN_to_IN_nostats(module: nn.modules.instancenorm._InstanceNorm) -> nn.Module:
    return SurgicalProcedures.IN_to_IN_nostats(module)

def BN_to_GN(
    module: nn.modules.batchnorm._BatchNorm, num_groups: Union[str, int] = "default"
) -> nn.Module:
    return SurgicalProcedures.BN_to_GN(module, num_groups)

def BN_to_LN(
    module: nn.Module,
    normalized_shape: Union[int, list, Size],
) -> nn.Module:
    return SurgicalProcedures.BN_to_LN(module, normalized_shape)

# to make datamodules available as part of a module,
# so that they can be passed to LightningDataModule types from the configs.yaml
def MNISTDataModuleDPClass():
    return MNISTDataModuleDP
    
def CIFAR10DataModuleDPClass():
    return CIFAR10DataModuleDP

# other utility functions
def get_grad_norm(model, norm_type=2):
    parameters = [
        p for p in model.parameters() if p.grad is not None and p.requires_grad
    ]
    if len(parameters) == 0:
        total_norm = 0.0
    else:
        device = parameters[0].grad.device
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            2.0,
        ).item()
    return total_norm

# used in models.py
# Utility Modules, Functions
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

def getActivationFunction(
        activation_fc_str : str = "selu"
    ): 
    """
    This is a helper function to return all the different activation functions
    we want to consider. This is written in a dedicated function because
    it is called from different classes and because this is the central place
    where all possible activation_fc are listed.
    """
    if activation_fc_str == "selu": 
        activation_fc = nn.SELU()
    elif activation_fc_str == "relu": 
        activation_fc = nn.ReLU()
    elif activation_fc_str == "leaky_relu":
        activation_fc = nn.LeakyReLU()

    return activation_fc


def getAfterConvFc(
        after_conv_fc_str : str, 
        num_features : int,
        **kwargs
    ): 
    """
    This is a helper function to return all the different after_conv_fcs
    we want to consider. This is written in a dedicated function because
    it is called from different classes and because this is the central place
    where all possible after_conv_fct are listed.

    Args: 
        after_conv_fc_str: str to select the specific function
        num_features: only necessary for norms 

    """
    if after_conv_fc_str == 'batch_norm':
        after_conv_fc = nn.BatchNorm2d(
            num_features=num_features
        )
    elif after_conv_fc_str == 'group_norm':
        # for num_groups = num_features => LayerNorm
        # for num_groups = 1 => InstanceNorm
        after_conv_fc = nn.GroupNorm(
            num_groups=min(8, num_features), 
            num_channels=num_features, 
            affine=True
        )
    elif after_conv_fc_str == 'instance_norm':
        after_conv_fc = nn.InstanceNorm2d(
            num_features=num_features,
        )
    elif after_conv_fc_str == 'max_pool': 
        # keep dimensions for CIFAR10 dimenions assuming a downsampling 
        # only through halving. 
        after_conv_fc = nn.MaxPool2d(
            kernel_size=3, 
            stride=1, 
            padding=1
        )
    elif after_conv_fc_str == 'avg_pool': 
        # keep dimensions for CIFAR10 dimenions assuming a downsampling 
        # only through halving. 
        after_conv_fc = nn.AvgPool2d(
            kernel_size=3, 
            stride=1, 
            padding=1
        )
    elif after_conv_fc_str == 'max_pool_gn': 
        # keep dimensions for CIFAR10 dimenions assuming a downsampling 
        # only through halving. 
        after_conv_fc = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=3, 
                stride=1, 
                padding=1
            ), 
            nn.GroupNorm(
                num_groups=min(8, num_features), 
                num_channels=num_features, 
                affine=True
            )
        )
    elif after_conv_fc_str == 'identity': 
        after_conv_fc = nn.Identity()
    
    return after_conv_fc

def initialize_weight(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain("selu"))
        # nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
    # elif isinstance(module, nn.GroupNorm):
    #     nn.init.xavier_uniform_(module.weight, 1)
    #     nn.init.xavier_uniform_(module.bias, 0)
        
def normalize_weight(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)): 
        nn.utils.weight_norm(module, name='weight')