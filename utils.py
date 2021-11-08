import torch
from torch import nn, Size
from typing import Union

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
    elif after_conv_fc_str == 'identity': 
        after_conv_fc = nn.Identity()
    
    return after_conv_fc