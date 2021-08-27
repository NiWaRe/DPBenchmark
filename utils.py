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