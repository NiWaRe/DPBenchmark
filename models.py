# torch 
from torch.utils import data
import torchvision
from torch import nn

# differential privacy
from deepee import (PrivacyWrapper, PrivacyWatchdog, UniformDataLoader,
                     ModelSurgeon, SurgicalProcedures, watchdog)

def get_model(name, pretrained, num_classes, data_name) -> nn.Module:
    model = None
    if data_name=="resnet18":
        if data_name=="cifar10":
            model = torchvision.models.resnet18(pretrained=pretrained)
        else:
            model = torchvision.models.resnet18(pretrained=pretrained, num_classes=num_classes)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            model.maxpool = nn.Identity()
    elif name=="vgg11":
        model = torchvision.models.vgg11(pretrained=pretrained, num_classes=num_classes)

    # TODO: temp model surgeries happen here 
    model = ModelSurgeon(SurgicalProcedures.BN_to_GN).operate(model)
    return model

# standard torchvision models are created and adapted in get_model 
# TODO: think about how to adapt models dynamically based on choice of dataset (not relevant for now)

# TODO: custom models are created in the following