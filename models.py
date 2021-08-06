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

def get_model(name, pretrained, num_classes, data_name) -> nn.Module:
    model = None
    if name=="resnet18":
        if data_name=="cifar10":
            model = torchvision.models.resnet18(pretrained=pretrained, num_classes=num_classes)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            model.maxpool = nn.Identity()
        elif data_name=="mnist":
            model = torchvision.models.resnet18(pretrained=pretrained, num_classes=num_classes)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            model.maxpool = nn.Identity()
        else:
            model = torchvision.models.resnet18(pretrained=pretrained)
    elif name=="vgg11":
        if data_name=="mnist":
            model = torchvision.models.vgg11(pretrained=pretrained, num_classes=num_classes)
            model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            model.features[2] = nn.Identity()
        else:
            model = torchvision.models.vgg11(pretrained=pretrained, num_classes=num_classes)

    # TODO: surgical procedures have to be done automatically depending on model.
    #model = ModelSurgeon(SurgicalProcedures.BN_to_GN).operate(model)
    return model

# standard torchvision models are created and adapted in get_model 
# TODO: think about how to adapt models dynamically based on choice of dataset (not relevant for now)

# TODO: custom models are created in the following