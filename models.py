# torch 
import torchvision
from torch import nn

def get_model(name, pretrained, num_classes) -> nn.Module:
    model = None
    if name=="resnet18":
        if num_classes==1000:
            model = torchvision.models.resnet18(pretrained=pretrained)
        else:
            model = torchvision.models.resnet18(pretrained=pretrained, num_classes=num_classes)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            model.maxpool = nn.Identity()
    return model

# standard torchvision models are created and adapted in get_model 
# TODO: think about how to adapt models dynamically based on choice of dataset (not relevant for now)

# TODO: custom models are created in the following