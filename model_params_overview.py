from timm import create_model
from models import get_model
from torch import onnx, rand, nn

from deepee import ModelSurgeon
from deepee.surgery import SurgicalProcedures
from functools import partial

def num_non_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad )
def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    surgeon = ModelSurgeon(partial(SurgicalProcedures.BN_to_GN, num_groups=8))
    print("CIFAR-10")
    resnet18=get_model("resnet18", False, 10, "cifar10", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    print(f"ResNet-18: \t Trainable: {num_trainable_parameters(resnet18):,}\tNon-trainable: {num_non_trainable_parameters(resnet18):,}")
    resnet34=get_model("resnet34", False, 10, "cifar10", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    resnet34 = surgeon.operate(resnet34)
    resnet34.fc = nn.Linear(512, 10)
    print(f"ResNet-34: \t Trainable: {num_trainable_parameters(resnet34):,}\tNon-trainable: {num_non_trainable_parameters(resnet34):,}")
    resnet50=get_model("resnet50", False, 10, "cifar10", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    print(f"ResNet-50: \t Trainable: {num_trainable_parameters(resnet50):,}\tNon-trainable: {num_non_trainable_parameters(resnet50):,}")
    densenet121=get_model("densenet121", False, 10, "cifar10", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    print(f"DenseNet-121: \t Trainable: {num_trainable_parameters(densenet121):,}\tNon-trainable: {num_non_trainable_parameters(densenet121):,}")
    # smoothnet = get_model("en_scaling_residual_model", False, 10, "cifar10", 0, 0,0,8, 5, False, "max_pool", "selu", 0, True,  False)
    smoothnet = get_model('en_scaling_residual_model', False, 10, 'cifar10', 3, [16, 32], 1, 5, 8, True, 'mxp_gn', 'selu', 2, True, False)
    # smoothnet = get_model('en_scaling_residual_model', False, 10, 'cifar10', 3, [16, 32], 1, 8.0, 5.0, True, 'mxp_gn', 'selu', 2, True, False)
    print(f"SmoothNet: \t Trainable: {num_trainable_parameters(smoothnet):,}\tNon-trainable: {num_non_trainable_parameters(smoothnet):,}")
    onnx.export(smoothnet, rand((1, 3, 32, 32)), "smoothnet_cifar.onnx", input_names=["Image"], output_names=["Prediction"])


    print("\n\nImageNette")
    resnet18=get_model("resnet18", False, 10, "imagenette", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    print(f"ResNet-18: \t Trainable: {num_trainable_parameters(resnet18):,}\tNon-trainable: {num_non_trainable_parameters(resnet18):,}")
    resnet34=get_model("resnet34", False, 10, "imagenette", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    resnet34 = surgeon.operate(resnet34)
    resnet34.fc = nn.Linear(512, 10)
    print(f"ResNet-34: \t Trainable: {num_trainable_parameters(resnet34):,}\tNon-trainable: {num_non_trainable_parameters(resnet34):,}")
    resnet50=get_model("resnet50", False, 10, "imagenette", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    print(f"ResNet-50: \t Trainable: {num_trainable_parameters(resnet50):,}\tNon-trainable: {num_non_trainable_parameters(resnet50):,}")
    densenet121=get_model("densenet121", False, 10, "imagenette", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    print(f"DenseNet-121: \t Trainable: {num_trainable_parameters(densenet121):,}\tNon-trainable: {num_non_trainable_parameters(densenet121):,}")
    efficientnet = create_model("efficientnet_b0")
    efficientnet.classifier = nn.Linear(1280, 10)
    model = surgeon.operate(efficientnet)
    print(f"EfficientNet-B0: \t Trainable: {num_trainable_parameters(efficientnet):,}\tNon-trainable: {num_non_trainable_parameters(efficientnet):,}")
    # smoothnet = get_model("en_scaling_residual_model", False, 10, "imagenette", 0, 0,0,8, 5, False, "max_pool", "selu", 0, True,  False)
    smoothnet = get_model('en_scaling_residual_model', False, 10, 'imagenette', 3, [16, 32], 1, 5, 8, True, 'mxp_gn', 'selu', 2, True, False)
    # smoothnet = get_model('en_scaling_residual_model', False, 10, 'imagenette', 3, [16, 32], 1, 8.0, 5.0, True, 'mxp_gn', 'selu', 2, True, False)
    print(f"SmoothNet: \t Trainable: {num_trainable_parameters(smoothnet):,}\tNon-trainable: {num_non_trainable_parameters(smoothnet):,}")

    print(smoothnet)
    onnx.export(smoothnet, rand((1, 3, 160, 160)), "smoothnet_imagenette.onnx", input_names=["Image"], output_names=["Prediction"])