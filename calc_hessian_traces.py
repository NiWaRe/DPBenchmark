# %%
import numpy as np
import torch 
from torchvision import datasets, transforms
from pyhessian import hessian # Hessian computation
from density_plot import get_esd_plot # ESD plot
from pytorchcv.model_provider import get_model as ptcv_get_model # model
from models import get_model
from data import ImageNetteDataClass
from deepee import ModelSurgeon
from deepee.surgery import SurgicalProcedures
from functools import partial
import timm
import pandas as pd
from tqdm import tqdm
from random import seed as rseed
from argparse import ArgumentParser

import matplotlib.pyplot as plt
# enable cuda devices
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

torch.random.manual_seed(1302)
rseed(1302)
np.random.seed(1302)
# %%
parser = ArgumentParser()
parser.add_argument("--data_name", type=str, required=True, choices=["cifar10", "imagenette"], help='Which dataset')
parser.add_argument("--iterations", type=int, required=True, help="Max Iterations for hessian trace power iteration")
parser.add_argument("--num_samples", type=int, required=True, help='Number of samples to calculate hessian')
parser.add_argument("--no_cuda", action='store_false', help='Refrain from using cuda')
parser.add_argument("--compute_batch_size", type=int, default=32, help='Virtual batch size to calculate hessian')
args = parser.parse_args()

# %%
def getData(name='cifar10', train_bs=128, test_bs=1000):
    """
    Get the dataloader
    """
    if name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR10(root='../data',
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=train_bs,
                                                   shuffle=True)

        testset = datasets.CIFAR10(root='../data',
                                   train=False,
                                   download=False,
                                   transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=test_bs,
                                                  shuffle=False)
    elif name == 'cifar10_without_dataaugmentation':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR10(root='../data',
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=train_bs,
                                                   shuffle=True)

        testset = datasets.CIFAR10(root='../data',
                                   train=False,
                                   download=False,
                                   transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=test_bs,
                                                  shuffle=False)
    elif name == 'imagenette':
        imagenette_transforms = transforms.Compose(
                [
                    # NOTE: change dimension if other dataset!
                    transforms.Resize(224 + 32),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
        )
        trainset = ImageNetteDataClass("./imagenette", train=True, transform=imagenette_transforms)
        testset = ImageNetteDataClass("./imagenette", train=False, transform=imagenette_transforms)

        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=train_bs,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=test_bs,
                                                  shuffle=False)

    return train_loader, test_loader


def test(model, test_loader, cuda=True):
    """
    Get the test performance
    """
    model.eval()
    correct = 0
    total_num = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.data.max(
            1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        total_num += len(data)
    print('testing_correct: ', correct / total_num, '\n')
    return correct / total_num
# %%
surgeon = ModelSurgeon(partial(SurgicalProcedures.BN_to_GN, num_groups=8))
# get the model 
models = {}
data_name = args.data_name
if data_name == 'imagenette':
    densenet121=get_model("densenet121", False, 10, "imagenette", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    densenet121 = surgeon.operate(densenet121)
    dense_dict = torch.load("/home/alex/DPBenchmark/densenet121_gn_8_imagenette_eps7.pt")
    dense_dict = {k.replace("_module.", ""):v for k,v in dense_dict.items()}
    densenet121.load_state_dict(dense_dict)
    models["densenet121"] = densenet121
    smoothnet = get_model('en_scaling_residual_model', False, 10, 'imagenette', 3, [16, 32], 1, 5, 8, True, 'mxp_gn', 'selu', 2, True, False)
    smooth_dict = {k.replace("_module.", ""):v for k,v in torch.load("smoothnetw80d50_imagenette_eps7.pt", map_location='cpu').items()}
    smoothnet.load_state_dict(smooth_dict) 
    models["smoothnet"] = smoothnet
    resnet34=get_model("resnet34", False, 10, "imagenette", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    resnet34 = surgeon.operate(resnet34)
    resnet34.fc = torch.nn.Linear(512, 10)
    res_dict = torch.load("/home/alex/DPBenchmark/resnet34_gn8_imagenette_eps7.pt")
    res_dict = {k.replace("_module.", ""):v for k,v in res_dict.items()}
    resnet34.load_state_dict(res_dict)
    models["resnet34"] = resnet34
    efficientnetb0 = timm.create_model("efficientnet_b0")
    # model=get_model("efficientnet_b0", False, 10, "imagenette", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    # NOTE: b0 was 1280, b3 was 1536, b5 was 2048, b7 was 2560
    efficientnetb0.classifier = torch.nn.Linear(1280, 10)
    efficientnetb0 = surgeon.operate(efficientnetb0)
    eff_dict = torch.load("efficientb0_gn8_imagenette_eps7.pt")
    eff_dict = {k.replace("_module.", ""):v for k,v in eff_dict.items()}
    efficientnetb0.load_state_dict(eff_dict)
    models["efficientnetb0"] = efficientnetb0


    # compute_batch_size = 12
    # iterations = 128
    # num_samples = 256
    # cuda = True
elif data_name == 'cifar10':

    # resnet18=get_model("resnet18", False, 10, "cifar10", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    resnet34=get_model("resnet34", False, 10, "cifar10", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    resnet34 = surgeon.operate(resnet34)
    resnet34.fc = torch.nn.Linear(512, 10)
    rd = torch.load("/home/alex/DPBenchmark/resnet34_gn8_cifar10_eps7.pt", map_location='cpu')
    rd = {k.replace("_module.", ""):v for k,v in rd.items()}
    resnet34.load_state_dict(rd)
    models["resnet34"]=resnet34
    densenet121=get_model("densenet121", False, 10, "cifar10", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    densenet121 = surgeon.operate(densenet121)
    dense_dict = torch.load("densenet121_gn8_cifar10_eps7.pt")
    dense_dict = {k.replace("_module.", ""):v for k,v in dense_dict.items()}
    densenet121.load_state_dict(dense_dict)
    models["densenet121"]=densenet121
    # resnet50=get_model("resnet50", False, 10, "cifar10", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    # densenet121=get_model("densenet121", False, 10, "cifar10", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    # smoothnet = get_model("en_scaling_residual_model", False, 10, "cifar10", 0, 0,0,8, 5, False, "max_pool", "selu", 0, True,  False)
    smoothnet = get_model('en_scaling_residual_model', False, 10, 'cifar10', 3, [16, 32], 1, 5, 8, True, 'mxp_gn', 'selu', 2, True, False)
    sd = torch.load("/home/alex/DPBenchmark/smoothnet_w80d50_cifar10_eps7.pt", map_location='cpu')
    sd = {k.replace("_module.", ""):v for k,v in sd.items()}
    smoothnet.load_state_dict(sd)
    models["smoothnet"] = smoothnet

# compute_batch_size = 128
#     iterations = 128
#     num_samples = 10000
#     cuda = True
    
# %%
# %%
# create loss function
criterion = torch.nn.CrossEntropyLoss()

# get dataset 
train_loader, test_loader = getData(train_bs=1, name=data_name)

# for illustrate, we only use one batch to do the tutorial
input_targets = []
for i, (inputs, targets) in enumerate(train_loader):
    if i>=args.num_samples:
        break
    input_targets.append((inputs.squeeze(), targets.squeeze()))
class SmallDataset(torch.utils.data.Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)
# if data_name == 'cifar10':
#     pass
# elif data_name == 'imagenette':
train_loader = torch.utils.data.DataLoader(SmallDataset(input_targets), batch_size=args.compute_batch_size)
    # data = next(iter(smallDataloader))

# %%
results = {}
for name, model in tqdm(models.items(), total=len(models), desc='Models:', leave=True):
    # we use cuda to make the computation fast
    traces = []
    # for input, targets in tqdm(input_targets, total=len(input_targets), desc="Iterations", leave=False):
    if not args.no_cuda:
        model = model.cuda()
        inputs, targets = inputs.cuda(), targets.cuda()
    # hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=cuda)
    hessian_comp = hessian(model, criterion, dataloader=train_loader, cuda=not args.no_cuda)
        # top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(maxIter=iterations)
        # top_eigenvalues = top_eigenvalues[-1]
        # model_results["TopEigenvalue":top_eigenvalues]
        # print(f"The top Hessian eigenvalue of {name} is {top_eigenvalues:.4f}\t(iter={iterations})")
    trace = np.mean(hessian_comp.trace(maxIter=args.iterations))
        # traces.append(trace)
        # print(f"Trace for {name}: {trace:.2f}\t(iter={iterations})")

    results[name] = {"Trace":trace}
df = pd.DataFrame.from_dict(results, orient="columns")
df.to_csv(f"smoothness_{args.iterations}iter_{args.num_samples}samples_{data_name}.csv", index=False)
print(df)

    


