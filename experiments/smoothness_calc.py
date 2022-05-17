import numpy as np
import torch 
from torch import nn
import timm
from torchvision import datasets, transforms
from pyhessian import hessian # Hessian computation
from density_plot import get_esd_plot # ESD plot
from pytorchcv.model_provider import get_model as ptcv_get_model # model

import sys
sys.path.append('../')
from lean.models import get_model
from lean.data import etl_data

from deepee import ModelSurgeon
from deepee.surgery import SurgicalProcedures
from functools import partial

import matplotlib.pyplot as plt

# # enable cuda devices
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

def getData(name='cifar10', data_root='/u/home/remersch/DPBenchmark/data', max_train_samples=0, train_bs=128, test_bs=1000):
    """
    Get the dataloader
    """

    # extract, transform, load data
    train_dataset, _, test_dataset = etl_data(
        data_name=name,
        root=data_root,
        val_split=0.0, # validation set is not necssary in this case
    )

    # in case we want to have a smaller data loader
    if max_train_samples: 
            # create shuffled indices
            rand_sub_ind = np.array(list(range(0, len(train_dataset))))
            np.random.shuffle(rand_sub_ind)
            rand_sub_ind = rand_sub_ind[:max_train_samples]
            # select shuffled subset
            train_dataset = torch.utils.data.Subset(train_dataset, rand_sub_ind)

    # create data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=train_bs,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
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

def get_model_simple(
    architecture = 'smoothnet', 
    data = 'cifar10',
    cuda = True,
): 
    # def get_model(
    #     model_name, 
    #     pretrained, 
    #     num_classes, 
    #     data_name, 
    #     kernel_size, 
    #     conv_layers,
    #     nr_stages, 
    #     depth, 
    #     width,
    #     halve_dim, 
    #     after_conv_fc_str, 
    #     activation_fc_str,
    #     skip_depth,
    #     dense,
    #     dsc
    # )
    surgeon = ModelSurgeon(partial(SurgicalProcedures.BN_to_GN, num_groups=8))
    if architecture=='densenet':
        model=get_model("densenet121", False, 10, data, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        model = surgeon.operate(model)
        dense_dict = torch.load(
            "./trained_models/CIFAR10_models/densenet121_gn8_cifar10_120e_eps7.pt" 
            if data=="cifar10" else
            "./trained_models/ImageNette_models/", 
            map_location=torch.device('cuda' if cuda else 'cpu'),
        )
        dense_dict = {k.replace("_module.", ""):v for k,v in dense_dict.items()}
        model.load_state_dict(dense_dict)
    elif architecture == 'smoothnet':
        model = get_model('en_scaling_residual_model', False, 10, data, 3, [16, 32], 1, 5, 8, True, 'mxp_gn', 'selu', 2, True, False)
        res_dict = torch.load(
            "./trained_models/CIFAR10_models/smoothnetw80d50_cifar10_180e_eps7.pt",
            map_location=torch.device('cuda' if cuda else 'cpu'),
        )
        res_dict = {k.replace("_module.", ""):v for k,v in res_dict.items()}
        model.load_state_dict(res_dict)
    elif architecture=='resnet34':
        model=get_model("resnet34", False, 10, "cifar10", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        model = surgeon.operate(model)
        model.fc = torch.nn.Linear(512, 10)
        res_dict = torch.load(
            "./trained_models/CIFAR10_models/resnet34_gn8_cifar10_120e_eps7.pt",
            map_location=torch.device('cuda' if cuda else 'cpu'),
        )
        res_dict = {k.replace("_module.", ""):v for k,v in res_dict.items()}
        model.load_state_dict(res_dict)
    elif architecture=='resnet18':
        model=get_model("resnet18", False, 10, data, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        model = surgeon.operate(model)
        res_dict = torch.load(
            "./trained_models/CIFAR10_models/resnet18_gn8_cifar10_180e_eps7.pt",
            map_location=torch.device('cuda' if cuda else 'cpu'),
        )
        res_dict = {k.replace("_module.", ""):v for k,v in res_dict.items()}
        model.load_state_dict(res_dict)
    elif architecture=='resnet9':
        model=get_model("resnet9", False, 10, data, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        model = surgeon.operate(model)
        res_dict = torch.load(
            "./trained_models/CIFAR10_models/resent9_gn8_cifar10_120e_eps7.pt",
            map_location=torch.device('cuda' if cuda else 'cpu'),
        )
        res_dict = {k.replace("_module.", ""):v for k,v in res_dict.items()}
        model.load_state_dict(res_dict)
    elif architecture=='efficientnetb0':
        model = timm.create_model("efficientnet_b0")
        model.classifier = nn.Linear(1280, 10)
        model = surgeon.operate(model)
        eff_dict = torch.load(
            "./trained_models/CIFAR10_models/efficientnetb0_gn8_cifar10_180e_eps7.pt",
            map_location=torch.device('cuda' if cuda else 'cpu'),
        )
        eff_dict = {k.replace("_module.", ""):v for k,v in eff_dict.items()}
        model.load_state_dict(eff_dict)

    if cuda:
        # torch.cuda.set_device(0)
        model = model.cuda()

    return model


def calc_ev(
    architecture = 'smoothnet',
    data = 'cfiar10',
    data_root = './data',
    train_bs = 512,
    max_train_samples = 4096,
    max_iter = 100,
    cuda = True,
    ): 

    # get model
    model = get_model_simple(architecture, data, cuda)

    # create loss function
    criterion = torch.nn.CrossEntropyLoss()

    # get dataset 
    train_loader, test_loader = getData(name=data, data_root=data_root, max_train_samples=max_train_samples, train_bs=train_bs)

    # create the hessian computation module 
    hessian_comp = hessian(model, criterion, dataloader=train_loader, cuda=cuda)

    # Now let's compute the top 2 eigenavlues and eigenvectors of the Hessian
    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)
    print("The top two eigenvalues of this model are: %.4f %.4f"% (top_eigenvalues[-1],top_eigenvalues[-2]))

    # save the top ev
    torch.save(top_eigenvector, './smoothness_calc/'+data+'/'+architecture+'_evecs_'+str(max_train_samples)+'_'+str(max_iter)+'.pt')
    torch.save(top_eigenvalues, './smoothness_calc/'+data+'/'+architecture+'_evals_'+str(max_train_samples)+'_'+str(max_iter)+'.pt')

    print("Calculated and saved EVs of "+ architecture)

def calc_trace(
    architecture = 'smoothnet',
    data = 'cfiar10',
    data_root = './data',
    train_bs = 512,
    max_train_samples = 4096,
    max_iter = 100,
    cuda = True,
    ): 

    # get model
    model = get_model_simple(architecture, cuda)

    # create loss function
    criterion = torch.nn.CrossEntropyLoss()

    # get dataset 
    train_loader, test_loader = getData(name=data, data_root=data_root, max_train_samples=max_train_samples, train_bs=train_bs)

    # create the hessian computation module 
    hessian_comp = hessian(model, criterion, dataloader=train_loader, cuda=cuda)

    # calculate the trace
    trace = hessian_comp.trace(maxIter=100) # 100 is default
    print("The trace of this model is: %.4f"%(np.mean(trace)))

    # save the top ev
    torch.save(trace, './smoothness_calc/'+data+'/'+architecture+'_trace_'+str(max_train_samples)+'_'+str(max_iter)+'.pt')

    print("Calculated and saved trace of "+ architecture)

# TODO: resolve efficientnet problem + make this script modular, same as main training script
# torch.random.manual_seed(1302)
# rseed(1302)
# np.random.seed(1302)
if __name__ == "__main__":
    models = ['resnet9', 'resnet18', 'densenet', 'smoothnet'] # 'efficientnetb0' CUDA memory access error, # done: 'resnet34',   
    data = 'imagenette' # alternative: cifar10
    cuda = True
    for model in models: 
        # calculate 1st and 2nd eval and evec of hessian
        calc_ev(
            architecture = model,
            data = data,
            data_root = '/u/home/remersch/DPBenchmark/data',
            train_bs = 64,
            max_train_samples = 4096,
            max_iter = 100,
            cuda = cuda,
        )

        # calculate the hessian trace
        calc_trace(
            architecture = model,
            data = data,
            data_root = '/u/home/remersch/DPBenchmark/data',
            train_bs = 64,
            max_train_samples = 4096,
            max_iter = 100,
            cuda = cuda,
        )
  