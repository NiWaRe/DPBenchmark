#!/usr/bin/env python

"""
New Opacus Library without PyTorch Lightning.
"""

# general and logging
import wandb
import os
import argparse
import yaml
from tqdm import tqdm

# general torch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

# data
from lean.data import etl_data 

# model 
from lean.models import get_model

# utility functions
from lean.utils import get_grad_norm, initialize_weight, normalize_weight

# opacus
from opacus.validators import ModuleValidator
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

###################
# TRAIN FUNCTIONS #
###################

def train(epoch, model, train_loader, optimizer, lr_scheduler, criterion, privacy_engine, config): 
    """
        Train for one epoch.
    """
    model.train()
    with BatchMemoryManager(
        data_loader=train_loader, 
        max_physical_batch_size=config.physical_batch_size, 
        optimizer=optimizer
    ) as memory_safe_data_loader:
        for i, (images, labels) in enumerate(
                tqdm(memory_safe_data_loader, desc="Training iterations/epochs")
            ):
            # shift to device
            images = Variable(images).to(config.device)
            labels = Variable(labels).to(config.device)

            # standard training loop block
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()

            ggn = get_grad_norm(model)
            optimizer.step()

            if i % config.print_every_iter == 0:
                # TODO: add accuracy per class
                # for label in range(10):
                #     metrics['Accuracy ' + label_names[label]] = correct_arr[label] / total_arr[label]

                metrics = {
                    'train_loss': loss, 
                    'lr-SGD': lr_scheduler.get_last_lr()[0],
                    'global grad norm': ggn,     
                }
                if config.dp: 
                    metrics['spent_epsilon'] = privacy_engine.get_epsilon(config.target_delta)
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Train Loss: {loss:.6f} "
                    f"lr-SGD: {metrics['lr-SGD']:.6f} "
                    f"(ε = {metrics['spent_epsilon']:.2f}, δ = {config.target_delta})" if config.dp else ""
                )
                wandb.log(metrics)

        # after every epoch do a learning rate step 
        lr_scheduler.step()
    
def test(model, test_loader, criterion, config, test): 
    """
        Test for given data - either with test or validation dataset.
    """
    model.eval()

    # validate after each epoch
    correct = 0.0
    correct_arr = [0.0] * config.num_classes
    total = 0.0
    total_arr = [0.0] * config.num_classes

    # iterate through validation dataset
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Validation"):
            # shift to device
            images = Variable(images).to(config.device)
            labels = Variable(labels).to(config.device)

            # forward pass only to get logits/output
            outputs = model(images)

            # track validation loss
            val_loss = criterion(outputs, labels)

            # get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            # total number of labels
            total += labels.size(0)
            correct += (predicted == labels).sum().detach().cpu()

            # TODO: add accuracy per class
            # for label in range(10):
            #     correct_arr[label] += (((predicted == labels) & (labels==label)).sum())
            #     total_arr[label] += (labels == label).sum()

    val_accuracy = correct / total
    if test: 
        metric_acc = "val_acc"
        metric_loss = "val_loss"
    else: 
        metric_acc = "test_acc"
        metric_loss = "test_loss"
    metrics = {
        metric_acc: val_accuracy, 
        metric_loss: val_loss,  
    }
    print("Testing" if test else "Validating")
    print(f'Loss: {val_loss} Accuracy: {val_accuracy}')
    wandb.log(metrics)  

#################
# MAIN FUNCTION #
#################

def main(project_name, experiment_name, config):
    # log config 
    wandb.init(config=config, name=experiment_name, project=project_name)
    config = wandb.config

    ########
    # DATA #
    ########
    # extract, transform, load data
    train_dataset, validation_dataset, test_dataset = etl_data(
        data_name=config.data_name,
        root=config.data_root,
        val_split=config.val_split,
    )

    # TODO: can be shifted to the data.py file as well | stop shuffling for DP
    # create data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=False if config.dp else True)

    val_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config.batch_size,
                                              shuffle=False)

    #########
    # MODEL #
    #########
    # build model based on params
    model = get_model(
        model_name=config.model_name, 
        pretrained=config.pretrained, 
        num_classes=config.num_classes, 
        data_name=config.data_name, 
        kernel_size=config.kernel_size,
        conv_layers=config.conv_layers, 
        nr_stages=config.nr_stages,
        depth=config.depth, 
        width=config.width, 
        halve_dim=config.halve_dim, 
        after_conv_fc_str=config.after_conv_fc_str, 
        activation_fc_str=config.activation_fc_str,
        skip_depth=config.skip_depth,
        dense=config.dense,
        dsc=config.dsc,
    )

    # validate model, with or without DP, to ensure comparability
    # NOTE: check opacus/validators to see what checks and fixes have been introduced.
    # BatchNorm is replaced by GroupNorm[8] by default - check batch_norm.py to change.
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)

    # add weight init
    if config.weight_init:
        model.apply(initialize_weight)

    # add weight norm
    if config.weight_norm: 
        model.apply(normalize_weight)

    # shift to CUDA
    if config.device == 'cuda:0' and torch.cuda.device_count() > 1:
        print("Using multiple GPUs: ", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model = model.to(config.device)

    # track model with wandb
    wandb.watch(model)
    wandb.config.model_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.model_params_total = sum(p.numel() for p in model.parameters())

    ############
    # TRAINING #
    ############
    # define loss
    criterion = nn.CrossEntropyLoss()

    # define optimizer and learning rate scheduler
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=config.lr, 
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    if config.lr_scheduler:
        lr_scheduler = StepLR(
                optimizer, 
                step_size=5 if config.dp else 10, #1, 10
                gamma=0.9 if config.dp else 0.7, #0.9 for DP, 0.7 for no-DP 
        )

    # change privacy settings if necessary 
    if config.dp:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=config.max_epochs,
            target_epsilon=config.target_epsilon,
            target_delta=config.target_delta,
            max_grad_norm=config.L2_clip,
        )
        wandb.config.noise_multiplier = optimizer.noise_multiplier
        print(f"Using sigma={optimizer.noise_multiplier} and C={config.L2_clip}")

    # training loop
    for epoch in tqdm(range(config.max_epochs), desc="Epochs"):
        train(epoch, model, train_loader, optimizer, lr_scheduler, criterion, privacy_engine, config)
        test(model, val_loader, criterion, config, test=False)
    
    # test on real training set after training
    test(model, test_loader, criterion, config, test=True)

    ##############
    # SAVE MODEL #
    ##############
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))

if __name__ == '__main__':
    # load in YAML configuration
    config = {}
    base_config_path = 'lean/config.yaml'
    with open(base_config_path, 'r') as file:
        config.update(yaml.safe_load(file))

    # TODO: add more if more parameters should be "sweepable"
    # overwrite with sweep parameters - have to be given in with ArgumentParser for wandb
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--L2_clip', type=float, default=config['L2_clip'], help='L2 clip for DP')
    parser.add_argument('--max_epochs', type=float, default=config['max_epochs'], help='Max epochs to train')
    args = parser.parse_args()

    # TODO: check for easy way to convert args to dict to simply update config
    config['L2_clip'] = args.L2_clip
    config['max_epochs'] = args.max_epochs

    # start training with name and config 
    main(config['project_name'], config['experiment_name'], config)

# [x] 1. Do Validation Data
# [x] 2. Test with WandB Monitoring
# [x] 3. Add Opacus
# [x] 4. Test with WandB Sweep
# [ ] 5. Example pictures and model checkpoint in wandb

## issues 
# step size the same?
# patience, divergence checks not implemented
# model.eval() in other would work as well (val loader error)? last time val loader was also data loader.
# update torch? (to not get warning)
