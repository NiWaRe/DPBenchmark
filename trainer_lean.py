#!/usr/bin/env python

"""
New Opacus Library without PyTorch Lightning.
"""

# general and logging
import wandb
import os
import argparse
import yaml

# general torch
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F

# data
from lean.data import etl_data 

# TODO: for now we're using the same model factory as for the old file
# model 
from models import get_model

#################
# MAIN FUNCTION #
#################

def main(project_name, config):
    # log config 
    wandb.init(config=config, project=project_name)
    config = wandb.config

    ########
    # DATA #
    ########
    # extract, transform, load data
    train_dataset, test_dataset = etl_data()

    # TODO: can be shifted to the data.py file as well
    # create data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
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

    # shift to CUDA
    model = model.to(config.device)

    # track model with wandb
    wandb.watch(model)

    ############
    # TRAINING #
    ############
    # define loss
    criterion = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # training loop
    iter = 0
    for epoch in range(config.max_epochs):
        for i, (images, labels) in enumerate(train_loader):

            images = Variable(images).to(config.device)
            labels = Variable(labels).to(config.device)

            # standard training loop block
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # TODO: make more adaptable for other datasets (only track epochs)
            iter += 1
            if iter % 100 == 0:
                # Calculate Accuracy
                correct = 0.0
                correct_arr = [0.0] * 10
                total = 0.0
                total_arr = [0.0] * 10

                # Iterate through test dataset
                for images, labels in test_loader:
                    images = Variable(images).to(config.device)

                    # Forward pass only to get logits/output
                    outputs = model(images)

                    # Get predictions from the maximum value
                    _, predicted = torch.max(outputs.data.detach().cpu(), 1)

                    # Total number of labels
                    total += labels.size(0)
                    correct += (predicted == labels).sum()

                    # TODO: add accuracy per class
                    # for label in range(10):
                    #     correct_arr[label] += (((predicted == labels) & (labels==label)).sum())
                    #     total_arr[label] += (labels == label).sum()

                accuracy = correct / total
                metrics = {'accuracy': accuracy, 'loss': loss}

                # TODO: add accuracy per class
                # for label in range(10):
                #     metrics['Accuracy ' + label_names[label]] = correct_arr[label] / total_arr[label]

                wandb.log(metrics)

                # Print Loss
                print('Iteration: {0} Loss: {1:.2f} Accuracy: {2:.2f}'.format(iter, loss, accuracy))

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
    project_name = "dp_benchmark"
    main(project_name, config)

# 1. Do Validation Data
# 2. Test with WandB Monitoring
# 3. Add Opacus
# 3. Test with WandB Sweep
