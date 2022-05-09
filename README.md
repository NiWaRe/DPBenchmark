# DPBenchmark 
This is the WIP codebase of the ongoing research project on the benchmarking of different popular machine learning design choices in differentially private learning systems. This work mainly focuses on Image Classification and Segmentation on standard image datasets and medical datasets. Main tools used are Opacus, PyTorch Lightning and Weights&Biases.

Three important files: 
* `trainer.py` - lightningCLI with overloaded training functions and implemented hooks, and the main lightning module wrapper class to work with deepee DP.
* `data.py` - lightning datamodules adapted to work with DP.
* `models.py` - raw models. 

# Installation 
1. Create conda env with `conda env create -f setup/environment-yaml`
2. Download necessary data sources 
    * Or create symlink to shared data-folder with `ln -s /path/to/shared/data data`
3. Login to weights and biases
    * Login to wandb: `wandb login`
    * Start/Stop sync: `wandb online`/`wandb offline`
4. *Optional: Change Opacus BatchNorm Layers*
    * For example in *opacus/validators/batch_norm.py*: `nn.GroupNorm(math.gcd(32, module.num_features), module.num_features, affine=module.affine)`to `nn.GroupNorm(math.gcd(8, module.num_features), module.num_features, affine=module.affine)`.

# Usage
1. Configure `configs.yaml` 
2. Run `python trainer_lean.py`
3. It is compatible with sweeps over the L2-clip and number of epochs using the sweep functionality of weights&biases, which runs `python trainer_lean.py --L2_clip=1.0 --max_epochs=90` in the background. 
    * Under `sweeps/dp_params_sweep.yaml` the main sweep params are defined

# Experiments 
See under `/experiments` to see more information and also the most important values of the experiments.