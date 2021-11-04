# DPBenchmark 
This is the WIP codebase of the ongoing research project on the benchmarking of different popular machine learning design choices in differentially private learning systems. This work mainly focuses on Image Classification and Segmentation on standard image datasets and medical datasets. 

Three important files: 
* `trainer.py` - lightningCLI with overloaded training functions and implemented hooks, and the main lightning module wrapper class to work with deepee DP.
* `data.py` - lightning datamodules adapted to work with DP.
* `models.py` - raw models. 
