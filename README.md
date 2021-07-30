# dp_benchmark
Three important files: 
* `trainer.py` - lightningCLI with overloaded training functions and implemented hooks, and the main lightning module wrapper class to work with deepee DP.
* `data.py` - lightning datamodules adapted to work with DP.
* `models.py` - raw models. 