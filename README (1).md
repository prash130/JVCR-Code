## Training Setup Expectations

#### Expected Directory Structure:
```
{project}
  |___ data/ 
  |___ logs/ (--> /tmp/{project}/logs)
  |___ models/ 
        |___ tmp (--> /tmp/{project}/model)
        |___ best
        |___ latest
  |___ src/
  |___ requirements.txt
  |___ setup.sh
  |___ purge.sh
  |___ train.py
  |___ run.sh
  |___ README.md
```

##### Desciption

* `data` - training data. 
* `logs` - tensorboard and program logs. This should be a softlink to `/tmp/{project}/logs`
* `models/tmp` - intermediate checkpoints. This should be a softlink to `/tmp/{project}/model`
* `models/best` - best checkpoint yet.
* `models/latest` - latest checkpoint. This is loaded while resuming the training.
* `src/` - contains source code
* `requirements.txt` - required python packages
* `setup.sh` - data download and conda environment setup script. This script should create a new conda environment under your project name and using `requirements.txt` install all necessary packages. This should download all the necessary training data to `data/` directory. Create appropriate softlinks.
* `purge.sh` - running this script should remove the conda environment and all the temporary files.
* `train.py` - main python file which uses`argparse` to control all aspects of training and logistics. This should enable,
  * Ability to resume training using latest checkpoint. (controlled by argument)
  * Ability to change hyperparmater values.

* `run.sh` - this should do the following
  * run `setup.sh`
  * activate appropriate conda env
  * start training in background with appropriate arguments and dump logs to `logs/`
  * run tensorboard in background on `logs/`.
* `README.sh` - Ideally it should be this document 😜. Any specific set of instructions and good to know information should go in this.


_Notes:_

1. Code to be shared on github
2. Any visualizations/stats to be analyzed should be written to tensorboard.
3. [Using Tensorboard with PyTorch](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard)
4. Please test your scripts for conda env creation/deletion, temporary data creation/deletion. 
5. Your code will be run on CUDA 9.2 and CUDNN 7.1. If you need to run the code on any specific version of CUDA, please let me know

