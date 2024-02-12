<div align="center">

# Remote Sensing Learned Data Compression

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

This repo is just my fork from [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) where I modified some setup parameters to be ready to go directly after cloning.
Reference the **Modified from template** section to see the changes.


## Modified from template

- Uncomment logger in `environment.yaml` (wandb) as well as in `requirements.txt`

### Tests
- Uncomment `sh` in `requirements.txt` to allow the tests in `test_sweeps.py`
- Execute `pytest` in an interactive SLURM session on Juwels to validate `@RunIf(min_gpus=1)` in `test_train.py` (make sure Pytorch is installed with GPU support)
**=> Get all tests to be executed and None skipped.**

## Installation

```bash
# clone project
git clone https://github.com/CedricLeon/test_setup.git
cd test_setup

# create conda environment
conda create -n myenv python=3.9
conda activate myenv

# /!\ install pytorch with GPU support, see https://pytorch.org/get-started/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# install requirements
pip install -r requirements.txt
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

## How to test

```bash
# run all tests
pytest

# run all tests except the ones marked as slow
pytest -k "not slow"
```
