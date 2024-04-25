# Your Project Name
<div align="center">

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

This repo is just my fork from [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) where I modified some setup parameters to be ready to go directly after cloning.
Reference the **Modified from template** section to see the changes. otherwise the main modifications are the following:

- I set up Hydra-submitit launcher for an easier usage of SLURM, and add example config setups for clusters (JUWELS, Terrabyte)
- I commit to use Weight & Biases the (W&B) logger. Other loggers are still possible to use, but everything is setup by default for W&B
- I use `wandb_osh` to support offline, real-time logging of my runs on W&B
  - To set that up you just need to turn `logger.wandb.offline` to `True` and have the "Farm" running on the login node
  - See further explanations and motivations on [wandb-osh GitHub](https://github.com/klieret/wandb-offline-sync-hook)

## Modified from template

### New features

- Add deterministic training support (can be unset from config)
- Add the W&B offline management using `wandb_osh` (automatically adds the Lightning Callback when the run is set offline)

### More dependencies

- Uncomment my favorite logger in `environment.yaml` (**wandb**) as well as in `requirements.txt`
- Add additional requirements:
  - `hydra-submitit-launcher`
  - `torchgeo`
  - `wandb_osh` (Wandb Offline Sync Hook)
- Uncomment `sh` in `requirements.txt` to allow the tests in `test_sweeps.py`

### How to test

- Execute `pytest` in an interactive SLURM session on Juwels to validate `@RunIf(min_gpus=1)` in `test_train.py` (make sure Pytorch is installed with GPU support)
**=> Get all tests to be executed and None skipped.**

## Installation

There are 2 different ways you can setup your new repository: by keeping track of the template, or by starting a fresh new git repo with all the files from the template.

> If you plan to host your code on DLR GitLab, you should make sure that when you create the new repository you create it as a "blank project", select **<your_user>** and not **<your_group>** in the Project URL and uncheck "Initialize repository with a README".
> Also, you should use the HTTPS URLs.

In both cases you first need to clone the template and rename the folder with `<your_project_name>`:

```bash
# Clone the template
git clone https://github.com/CedricLeon/Setup_Lightning_Hydra_template.git
# Rename the folder with your project name
mv Setup_Lightning_Hydra_template/ <your_project_name>/
cd <your_project_name>/
```

Then you can either delete the remote and commit history of the template, this is the most straightforward way:

```bash
# Reset the git repository
rm -rf .git/
git init --initial-branch=main
# Add your remote
git remote add origin <your_remote_URL>
# Stage and commit all files + set origin main as upstream
git add .
git commit -m "Initial commit"
git push --set-upstream origin main
```

or you can keep the remote but rename it to `template` and add a new `origin`.

I describe how to do that below, but you should know that it is just a homemade version of template repository from GitHub. It is less clean, but allows to host the new repo on a server that isn't GitHub (I didn't find a way to do that using the GitHub template feature). *If someone has a cleaner way of doing it, please open a pull-request.*

```bash
# Rename the template remote
git remote rename origin template
# Add your new repository remote. So, yes, you need to create it before
git remote add origin <your_remote_URL>
git remote -v

# Synchronize your (empty new repo) with a rebase to avoid non-fast-forward errors
git pull --rebase origin main
# Push the commit history and all the template files on the new repo (also set the origin/main branch as upstream)
git push --set-upstream origin main
```

### Set your conda environment

```bash
conda create -n <your_env_name> python=3.11
conda activate <your_env_name>

# /!\ install pytorch with GPU support, see https://pytorch.org/get-started/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# install requirements
pip install -r requirements.txt
```

### Overwrite project template parameters

I have fixed some parameters with generic names (e.g., `logger.wandb.project: "lightning-hydra-template"`).
Here is a list you should check and replace:

- In `configs/logger/wandb.yaml`: `logger.wandb.team` and `logger.wandb.project`

As a general comment, I advise to run a mock run (**/!\ not with `debug=fdr` /!\**, it hides most of the config) and have a careful look at your config. @TODO

### Test the environment

You can try running a 10 epochs training of a SimpleDenseNet on MNIST classification problem to check if everything runs smooth. If you already logged on W&B on your system you should not need to do anything else for the setup to be complete.

```bash
# If on a cluster, get on a compute node with a SLURM job or an interactive session
python src/train.py trainer=gpu # use trainer=cpu if you don't have gpus
```

## Usage / Run

@TODO: refine the usage examples with how I use experiments, etc.

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

## Run tests

@TODO: Specify how to add test

```bash
# run all tests
pytest

# run all tests except the ones marked as slow
pytest -k "not slow"
```
