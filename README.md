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
- Redirecting logs to subdirectories specific for each experiment
- Automating job submission on cluster using `hydra-submitit-launcher` through `--multirun` mode

### More dependencies

- Uncomment my favorite logger in `environment.yaml` (**wandb**) as well as in `requirements.txt`
- Add additional requirements:
  - `hydra-submitit-launcher`
  - `torchgeo`
  - `wandb_osh` (Wandb Offline Sync Hook)
- Uncomment `sh` in `requirements.txt` to allow the tests in `test_sweeps.py`

### CI/CD and Testing

- *To execute all tests:* Execute `pytest` in an interactive SLURM session on Juwels to validate `@RunIf(min_gpus=1)` in `test_train.py` (make sure Pytorch is installed with GPU support)
**=> Get all tests to be executed and None skipped.**
- I removed MacOS and Windows deployment test, as well as most of the different versions of python tested (reason: save computation resources)
- The tests to be executed in CI/CD are the `"not slow"` ones, for the same reason mentioned above

## Features to come, @TODO

- Upgrade and "automatise" the `task_name` parameter generation:
  - Either by using a specific name parameter in each Config Group option (config file) and `**kwargs` in the corresponding Modules.
  - Or by making it general and global in the "root" config file using Hydra interpolation system. Not that easy because it's impossible to interpolate in the Default List, see this [stackoverflow](https://stackoverflow.com/questions/67280041/interpolation-in-hydras-defaults-list-cause-and-error).
- Add a submitit setup for Terrabyte
- Increase test coverage, and provide classic examples to test Lightning Datamodule and Modules.

## Installation

### Setting up Git

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

#### Re-initializating Git history

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

#### Keeping track of the template Git history (*homemade*)

Or you can keep the remote but rename it to `template` and add a new `origin`.

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
# Force python version 3.11 for compatibility reasons (pytorch)
conda create -n <your_env_name> python=3.11
conda activate <your_env_name>

# /!\ install pytorch with GPU support, see https://pytorch.org/get-started/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# install requirements
pip install -r requirements.txt
```

### Optional: Setup `pre-commit` hooks

In case you deleted `.git` after cloning the template, you have to reinstall pre-commit.
It's also a good idea to run it against all files (if you have any) for the first time.

```bash
pre-commit install
pre-commit run --all-files
```

You can test that pre-commit is nicely setup with a dummy commit, or just by committing the changes of the next sections.

> Note: if you are using VSCode commit system, the output logs are redirected towards the `OUTPUT/Git` console. Nevertheless, you should still get an error message if you messed something.

### Personalize project template parameters

I have fixed some parameters with generic names (e.g., `logger.wandb.project: "lightning-hydra-template"`).
Here is a list you should check and replace:

- In `configs/logger/wandb.yaml`: `logger.wandb.team` and `logger.wandb.project`
- If you plan to use multiruns, in `configs/hydra/launcher/` change your account settings: `hydra.launcher.account` and if necessary your favorite `partition`.

As a general comment, I advise to run a mock run (**/!\ not with `debug=fdr` /!\**, it hides most of the config) and have a careful look at your config. @TODO

### Optional: Test the environment

You can try running a 10 epochs training of a SimpleDenseNet on MNIST classification problem to check if everything runs smooth. If you already logged on W&B on your system you should not need to do anything else for the setup to be complete.

```bash
# If on a cluster, get on a compute node with a SLURM job or an interactive session
python src/train.py experiment=example trainer=gpu # use trainer=cpu if you don't have gpus
```

## Usage / Run

@TODO: refine the usage examples with how I use experiments, etc.

### Classic usage

The method I describe below is **my** preferred way of using this template. Of course, that's only a theory and you are free to organize yourself differently, the repository is very flexible.
However, after trying out different setups I often found myself lost, e.g., trying to find out why a parameter kept its old value when I was overriding it. In any case, [Hydra documentation](https://hydra.cc/docs/patterns/configuring_experiments/) is your best friend.

Now that you are warned, here is are my best practices in short: I recommend always creating runs from an experiment config.
This enforces better hierarchy and organization, while having the advantage of grouping "all" modifications in a single file, making modifications easy.

See below an example to run with a chosen experiment configuration from [configs/experiment/](configs/experiment/):

```bash
python src/train.py experiment=example
```

### Overriding HYDRA config from CLI

From here, you can override minor parameters from the CLI for a quick check or a specific run:

```bash
python src/train.py experiment=example trainer.max_epochs=2
```

Whenever you find yourself, running several times similar commands with a high number of overrides, this is probably a good time to create a new `experiment.yaml`.

### Overriding a full config group

Sometimes, you might want to change big parts of your experiment config without wanting to redefine a new experiment, then, you can override packages.
Examples include checking run time on a different hardware, estimating results on a different dataset, or logging to csv because you're a boomer.

```bash
# Train on CPU
python src/train.py experiment=example trainer=cpu
# Quickly test another dataset
python src/train.py experiment=example data=kodak
# Change the logger
python src/train.py experiment=example logger=csv
```

### Debugging

Debugging is a instance of the previous case, where you override the debug package from the CLI. However, it's so common and important it deserves its own section.
Firstly, whenever you specify `debug` there won't be any logging or callbacks and the run will be executed without multithreading on CPU.
The best example is the `fast_dev_run` option of the Lightning Trainer which will run 1 step of training, validation and test. This is what 99% of the time.

```bash
python src/train.py experiment=example debug=fdr
```

If you still want some logging, or want to debug on GPU, etc. you can always specify that **after** your debug setup.

```bash
python src/train.py experiment=example debug=default trainer=gpu
```

## Run tests

@TODO: Specify how to add test

```bash
# run all tests
pytest

# run all tests except the ones marked as slow
pytest -k "not slow"
```
