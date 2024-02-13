<div align="center">

# Your Project Name

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

The following way of installing is a homemade version of GitHub template. Using these would probably be more straightforward, but at least this homemade template is easy to follow 😉

### Clone the template and setup up your new remote
> In case you are doing that on a server or with the Gitlab of your company, with specific rules. Be careful when choosing `<your_repo_remote>`

```bash
# Clone the template
git clone https://github.com/CedricLeon/Setup_Lightning_Hydra_template.git
# Rename the folder with your project name
mv Setup_Lightning_Hydra_template/ <your_project_name>/
cd <your_project_name>/

# Rename the template remote
git remote rename origin template
# Add your new repository remote. So, yes, you need to create it before 
git remote add origin <your_repo_remote>
git remote -v

# --- Now it's a mess /!\ Double-check ---
# Synchronize your (empty new repo) with a rebase to avoid non-fast-forward errors
git pull --rebase origin main
# Push the commit history and all the template files on the new repo (also set the origin/main branch as upstream)
git push -u origin main
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

### Test the environment
You can try running a 10 epochs training of a SimpleDenseNet on MNIST classification problem to check if everything runs smooth. If you already logged on W&B on your system you should not need to do anything else for the setup to be complete. 
```bash
# If on a cluster, get on a compute node with a SLURm job or an interactive session
python src/train.py trainer=gpu # use trainer=cpu if you don't have gpus 
```

## Run

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

```bash
# run all tests
pytest

# run all tests except the ones marked as slow
pytest -k "not slow"
```
