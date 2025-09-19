import hydra
from omegaconf import DictConfig, OmegaConf
import torch

# Add the root of the project to the python path (more info: https://github.com/ashleve/rootutils)
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class MyModel:
    def __init__(self, name: str, nb_channels: int, loss: str, lr: float = 0.001):
        self.name = name
        self.nb_channels = nb_channels
        self.loss = loss
        self.lr = lr
        print(f"Initialized MyModel with name={name}, nb_channels={nb_channels}, loss={loss}, lr={lr}")

class MyBIGModel:
    def __init__(self, name: str, nb_channels: int, loss: str, optimizer: torch.optim.Optimizer):
        self.name = name
        self.nb_channels = nb_channels
        self.loss = loss
        self.optimizer = optimizer

        print(f"Initialized MyBIGModel with name={name}, nb_channels={nb_channels}, loss={loss}, optimizer={optimizer.__class__.__name__}")

@hydra.main(version_base=None, config_path="../configs", config_name="basics.yaml")
def my_app(cfg: DictConfig) -> None:
    print(" --------------------- Configuration ----------------- ")
    print(OmegaConf.to_yaml(cfg))
    print(" --------------------- Model instantiation ----------------- ")
    if cfg.get("model") is not None:
        model = hydra.utils.instantiate(cfg.model)
        if isinstance(model, (MyModel, MyBIGModel)):
            print(model)
        else:
            print(f"{cfg.model} was not configured correctly.")
    else:
        print("No model specified in the config.")

if __name__ == "__main__":
    my_app()
