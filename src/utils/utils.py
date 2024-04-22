import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

import omegaconf
import wandb
from lightning_utilities.core.rank_zero import rank_zero_only

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def early_wandb_initialization(cfg: Dict[str, Any]) -> None:
    """Manual initialization of the W&B run. Extra logic is called is the run is set offline, see wandb_osh.
    Usually called before the Lightning Trainer is instantiated.
    We can safely call wandb.init() here, Lightning loggers will reuse the on-going run when instantiating: https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/loggers/wandb.html#WandbLogger
    Also, see: W&B with HYDRA: https://wandb.ai/adrishd/hydra-example/reports/Configuring-W-B-Projects-with-Hydra--VmlldzoxNTA2MzQw

    :param cfg: A dictionary containing the following objects:
    """
    # Manage offline syncing, see: https://github.com/klieret/wandb-offline-sync-hook
    if cfg.logger.wandb.offline:
        import wandb_osh

        # Add a Lightning callback triggering the sync after each epoch
        # Adding it this way makes it invisible for the user, but it won't appear in the HYDRA config (it will be printed in the logs though)
        with omegaconf.open_dict(cfg):
            cfg.callbacks.wandb_osh = {
                "_target_": "wandb_osh.lightning_hooks.TriggerWandbSyncLightningCallback"
            }
        log.info(
            "The W&B run is set offline. The Wandb Offline Sync Hook is initialized. \033[31mMake sure that the wandb_osh script is running on the login node.\033[0m"  # or somewhere where you have internet access
        )

        # Suppress logging messages (e.g., warnings about the syncing not being fast enough)
        wandb_osh.set_log_level("ERROR")  # for wandb_osh.__version__ >= 1.2.0

    # Manual cast of the config from a DictConfig to a regular dict (should be supported by W&B by now)
    wandb.config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(
        entity=cfg.logger.wandb.entity,
        project=cfg.logger.wandb.project,
        dir=cfg.logger.wandb.save_dir,
        # name=make_a_nice_run_name(cfg), @TODO: implement make_a_nice_run_name
        tags=cfg.tags,
        mode="offline" if cfg.logger.wandb.offline else "online",
        settings=wandb.Settings(start_method="thread"),
    )


def extras(cfg: omegaconf.DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: omegaconf.DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
