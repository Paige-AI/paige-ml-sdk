from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import *
from torch.optim import *

from paige.ml_sdk.dataset_universe.datamodule import init_datamodule_from_dataset_filepaths
from paige.ml_sdk.model_universe.aggregator import Aggregator

# By importing these lightningmodules, we can use them in the aggregator sdk without needing
# to specify the full import path (e.g., can do --model BinClsAgata instead of --model paige.ml_sdk.algos.BinClsAgata)
from paige.ml_sdk.model_universe.algos import BinClsAgata, MultiClsAgata


def main() -> None:
    """CLI for model training.

    This function makes use of the Lightning CLI. For more info on it, see here:
    https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html#lightning-cli

    Or refer to this repository's README.md
    """
    # callbacks can be overriden from cli as shown here:
    # https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced_3.html#trainer-callbacks-and-arguments-with-class-type
    #
    # For example, to use a metric other than `val_loss`, you can put this in config.yaml:
    # trainer:
    #   callbacks:
    #       - class_path: pytorch_lightning.callbacks.ModelCheckpoint
    #         init_args:
    #           monitor: val.cancer.auc
    #           mode: max
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')
    LightningCLI(
        model_class=Aggregator,
        datamodule_class=init_datamodule_from_dataset_filepaths,
        # LightningCLI will write the experiment's configuration to a yaml file called
        # `config.yaml`. Setting `'overwrite'=True` tells the CLI to overwrite the
        # file if it alread exists.
        save_config_kwargs={'overwrite': True},
        # These can be overriden, f.e., `--trainer.max_epochs 100`
        trainer_defaults={
            'accelerator': 'auto',  # will use GPU if available
            'log_every_n_steps': 3,
            'max_epochs': 2,
            'callbacks': [checkpoint_callback],
        },
        subclass_mode_model=True,  # Any subclass of `Aggregator` can be provided to the --model arg.
    )


if __name__ == '__main__':
    main()
