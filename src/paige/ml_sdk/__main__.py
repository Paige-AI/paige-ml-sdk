from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import *
from torch.optim import *

from paige.ml_sdk.dataset_universe.datamodule import init_aggregator_datamodule
from paige.ml_sdk.model_universe.aggregator import Aggregator

# By importing these lightningmodules, we can use them in the aggregator sdk without needing
# to specify the full import path (e.g., can do --model BinClsAgata instead of --model paige.ml_sdk...)
from paige.ml_sdk.model_universe.algos import BinClsAgata, MultiClsAgata

# Todo
# - perceiver
# - remove polars dependency- just use pandas.
# - single cls
# - args that could be shared across datamodule and model: missing label value, target cols
# - embedding loading is confusing and can be simplified


def main() -> None:
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')
    cli = LightningCLI(
        Aggregator,
        init_aggregator_datamodule,
        save_config_kwargs={'overwrite': True},
        trainer_defaults={
            'accelerator': 'auto',
            'log_every_n_steps': 3,
            'max_epochs': 2,
            'callbacks': [checkpoint_callback],
        },
        run=False,
        subclass_mode_model=True,
    )
    cli.trainer.fit(cli.model, cli.datamodule)
    if cli.datamodule.test_dataset:
        cli.trainer.test(cli.model, cli.datamodule, ckpt_path='best')


if __name__ == '__main__':
    main()
