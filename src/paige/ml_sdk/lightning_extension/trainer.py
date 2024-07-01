from typing import Optional, Union

from lightning import Trainer, LightningModule, LightningDataModule
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS, _EVALUATE_OUTPUT
from paige.ml_sdk.dataset_universe.datamodule import AggregatorDataModule


# Subclassing Lightning's Trainer to offer a fit_and_test method
class AggregatorTrainer(Trainer):
    def fit_and_test(
        self,
        model: LightningModule,
        datamodule: AggregatorDataModule,
        ckpt_path: Optional[_PATH] = None,
    ) -> _EVALUATE_OUTPUT:
        "Runs the full optimization routine, followed by one evaluation epoch over the test set."
        self.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        return self.test(model=model, datamodule=datamodule, ckpt_path='best')
