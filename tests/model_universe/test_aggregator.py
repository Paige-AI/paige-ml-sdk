from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union

import pytest
import torch
import torch.nn as nn
from lightning import Trainer
from torch import Tensor
from torch.jit import ScriptModule

from paige.ml_sdk.dataset_universe.datamodule import AggregatorDataModule
from paige.ml_sdk.dataset_universe.datasets.dataset import EmbeddingDataset
from paige.ml_sdk.model_universe.aggregator import (
    Aggregator,
    AggregatorMetricsComputer,
    AggregatorNetwork,
)
from paige.ml_sdk.model_universe.losses.loss_computer import AggregatorLossComputer
from paige.ml_sdk.model_universe.metrics.metrics_computers.aggregator_metrics import (
    AggregatorBinClsGroupMetricsComputer,
    AggregatorMulticlassGroupMetricsComputer,
)

# ================ Constants Used Throughout Fixtures ======================


BATCH_SIZE = 8
IN_EMBEDDING_SIZE = 5
N_EMBEDDINGS = 50
LABEL_NAME_TO_N_CLASSES = {'cancer': 2, 'precursor': 2}
LABEL_NAME_TO_OUTPUT_DIMS = {'cancer': 2, 'precursor': 1}  # 1-dim'l to test softmax vs. sigmoid.

# ================ Data Fixtures ======================


@pytest.fixture
def f_aggregator_fit_datamodule(
    f_path_to_dataset_csv: Path, f_path_to_embeddings_dir: Path
) -> AggregatorDataModule:
    dataset = EmbeddingDataset.from_filepath(
        dataset=f_path_to_dataset_csv,
        embeddings_dir=f_path_to_embeddings_dir,
        label_missing_value=-999,
        label_columns={'cancer', 'precursor'},
        embeddings_filename_column='image_uri',
    )

    datamodule = AggregatorDataModule(
        train_dataset=dataset,
        tune_dataset=dataset,
        test_dataset=dataset,
        num_workers=0,
        batch_size=4,
    )
    return datamodule


# ================ Model Fixtures ======================


# To test our Aggregator lightningmodule we must provide it with an arbitrary aggregator. Here
# we define a simple aggregator that sums the embeddings and then learns a linear classifier. We
# then pass this model to `f_algo` defined below. We opt not to return this model via a fixture
# because when we try to do that PL inexplicably drops it while doing f_algo.save_checkpoint (i.e.
# `f_algo.model` does not get saved within `f_algo`'s checkpoint if `model` is a fixture).
class _BoringBackpropableModel(nn.Module):
    def __init__(self) -> None:
        """The simplest aggregator we can do that tests multiple heads and backpropagation.
        Exists for testing purposes only."""
        super().__init__()
        self.classifier_1 = nn.Linear(IN_EMBEDDING_SIZE, LABEL_NAME_TO_OUTPUT_DIMS['cancer'])
        self.classifier_2 = nn.Linear(IN_EMBEDDING_SIZE, LABEL_NAME_TO_OUTPUT_DIMS['precursor'])
        self.activation_1 = nn.Softmax(dim=1)
        self.activation_2 = nn.Sigmoid()

    def forward(  # type: ignore
        self, x: Tensor, padding_masks: Optional[Tensor]
    ) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        x = torch.sum(x, 1)
        logits_1 = self.classifier_1(x)
        logits_2 = self.classifier_2(x)
        activations_1 = self.activation_1(logits_1)
        activations_2 = self.activation_2(logits_2)
        return {
            'heads_logits': {'cancer': logits_1, 'precursor': logits_2},
            'heads_activations': {'cancer': activations_1, 'precursor': activations_2},
        }


@pytest.fixture
def f_aggregator_model() -> AggregatorNetwork:
    return _BoringBackpropableModel()


@pytest.fixture
def f_loss_computers() -> Dict[str, AggregatorLossComputer]:
    return {
        'cancer': AggregatorLossComputer(
            loss_fn=nn.CrossEntropyLoss(),
            match_target_dim_to_input_dim=False,
            target_to_dtype=torch.long,
        ),
        'precursor': AggregatorLossComputer(
            loss_fn=nn.BCEWithLogitsLoss(),
            match_target_dim_to_input_dim=True,
            target_to_dtype=torch.float,
        ),
    }


@pytest.fixture
def f_metrics_computers() -> List[AggregatorMetricsComputer]:
    return [
        AggregatorBinClsGroupMetricsComputer(label_name='cancer'),
        AggregatorBinClsGroupMetricsComputer(label_name='precursor'),
    ]


@pytest.fixture
def f_aggregator_algo(
    f_aggregator_model: AggregatorNetwork,
    f_loss_computers: Dict[str, AggregatorLossComputer],
    f_metrics_computers: List[AggregatorMetricsComputer],
) -> Aggregator:
    return Aggregator(
        model=f_aggregator_model,
        loss_computers=f_loss_computers,
        train_metrics_computers=f_metrics_computers,
        val_metrics_computers=f_metrics_computers,
        optimizer_partial=partial(torch.optim.SGD, lr=0.0001),
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=10),
    )


# ================ Test ======================


class TestAggregator:
    @pytest.mark.integration
    def test_should_fit_cpu_no_distributed_without_runtime_exception(
        self,
        f_aggregator_algo: Aggregator,
        f_aggregator_fit_datamodule: AggregatorDataModule,
    ) -> None:
        trainer = Trainer(fast_dev_run=True, max_epochs=1)

        trainer.fit(model=f_aggregator_algo, datamodule=f_aggregator_fit_datamodule)

    # @pytest.mark.integration
    # @pytest.mark.gpu
    # @pytest.mark.parametrize('devices', [1])
    # fails w/ nccl error
    # def test_should_fit_gpu_distributed_without_runtime_exception(
    #     self,
    #     f_aggregator_algo: Aggregator,
    #     f_aggregator_fit_datamodule: AggregatorDataModule,
    #     devices: int,
    # ) -> None:
    #     trainer = Trainer(
    #         fast_dev_run=True,
    #         accelerator='gpu',
    #         strategy='ddp_find_unused_parameters_false',
    #         devices=devices,
    #         max_epochs=1,
    #     )

    #     trainer.fit(model=f_aggregator_algo, datamodule=f_aggregator_fit_datamodule)

    @pytest.mark.integration
    def test_should_export_to_supported_formats_and_do_forward_pass_as_before(
        self,
        f_aggregator_algo: Aggregator,
        tmp_path: Path,
    ) -> None:
        # torchscript algo
        save_path = (tmp_path / 'algo_scripted.ts').as_posix()
        f_aggregator_algo.to_torchscript(
            file_path=save_path,
            method='script',
        )
        algo_scripted: ScriptModule = torch.jit.load(save_path)

        # run both the torchscripted and non-torchscripted models
        f_aggregator_algo.eval()
        algo_scripted.eval()
        x = torch.rand(BATCH_SIZE, N_EMBEDDINGS, IN_EMBEDDING_SIZE)
        padding_masks = torch.zeros((BATCH_SIZE, N_EMBEDDINGS), dtype=torch.uint8)
        preexport_fwdout = f_aggregator_algo(x, padding_masks)
        postexport_fwdout = algo_scripted(x, padding_masks)

        keys = preexport_fwdout.keys()
        assert preexport_fwdout.keys() == postexport_fwdout.keys()
        for output_key in keys:
            if isinstance(preexport_fwdout[output_key], Tensor):
                torch.testing.assert_close(
                    postexport_fwdout[output_key],
                    preexport_fwdout[output_key],
                    atol=1e-4,
                    rtol=1e-2,
                )
            else:
                for label_name in preexport_fwdout[output_key].keys():
                    torch.testing.assert_close(
                        postexport_fwdout[output_key][label_name],
                        preexport_fwdout[output_key][label_name],
                        atol=1e-4,
                        rtol=1e-2,
                    )
