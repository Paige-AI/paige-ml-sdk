import logging
from collections import abc
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypedDict,
    Union,
    runtime_checkable,
)

from environs import Env
from lightning.pytorch import LightningModule, Trainer
from torch import Tensor
from torch.optim import Optimizer

from paige.ml_sdk.collections_reduce import concat
from paige.ml_sdk.dataset_universe.collate_fns import (
    EmbeddingAggregatorFitCollatedItems,
    EmbeddingAggregatorPredictCollatedItems,
)
from paige.ml_sdk.dict_procedures import deep_update
from paige.ml_sdk.distributed.collections_map import gather_per_tensor, to_cpu_per_tensor
from paige.ml_sdk.distributed.collective import is_td_active, is_world_main_process
from paige.ml_sdk.enforce_type import enforce_not_none_type, enforce_type
from paige.ml_sdk.model_universe.losses.losses_reducer import LossesReducer, SumLossesReducer
from paige.ml_sdk.model_universe.optimizers.lr_schedulers import (
    LRSchedulerPartial,
    LRSchedulerTypes,
)
from paige.ml_sdk.model_universe.optimizers.optimizer_partial import OptimizerPartial

logger = logging.getLogger(__name__)


@runtime_checkable
class AggregatorOutput(Protocol):
    heads_logits: Dict[str, Tensor]
    heads_activations: Dict[str, Tensor]


@runtime_checkable
class AggregatorNetwork(Protocol):
    def __call__(
        self, __x: Tensor, __padding_masks: Optional[Tensor]
    ) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        ...

    def forward(
        self, x: Tensor, padding_masks: Optional[Tensor]
    ) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        ...


def _convert_to_namedtuple(
    aggregator_output: Union[AggregatorOutput, Dict[str, Union[Tensor, Dict[str, Tensor]]]],
) -> AggregatorOutput:
    """convert to a NamedTuple and enforce that it adhere to the AggregatorOutput protocol."""
    if isinstance(aggregator_output, AggregatorOutput):
        return aggregator_output
    else:
        name_and_type = []
        for k, v in aggregator_output.items():
            if isinstance(v, Tensor):
                name_and_type.append((k, Tensor))
            # `out` has type Dict[str, Union[Tensor, Dict[str, Tensor]]], so we can assume it's
            # Dict[str, Tensor] if it wasnt Tensor.
            else:
                name_and_type.append((k, Dict[str, Tensor]))
        _AggregatorOutput = NamedTuple('_AggregatorOutput', name_and_type)
        return enforce_type(AggregatorOutput, _AggregatorOutput(**aggregator_output))


class AggregatorStepOutput(TypedDict):
    loss: Tensor
    label_map: Dict[str, Tensor]
    instance_mask_map: Dict[str, Tensor]
    heads_activations: Dict[str, Tensor]
    group_indices: Tensor


class AggregatorMetricsComputer(Protocol):
    def __call__(
        self,
        label_map: Dict[str, Tensor],
        instance_mask_map: Dict[str, Tensor],
        heads_activations: Dict[str, Tensor],
    ) -> Dict[str, Dict[str, Any]]:
        ...


class Aggregator(LightningModule):
    """Defines an aggregator."""

    def __init__(
        self,
        *,
        model: AggregatorNetwork,
        losses_reducer: LossesReducer = SumLossesReducer(),
        loss_computers: Optional[Mapping[str, Any]] = None,
        optimizer_partial: Optional[OptimizerPartial] = None,
        lr_scheduler_partial: Optional[LRSchedulerPartial] = None,
        train_metrics_computers: Optional[Sequence[AggregatorMetricsComputer]] = None,
        val_metrics_computers: Optional[Sequence[AggregatorMetricsComputer]] = None,
        test_metrics_computers: Optional[Sequence[AggregatorMetricsComputer]] = None,
        train_step_output_keys: Optional[Sequence[str]] = None,
        val_step_output_keys: Optional[Sequence[str]] = None,
        test_step_output_keys: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self._losses_reducer = losses_reducer
        self._loss_computers = loss_computers
        self._lr_scheduler_partial = lr_scheduler_partial
        self._optimizer_partial = optimizer_partial
        self._train_metrics_computers = train_metrics_computers or []
        self._val_metrics_computers = val_metrics_computers or []
        self._test_metrics_computers = test_metrics_computers or []
        self._train_step_output_keys = train_step_output_keys
        self._val_step_output_keys = val_step_output_keys
        self._test_step_output_keys = test_step_output_keys

        self.training_outputs: List[AggregatorStepOutput] = []
        self.validation_outputs: List[AggregatorStepOutput] = []
        self.test_outputs: List[AggregatorStepOutput] = []

    def forward(  # type: ignore
        self, x: Tensor, padding_masks: Optional[Tensor]
    ) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        """Runs a forward pass.

        .. note:: Torchscript does not support Protocols, so this method returns a list of
           tensors and/or dicts in order to be torchscript-compatible yet to enable different
           aggregator architectures to return different things. In reality think of the output of
           this method as something that adheres to the `AggregatorOutput` Protocol.

        """
        return self.model(x, padding_masks)

    def training_step(  # type: ignore
        self,
        batch: EmbeddingAggregatorFitCollatedItems,
        batch_idx: int,
    ) -> AggregatorStepOutput:  # type: ignore
        """Runs a forward pass on a single batch of train dataset and returns the loss."""
        model_output = self._shared_forward_pass(batch)
        loss = self._compute_per_batch_loss(batch, model_output, stage='train')

        output = self._return_step_output(
            batch, model_output, loss, batch_idx, self._train_step_output_keys, stage='train'
        )

        if Env().bool('PAIGE_ml_sdk__USE_AGGREGATOR_TRAINING_EPOCH_END', True):
            self.training_outputs.append(output)

        return output

    def at_epoch_end(
        self,
        outputs: List[AggregatorStepOutput],
        metric_computers: Sequence[AggregatorMetricsComputer],
        stage: Literal['train', 'val', 'epoch', 'test'],
    ) -> None:  # type: ignore
        """Performs a routine at the end of each epoch."""
        # concatenate batch outputs
        batch_model_output_or_none = self._concat_training_step_outputs_on_main_rank(outputs)
        if batch_model_output_or_none is None:
            assert is_td_active() and not is_world_main_process()
            return None

        if is_td_active() and not is_world_main_process():
            raise RuntimeError('only main process is allowed to perform metrics computation.')

        # Compute epoch metrics. Must do 2 things:
        # 1. Flatten them because `log_dict`` only accepts flat dictionaries
        # 2. Remove null metrics because `log_dict` only accepts Tensors, floats, and integers
        #    This happens when a metric cannot be computed due to insufficient sample size
        epoch_metrics = self._compute_metrics_per_epoch(
            *batch_model_output_or_none, metric_computers
        )
        epoch_metrics = flatten({stage: epoch_metrics})
        epoch_metrics = {k: v for k, v in epoch_metrics.items() if v}
        self.log_dict(epoch_metrics)

    def on_train_epoch_end(self) -> None:
        if Env().bool('PAIGE_ml_sdk__USE_AGGREGATOR_TRAINING_EPOCH_END', True):
            self.at_epoch_end(
                outputs=self.training_outputs,
                metric_computers=self._train_metrics_computers,
                stage='train',
            )
            self.training_outputs = []

    def validation_step(  # type: ignore
        self,
        batch: EmbeddingAggregatorFitCollatedItems,
        batch_idx: int,
    ) -> AggregatorStepOutput:
        """Runs a forward pass on a single batch of validation dataset and returns the loss."""
        model_output = self._shared_forward_pass(batch)
        loss = self._compute_per_batch_loss(batch, model_output, stage='val')
        output = self._return_step_output(
            batch, model_output, loss, batch_idx, self._val_step_output_keys, stage='val'
        )

        if Env().bool('PAIGE_ML_SDK__USE_AGGREGATOR_VALIDATION_EPOCH_END', True):
            self.validation_outputs.append(output)

        return output

    def on_validation_epoch_end(self) -> None:
        if Env().bool('PAIGE_ML_SDK__USE_AGGREGATOR_VALIDATION_EPOCH_END', True):
            self.at_epoch_end(
                outputs=self.validation_outputs,
                metric_computers=self._val_metrics_computers,
                stage='val',
            )
            self.validation_outputs = []

    def test_step(  # type: ignore
        self,
        batch: EmbeddingAggregatorFitCollatedItems,
        batch_idx: int,
    ) -> AggregatorStepOutput:
        """Runs a forward pass on a single batch of validation dataset and returns the loss."""
        model_output = self._shared_forward_pass(batch)
        loss = self._compute_per_batch_loss(batch, model_output, stage='test')
        output = self._return_step_output(
            batch, model_output, loss, batch_idx, self._test_step_output_keys, stage='test'
        )

        if Env().bool('PAIGE_ml_sdk__USE_AGGREGATOR_TEST_EPOCH_END', True):
            self.test_outputs.append(output)

        return output

    def on_test_epoch_end(self) -> None:
        if Env().bool('PAIGE_ml_sdk__USE_AGGREGATOR_TEST_EPOCH_END', True):
            self.at_epoch_end(
                outputs=self.test_outputs,
                metric_computers=self._test_metrics_computers,
                stage='test',
            )
            self.test_outputs = []

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[List[Optimizer], List[LRSchedulerTypes]]]:
        """Configures an optimizer and LR scheduler."""
        optimizer_partial = enforce_not_none_type(self._optimizer_partial)
        optimizer: Optimizer = optimizer_partial(self.parameters())

        if self._lr_scheduler_partial is not None:
            lr_scheduler: LRSchedulerTypes = self._lr_scheduler_partial(optimizer)
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    # ================ APIs we use during predict ======================
    def predict_step(  # type: ignore
        self,
        batch: Union[EmbeddingAggregatorFitCollatedItems, EmbeddingAggregatorPredictCollatedItems],
        batch_idx: int,
    ) -> AggregatorOutput:
        return self._shared_forward_pass(batch)

    # ================ custom APIs ======================
    def _return_step_output(
        self,
        batch: EmbeddingAggregatorFitCollatedItems,
        model_output: AggregatorOutput,
        loss: Tensor,
        batch_idx: int,
        keys: Optional[Sequence[str]],
        stage: Literal['train', 'val', 'epoch', 'test'],
    ) -> AggregatorStepOutput:
        model_output_dict = model_output._asdict()  # type: ignore
        # NOTE: At the moment this may cause OOM bc the embedding is much larger
        #       than other members of EmbeddingAggregatorFitCollatedItems.
        #       When we have a metric that requires this Aggregator embedding
        #       we can rethink how to work this back in.
        model_output_dict.pop('backbone_embedding', None)

        output = dict(
            group_indices=batch.group_indices,
            label_map=batch.label_map,
            instance_mask_map=batch.instance_mask_map,
            **model_output_dict,
        )

        if keys:
            d = dict(loss=loss, **output)
            return {k: v for k, v in d.items() if k in keys}  # type: ignore
        else:
            # keeping this branch for backwards compatibility
            if batch_idx == 0:
                logger.warning(
                    f'Returning {stage}_step output for {output.keys()}. For memory efficiency, consider specifying Aggregator({stage}_step_output_keys=[...]) to include only what you need'
                )
            return AggregatorStepOutput(loss=loss, **output)  # type: ignore

    def _shared_forward_pass(
        self,
        batch: Union[EmbeddingAggregatorFitCollatedItems, EmbeddingAggregatorPredictCollatedItems],
    ) -> AggregatorOutput:
        """Common calling of forward on the model to centralize logic for all steps.

        .. note:: `forward` must return a torchscript-compatible type, but computations downstream
           from `forward` (e.g. loss/metric computation) would like to rely on more sophisticated
           types. Thus we immediately convert `forward`'s output to a NamedTuple and enforce that it
           adhere to the AggregatorOutput protocol.
        """
        out = self(batch.embeddings, batch.padding_mask)
        return _convert_to_namedtuple(out)

    def _compute_per_batch_loss(
        self, batch: EmbeddingAggregatorFitCollatedItems, output: AggregatorOutput, stage: str
    ) -> Tensor:
        """Computes a loss value for each loss computer."""
        losses = {}
        loss_computers = enforce_not_none_type(self._loss_computers)

        for label_name, loss_computer in loss_computers.items():
            per_label_loss = loss_computer(batch, output, label_name)
            losses[label_name] = per_label_loss
            self.log(f'{stage}_loss.{label_name}', per_label_loss)

        loss = self._losses_reducer.reduce(losses)
        self.log(f'{stage}_loss', loss)

        return loss

    @staticmethod
    def _concat_training_step_outputs_on_main_rank(
        outputs: List[AggregatorStepOutput],
    ) -> Optional[Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]]:
        """
        For each Tensor and NDArray in batches and model outputs:
        - concatenate along the 0th dimension on the each rank
        - move all GPU Tensors to CPU
        - gather Tensors and NDArrays on the main rank, concatenate along 0th dimension
        """
        batches_model_outputs = [
            (d['label_map'], d['instance_mask_map'], d['heads_activations']) for d in outputs
        ]
        return gather_per_tensor(to_cpu_per_tensor(concat(batches_model_outputs)))

    @staticmethod
    def _compute_metrics_per_epoch(
        label_map: Dict[str, Tensor],
        instance_mask_map: Dict[str, Tensor],
        heads_activations: Dict[str, Tensor],
        metrics_computers: Sequence[AggregatorMetricsComputer],
    ) -> Dict[str, Any]:
        """Computes a metric value for each metrics computer."""
        metrics: Dict[str, Any] = {}
        for metrics_computer in metrics_computers:
            deep_update(
                metrics,
                metrics_computer(label_map, instance_mask_map, heads_activations),
            )
        return metrics

    def _get_trainer(self) -> Trainer:
        """Fetches the trainer instance attached to the model at runtime."""
        # Enforcing Trainer type since we rely that PL must attach it to the module by this point.
        trainer: Trainer = enforce_type(Trainer, self.trainer)
        return trainer


def flatten(d: Any, parent_key: str = '', sep: str = '.'):
    """Flattens a dict according to sep.

    example: flatten({'a': {'b': 1}}) -> {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
