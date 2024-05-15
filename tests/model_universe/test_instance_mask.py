from dataclasses import dataclass
from typing import Type
from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor

from paige.ml_sdk.model_universe.instance_mask import (
    _verify_instance_mask_invariants,
    apply_instance_mask,
    multiply_instance_mask_elementwise,
)


@dataclass
class Case_instance_mask:
    name: str
    tensor: Tensor
    instance_mask: Tensor
    expected_tensor: Tensor

    def __post_init__(self) -> None:
        self.__name__ = self.name


@dataclass
class Case_instance_mask_fail:
    name: str
    instance_mask: Tensor
    err: Type[Exception]
    err_msg: str
    tensor: Tensor = MagicMock(spec=Tensor)

    def __post_init__(self) -> None:
        self.__name__ = self.name


class Test_apply_instance_mask:
    @pytest.mark.parametrize(
        'case',
        (
            Case_instance_mask(
                name='3 instances - 2nd instance ignored',
                tensor=torch.tensor([0, 0, 1]),
                instance_mask=torch.tensor([True, False, True]),
                expected_tensor=torch.tensor([0, 1]),
            ),
            Case_instance_mask(
                name='3 instances - 2nd and 3rd instance ignored',
                tensor=torch.tensor([0, 0, 1]),
                instance_mask=torch.tensor([True, False, False]),
                expected_tensor=torch.tensor([0]),
            ),
            Case_instance_mask(
                name='3 instances - no instance ignored',
                tensor=torch.tensor([0, 0, 1]),
                instance_mask=torch.tensor([True, True, True]),
                expected_tensor=torch.tensor([0, 0, 1]),
            ),
            Case_instance_mask(
                name='3 instances - all instances ignored',
                tensor=torch.tensor([0, 0, 1]),
                instance_mask=torch.tensor([False, False, False]),
                expected_tensor=torch.tensor([], dtype=torch.long),
            ),
        ),
    )
    def test_should_apply_instance_mask_correctly(
        self,
        case: Case_instance_mask,
    ) -> None:
        tensor = apply_instance_mask(case.tensor, instance_mask=case.instance_mask)

        torch.testing.assert_close(tensor, case.expected_tensor)

    @pytest.mark.parametrize(
        'case',
        (
            Case_instance_mask_fail(
                name='non boolean values',
                instance_mask=torch.tensor([True, True, False, 999]),
                tensor=torch.tensor([0.1, 0.2, 0.3, 0.4]),
                err=IndexError,
                err_msg=r'.*index 999 is out of bounds.*',
            ),
        ),
    )
    def test_should_detect_incorrect_instance_mask(
        self,
        case: Case_instance_mask_fail,
    ) -> None:
        with pytest.raises(case.err, match=case.err_msg):
            apply_instance_mask(case.tensor, instance_mask=case.instance_mask)


class Test_multiply_instance_mask_elementwise:
    @pytest.mark.parametrize(
        'case',
        (
            Case_instance_mask(
                name='3 instances - 2nd instance zeroed out',
                tensor=torch.tensor([1, 2, 3]),
                instance_mask=torch.tensor([True, False, True]),
                expected_tensor=torch.tensor([1, 0, 3]),
            ),
            Case_instance_mask(
                name='3 instances - all instances zeroed out',
                tensor=torch.tensor([1, 2, 3]),
                instance_mask=torch.tensor([False, False, False]),
                expected_tensor=torch.tensor([0, 0, 0]),
            ),
            Case_instance_mask(
                name='3 instances - no instances zeroed out',
                tensor=torch.tensor([1, 2, 3]),
                instance_mask=torch.tensor([True, True, True]),
                expected_tensor=torch.tensor([1, 2, 3]),
            ),
        ),
    )
    def test_should_multiply_correctly(self, case: Case_instance_mask) -> None:
        tensor = multiply_instance_mask_elementwise(case.tensor, instance_mask=case.instance_mask)

        torch.testing.assert_close(tensor, case.expected_tensor)

    @pytest.mark.parametrize(
        'case',
        (
            Case_instance_mask_fail(
                name='(N, 1) vs. (N, ) tensors',
                instance_mask=torch.tensor([True, True, False, False]),
                tensor=torch.tensor([0.1, 0.2, 0.3, 0.4]).unsqueeze(-1),
                err=ValueError,
                err_msg=r'Tensor shape mismatch.*',
            ),
        ),
    )
    def test_should_detect_incorrect_instance_mask(
        self,
        case: Case_instance_mask_fail,
    ) -> None:
        with pytest.raises(case.err, match=case.err_msg):
            multiply_instance_mask_elementwise(case.tensor, instance_mask=case.instance_mask)


@pytest.mark.parametrize(
    'case',
    (
        Case_instance_mask_fail(
            name='empty instance_mask_map',
            instance_mask=torch.tensor([]),
            err=ValueError,
            err_msg=r'instance_mask size should .*',
        ),
        Case_instance_mask_fail(
            name='instance_mask different size from batch size',
            instance_mask=torch.tensor([True]),
            err=ValueError,
            err_msg=r'instance_mask size should .*',
        ),
        Case_instance_mask_fail(
            name='mask ndim not 1d',
            instance_mask=torch.tensor([[0]]),
            err=ValueError,
            err_msg=r'instance_mask should be 1d.*',
        ),
        Case_instance_mask_fail(
            name='Having None instance_mask',
            instance_mask=None,  # type: ignore[arg-type]
            err=TypeError,
            err_msg=r'instance_mask must be Tensor but received None.',
        ),
    ),
)
def test__verify_instance_mask_invariants_should_detect_incorrect_instance_mask(
    case: Case_instance_mask_fail,
) -> None:
    with pytest.raises(case.err, match=case.err_msg):
        _verify_instance_mask_invariants(case.tensor, instance_mask=case.instance_mask)
