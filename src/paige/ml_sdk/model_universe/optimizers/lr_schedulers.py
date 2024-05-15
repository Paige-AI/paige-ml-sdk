"""Space for custom learning rate schedulers used with optimizers.

For more details about learning rate scheduling, refer to [1] and [2].

[1]: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
[2]: https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html#learning-rate-scheduling
"""

from typing import Callable, Union

from torch.optim.lr_scheduler import LRScheduler  # type: ignore
from torch.optim.lr_scheduler import ReduceLROnPlateau

LRSchedulerTypes = Union[LRScheduler, ReduceLROnPlateau]
LRSchedulerPartial = Callable[..., LRSchedulerTypes]
