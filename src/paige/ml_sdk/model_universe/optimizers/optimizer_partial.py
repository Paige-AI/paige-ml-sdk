from typing import Callable

from torch.optim import Optimizer

OptimizerPartial = Callable[..., Optimizer]
