__all__ = ('TorchProcessGroup',)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Use the unstable, private import location for type checking because
    # the public location is a stub class during type checking (as of torch==1.13.0).
    from torch.distributed.distributed_c10d import ProcessGroup as TorchProcessGroup
else:
    # Use the stable, public import location during runtime.
    from torch.distributed import ProcessGroup as TorchProcessGroup
