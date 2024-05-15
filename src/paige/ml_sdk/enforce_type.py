from typing import Any, Optional, Type, TypeVar

import torch

# by definition bound=object is covariant b/c there's no contravariant of object
T_co = TypeVar('T_co', bound=object)


# TODO (dh): investigate. type propagation is broken due to the jit unused decorator
@torch.jit.unused
def enforce_type(t: Type[T_co], o: Any, msg: Optional[str] = None) -> T_co:
    """
    Resolve type for mypy and check type at runtime.

    Args:
        t: Type to enforce, e.g. `str`.
        o: Object to check for type, e.g. 'abc'.
        msg: Custom error message.

    Raises:
        TypeError: If type of `o` doesn't match `t`.

    Returns:
        T_co: Return `o` (pass-through) if its type is correct.
    """
    if not isinstance(o, t):
        msg = msg if msg is not None else f'Expected {t} but got: {o}'
        raise TypeError(msg)
    return o


@torch.jit.unused
def enforce_optional_type(t: Type[T_co], o: Any, msg: Optional[str] = None) -> Optional[T_co]:
    """
    Resolve optional type for mypy and check type at runtime.

    Args:
        t: Type to enforce, e.g. `str`.
        o: Object to check for type, e.g. 'abc'.
        msg: Custom error message.

    Raises:
        TypeError: If type of `o` doesn't match `t`.

    Returns:
        Optional[T_co]: Return `o` (pass-through) if its type is correct or it is None.
    """
    if o is None:
        return o
    verified_o: T_co = enforce_type(t, o, msg=msg)
    return verified_o


@torch.jit.unused
def enforce_not_none_type(o: Optional[T_co], msg: Optional[str] = None) -> T_co:
    """Resolve type for mypy and check type at runtime.

    Args:
        o: Object to check for type, e.g. 'abc'.
        msg: Custom error message.

    Raises:
        TypeError: If `o` is `None`.

    Returns:
        T_co: Return `o` (pass-through) if `o` is not None.

    .. note::
        This is useful when dealing with optional types.
    """
    if o is None:
        msg = msg if msg is not None else 'Expected not None.'
        raise TypeError(msg)
    return o
