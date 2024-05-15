from typing import MutableMapping


def deep_update(d: MutableMapping, u: MutableMapping) -> MutableMapping:
    """
    Updates a dictionary where values are also dictionaries.
    """
    for k, v in u.items():
        if isinstance(v, MutableMapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
