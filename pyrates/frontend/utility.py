from typing import Union


def deep_freeze(freeze: Union[dict, list, set, tuple]):
    """

    Parameters
    ----------
    freeze

    Returns
    -------
    frozen
    """

    if isinstance(freeze, dict):
        try:
            frozen = frozenset(freeze.items())
        except TypeError:
            temp = set()
            for key, item in freeze.items():
                temp.add((key, deep_freeze(item)))
            frozen = frozenset(temp)
    elif isinstance(freeze, list):
        try:
            frozen = tuple(freeze)
        except TypeError as e:
            # Don't know what to do
            raise e
    else:
        try:
            hash(freeze)
        except TypeError as e:
            raise e
        else:
            frozen = freeze

    return frozen