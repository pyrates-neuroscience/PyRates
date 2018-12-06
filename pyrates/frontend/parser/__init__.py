"""
"""
from typing import Union

import networkx as nx

from pyrates.frontend.parser.yaml import TemplateLoader

__author__ = "Daniel Rose"
__status__ = "Development"

type_mapping = {dict: "dictionary",
                nx.MultiDiGraph: "graph",
                str: "yaml"}

file_loader_mapping = {"yaml": TemplateLoader.load_template_from_yaml,
                       "yml": TemplateLoader.load_template_from_yaml}


# alternative:
class DictLoader: ...


loader_mapping = {dict: DictLoader}


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
