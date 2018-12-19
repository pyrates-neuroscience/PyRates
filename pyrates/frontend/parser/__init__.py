
# -*- coding: utf-8 -*-
#
#
# PyRates software framework for flexible implementation of neural 
# network models and simulations. See also: 
# https://github.com/pyrates-neuroscience/PyRates
# 
# Copyright (C) 2017-2018 the original authors (Richard Gast and 
# Daniel Rose), the Max-Planck-Institute for Human Cognitive Brain 
# Sciences ("MPI CBS") and contributors
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
# 
# CITATION:
# 
# Richard Gast and Daniel Rose et. al. in preparation
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
