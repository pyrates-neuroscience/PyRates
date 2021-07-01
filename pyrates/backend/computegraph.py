# -*- coding: utf-8 -*-
#
#
# PyRates software framework for flexible implementation of neural
# network model_templates and simulations. See also:
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

"""This module provides the backend class that should be used to set up any backend in pyrates.
"""

# external imports
from typing import Optional, Any, Callable, Union
from networkx import MultiDiGraph
from sympy import Symbol, Expr

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


class ComputeGraph(MultiDiGraph):
    """Creates a compute graph where nodes are all constants and variables of the network and edges are the mathematical
    operations linking those variables/constants together to form equations.
    """

    def add_var(self, label: str, symbol: Union[Symbol, Expr], value: Any, **kwargs):

        super().add_node(label, symbol=symbol, value=value, **kwargs)
        return self[label]

    def add_op(self, inputs: Union[list, tuple], label: str, expr: str, func: Callable, **kwargs):

        # add target node that contains result of operation
        super().add_node(label, expr=expr, func=func, **kwargs)

        # add edges from source nodes to target node
        for i, v in enumerate(inputs):
            super().add_edge(v, label, key=i)

        return self[label]

    def eval(self):

        # TODO: evaluate whole graph
        pass

    def to_str(self):

        # TODO: generate function string from compute graph
        pass

    def compile(self):

        # TODO: collect layers of nodes with 'func' attributes.
        pass
