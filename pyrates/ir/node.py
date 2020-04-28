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
"""
"""
from copy import copy, deepcopy
from typing import Iterator

import numpy as np

from pyrates.ir.abc import AbstractBaseIR
from pyrates.ir.operator_graph import OperatorGraph, VectorizedOperatorGraph, cache_op_graph

__author__ = "Daniel Rose"
__status__ = "Development"


class NodeIR(AbstractBaseIR):
    __slots__ = ["_op_graph", "values"]

    def __init__(self, operators: dict = None, values: dict = None, template: str = None):
        super().__init__(template)
        self._op_graph, changed_labels = cache_op_graph(OperatorGraph)(operators)
        # ToDo: Move caching function to NodeIR instead of using a decorator, for clarity
        try:
            for old_name, new_name in changed_labels.items():
                values[new_name] = values.pop(old_name)
        except AttributeError:
            pass

        self.values = values

    @property
    def op_graph(self):
        return self._op_graph

    def getitem_from_iterator(self, key: str, key_iter: Iterator[str]):
        """Alias for self.op_graph.getitem_from_iterator"""

        return self.op_graph.getitem_from_iterator(key, key_iter)

    def __iter__(self):
        """Return an iterator containing all operator labels in the operator graph."""
        return iter(self.op_graph)

    @property
    def operators(self):
        return self.op_graph.operators

    def __hash__(self):
        raise NotImplementedError


class VectorizedNodeIR(AbstractBaseIR):
    """Alternate version of NodeIR that takes a full NodeIR as input and creates a vectorized form of it."""

    __slots__ = ["op_graph", "_length"]

    def __init__(self, node_ir: NodeIR):

        super().__init__(node_ir.template)
        self.op_graph = VectorizedOperatorGraph(node_ir.op_graph, node_ir.values)
        values = {}
        # reformat all values to be lists of themselves (adding an outer vector dimension)
        # if len(node_ir.op_graph) == 0:
        #     op_key, data = next(iter(self.op_graph.node(data=True)))
        #     for var in data["variables"]:
        #         values[op_key] = {var: [0.]}
        # else:

        # save current length of this node vector.
        self._length = 1

    def getitem_from_iterator(self, key: str, key_iter: Iterator[str]):
        """Alias for self.op_graph.getitem_from_iterator"""

        return self.op_graph.getitem_from_iterator(key, key_iter)

    def __iter__(self):
        """Return an iterator containing all operator labels in the operator graph."""
        return iter(self.op_graph)

    @property
    def operators(self):
        return self.op_graph.operators

    def __hash__(self):
        raise NotImplementedError

    def extend(self, node: NodeIR):
        """ Extend variables vectors by values from one additional node.

        Parameters
        ----------
        node
            A node whose values are used to extend the vector dimension of this vectorized node.

        Returns
        -------
        """

        # add values to respective lists in collapsed node
        self.op_graph.append_values(node.values)

        self._length += 1

    def __len__(self):
        """Returns size of this vector node as recorded in self._length.

        Returns
        -------
        self._length
        """
        return self._length

    def add_op(self, op_key: str, inputs: dict, output: str, equations: list, variables: dict):
        """Wrapper for internal `op_graph.add_operator` that adds any values to node-level values dictionary for quick
        access

        Parameters
        ----------
        op_key
            Name of operator to be added
        inputs
            dictionary definining input variables of the operator
        output
            string defining name of single output variable
        equations
            list of equations (strings)
        variables
            dictionary describing variables

        Returns
        -------

        """

        # add operator to op_graph
        self.op_graph.add_operator(op_key, inputs=inputs, output=output, equations=equations, variables=variables)

    def add_op_edge(self, source_op_key: str, target_op_key: str, **attr):
        """ Alias to `self.op_graph.add_edge`

        Parameters
        ----------
        source_op_key
        target_op_key
        attr

        Returns
        -------

        """

        self.op_graph.add_edge(source_op_key, target_op_key, **attr)
