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
from typing import Iterator, Callable

from pyrates.ir.abc import AbstractBaseIR
from pyrates.ir.operator_graph import OperatorGraph, VectorizedOperatorGraph
from pyrates.backend.parser import get_unique_label

__author__ = "Daniel Rose"
__status__ = "Development"

node_cache = {}
op_cache = {}
node_labels = []


def clear_ir_caches():
    node_cache.clear()
    op_cache.clear()
    node_labels.clear()


def cache_func(label: str, operators: dict, values: dict = None, template: str = None, ir_class: Callable = None,
               **kwargs):
    if operators is None:
        operators = {}

    # compute hash from incoming operators. Different order of operators in input might lead to different hash.
    op_graph = OperatorGraph(operators)
    h = hash(op_graph)

    changed_labels = dict()
    vectorize = kwargs.pop('vectorize', True)
    try:

        # if vectorization is to be skipped, ignore cached IRs
        if not vectorize:
            raise KeyError

        # extract node from cache
        node = node_cache[h]

        # change operator labels if necessary
        for name, op in operators.items():
            op_hash = hash(op)
            for cached_name, cached_op in op_cache[h]:
                if op_hash == hash(cached_op["operator"]):
                    changed_labels[name] = cached_name
                    for old_name, new_name in changed_labels.items():
                        try:
                            values[new_name] = values.pop(old_name)
                        except (AttributeError, KeyError):
                            pass

        # extend cached node
        var_ranges = node.extend(NodeIR(label, operators=op_graph, values=values, template=template), **kwargs)

    except KeyError:

        # create new node
        op_cache[h] = op_graph
        if label in node_labels:
            label = get_unique_label(label, node_labels)
        node = ir_class(label, operators=op_graph, values=values, template=template, **kwargs)
        node_cache[h] = node
        node_labels.append(label)
        var_ranges = {key: (0, val) for key, val in node.op_graph.var_lengths.items()}

    return node, changed_labels, var_ranges


class NodeIR(AbstractBaseIR):
    __slots__ = ["_op_graph", "values"]

    def __init__(self, label: str, operators: OperatorGraph, values: dict = None, template: str = None):
        super().__init__(label, template)
        self._op_graph = operators
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

    __slots__ = ["op_graph", "length", "_var_lengths"]

    def __init__(self, label: str, operators: OperatorGraph, values: dict = None, template: str = None):

        super().__init__(label, template)

        self.op_graph = VectorizedOperatorGraph(operators, values=values)

        # save current length of this node vector.
        self.length = 1

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

    def extend(self, node: NodeIR) -> dict:
        """ Extend variables vectors by values from one additional node.

        Parameters
        ----------
        node
            A node whose values are used to extend the vector dimension of this vectorized node.

        Returns
        -------
        dict
            Dictionary containing the indices of the appended variable values in the overall vectorized variables.
        """

        # add values to respective lists in collapsed node
        var_ranges = self.op_graph.append_values(node.values)
        self.length += 1
        return var_ranges

    def __len__(self):
        """Returns size of this vector node as recorded in self._length.

        Returns
        -------
        self._length
        """
        return self.length

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
