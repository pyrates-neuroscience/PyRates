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
from typing import Iterator, Dict, List

import numpy as _np
from networkx import DiGraph, find_cycle, NetworkXNoCycle

from pyrates.backend import PyRatesException
from pyrates.ir.operator import OperatorIR

__author__ = "Daniel Rose"
__status__ = "Development"


class OperatorGraph(DiGraph):
    """Intermediate representation for nodes and edges."""

    def __init__(self, operators: Dict[str, OperatorIR] = None, template: str = ""):

        super().__init__()
        if operators is None:
            operators = {}

        # compute hash from incoming operators. Different order of operators in input might lead to different hash.
        self._h = hash(tuple(operators.values()))

        # collect all information about output variables of operators
        #############################################################
        all_outputs = {}  # type: Dict[str, List[str]]
        for key, operator in operators.items():

            # add operator as node to local operator_graph
            inputs = {var: dict(sources=set(), reduce_dim=True) for var in operator.inputs}
            self.add_node(key, operator=operator, inputs=inputs, label=key)

            # collect all output variables
            out_var = operator.output

            # check, if variable name exists in outputs and create empty list if it doesn't
            if out_var not in all_outputs:
                all_outputs[out_var] = []

            all_outputs[out_var].append(key)

        # link outputs to inputs
        ########################
        for label, data in self.nodes(data=True):
            op = data["operator"]
            for in_var in op.inputs:
                if in_var in all_outputs:
                    # link all collected outputs of given variable in inputs field of operator
                    for predecessor in all_outputs[in_var]:
                        # add predecessor output as source; this would also work for non-equal variable names
                        data["inputs"][in_var]["sources"].add(predecessor)
                        # adding into set means there will be no order, but also nothing can be added twice
                        self.add_edge(predecessor, label)
                else:
                    pass  # means, that 'source' will remain an empty list and no incoming edge will be added

        # check for cycle in operator graph
        ###################################
        try:
            find_cycle(self)
        except NetworkXNoCycle:
            pass
        else:
            raise PyRatesException("Found cyclic operator graph. Cycles are not allowed for operators within one node "
                                   "or edge.")

    def __hash__(self):
        return self._h

    def getitem_from_iterator(self, key: str, key_iter: Iterator[str]):
        """
        Helper function for Python magic __getitem__. Accepts an iterator that yields string keys. If `key_iter`
        contains one key, an operator will be (looked for and) returned. If it instead contains two keys, properties of
        a variable that belong to an operator is returned.

        Parameters
        ----------
        key
        key_iter

        Returns
        -------
        item
            operator or variable properties
        """

        try:
            var = next(key_iter)
        except StopIteration:
            # no variable specified, so we return an operator
            item = self.nodes[key]["operator"]
        else:
            # variable specified, so we return variable properties instead
            item = self.nodes[key]["operator"].variables[var]

        return item

    def __iter__(self):
        """Return an iterator containing all operator labels in the operator graph."""
        return iter(self.nodes(data=True))

    def operators(self, get_ops=False, get_vals=False):
        """Alias for self.nodes"""

        if get_ops and get_vals:
            return ((data["label"], op, data["values"]) for op, data in self.nodes(data=True))
        elif get_ops:
            return ((data["label"], op) for op, data in self.nodes(data=True))
        elif get_vals:
            return ((data["label"], data["values"]) for op, data in self.nodes(data=True))
        else:
            return self.nodes


class VectorizedOperatorGraph(DiGraph):
    """Alternate version of `OperatorGraph` that is produced during vectorization. Contents of this version are not
    particularly protected and the instance is not cached."""

    def __init__(self, op_graph: OperatorGraph = None, values: dict = None):

        super().__init__()

        if op_graph is None:
            pass
        else:
            for op_key, data in op_graph:
                try:
                    op = data["operator"]
                except KeyError:
                    # need to add this for the copy capability
                    self.add_operator(op_key, **data)
                else:
                    self.add_operator(op_key,
                                      inputs=data["inputs"],
                                      equations=list(op.equations),
                                      variables=op.variables.to_dict(),
                                      output=op.output)

                # retrieve values from value dict and pass them into variable dictionary of operator
                op_values = values[op_key]  #deepcopy(values[op_key])
                op_vars = self.operators[op_key]["variables"]
                for var_key, value in op_values.items():
                    if op_vars[var_key]["vtype"] == "input_variable":
                        op_vars[var_key]["value"] = value
                        op_vars[var_key]["vtype"] = "input"
                    else:
                        op_vars[var_key]["value"] = value if type(value) is list else [value]

            self.add_edges_from(op_graph.edges)

    def add_operator(self, *args, **kwargs):
        """Alias for `self.add_node`"""
        self.add_node(*args, **kwargs)

    def getitem_from_iterator(self, key: str, key_iter: Iterator[str]):
        """
        Helper function for Python magic __getitem__. Accepts an iterator that yields string keys. If `key_iter`
        contains one key, an operator will be (looked for and) returned. If it instead contains two keys, properties of
        a variable that belong to an operator is returned.

        Parameters
        ----------
        key
        key_iter

        Returns
        -------
        item
            operator or variable properties
        """

        try:
            var = next(key_iter)
        except StopIteration:
            # no variable specified, so we return an operator
            item = self.nodes[key]
        else:
            # variable specified, so we return variable properties instead
            item = self.nodes[key]["variables"][var]

        return item

    @property
    def operators(self):  # , get_ops=False, get_vals=False):
        """Alias for self.nodes"""

        # if get_ops and get_vals:
        #     return ((data["label"], op, data["values"]) for op, data in self.nodes(data=True))
        # elif get_ops:
        #     return ((data["label"], op) for op, data in self.nodes(data=True))
        # elif get_vals:
        #     return ((data["label"], data["values"]) for op, data in self.nodes(data=True))
        # else:
        return self.nodes

    def append_values(self, value_dict: dict) -> dict:
        """Append value along vector dimension of operators.

        Parameters
        ----------
        value_dict

        Returns
        -------
        dict
            Dictionary containing the indices of the appended variable values in the overall vectorized variables.
        """
        var_ranges = {}
        for op_key, variables_updates in value_dict.items():
            original_variables = self.nodes[op_key]["variables"]
            for var_key, value in variables_updates.items():
                var = original_variables[var_key]
                shape = _np.shape(value)
                shape_sum = _np.sum(shape)
                if shape_sum > 1:
                    if _np.all(shape == _np.shape(var["value"])[1:]):
                        raise ValueError(f"Inconsistent dimensions of variable {var}. Dimension of value to add: "
                                         f"{shape} Internal dimension of vectorized value array: "
                                         f"{_np.shape(var['value'])[1:]}")
                    var["value"].append(value)
                elif shape_sum == 0:
                    # case, if shape == ()
                    var["value"].append(value)
                else:
                    var["value"].extend(value)

                # also recompute shape
                old_shape = var["shape"]
                var["shape"] = _np.shape(var["value"])
                var_ranges[(op_key, var_key)] = (old_shape[0], var['shape'][0])

        return var_ranges

    @property
    def var_lengths(self) -> dict:
        """Dictionary containing the length of each variable of each operator.
        """
        var_lengths = {}
        for op in self.operators:
            for var, data in self.nodes[op]['variables'].items():
                var_lengths[(op, var)] = data['shape'][0] if data['shape'] else 1
        return var_lengths
