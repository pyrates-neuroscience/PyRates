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
from typing import Iterator, Dict
from networkx import DiGraph, find_cycle, NetworkXNoCycle

from pyrates import PyRatesException

__author__ = "Daniel Rose"
__status__ = "Development"


class OperatorGraph(DiGraph):
    """Intermediate representation for nodes and edges."""

    def __init__(self, operators: dict = None, template: str = ""):

        super().__init__()
        if not operators:
            operators = {}
        all_outputs = {}  # type: Dict[str, dict]
        # op_inputs, op_outputs = set(), set()

        for key, item in operators.items():

            op_instance = item["operator"]
            op_variables = item["variables"]

            # add operator as node to local operator_graph
            self.add_node(key, operator=op_instance, variables=op_variables)

            # collect all output variables
            out_var = op_instance.output

            # check, if variable name exists in outputs and create empty list if it doesn't
            if out_var not in all_outputs:
                all_outputs[out_var] = {}

            all_outputs[out_var][key] = out_var
            # this assumes input and output variables map on each other by equal name
            # with additional information, non-equal names could also be mapped here

        # link outputs to inputs
        for op_key, op in self.operators(get_ops=True):
            for in_var in op.inputs:
                if in_var in all_outputs:
                    # link all collected outputs of given variable in inputs field of operator
                    for predecessor, out_var in all_outputs[in_var].items():
                        # add predecessor output as source; this would also work for non-equal variable names
                        if predecessor not in op.inputs[in_var]["sources"]:
                            op.inputs[in_var]["sources"].append(predecessor)
                        self.add_edge(predecessor, op_key)
                else:
                    pass  # means, that 'source' will remain an empty list and no incoming edge will be added

        try:
            find_cycle(self)
        except NetworkXNoCycle:
            pass
        else:
            raise PyRatesException("Found cyclic operator graph. Cycles are not allowed for operators within one node "
                                   "or edge.")

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
            item = self.nodes[key]["variables"][var]

        return item

    def __iter__(self):
        """Return an iterator containing all operator labels in the operator graph."""
        return iter(self.nodes)

    def operators(self, get_ops=False, get_vars=False):
        """Alias for self.nodes"""

        if get_ops and get_vars:
            return ((key, data["operator"], data["variables"]) for key, data in self.nodes(data=True))
        elif get_ops:
            return ((key, data["operator"]) for key, data in self.nodes(data=True))
        elif get_vars:
            return ((key, data["variables"]) for key, data in self.nodes(data=True))
        else:
            return self.nodes
