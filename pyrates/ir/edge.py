
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
from typing import List

from pyrates.backend import PyRatesException
from pyrates.ir.node import VectorizedNodeIR, NodeIR
from pyrates.ir.operator_graph import OperatorGraph

__author__ = "Daniel Rose"
__status__ = "Development"


class EdgeIR(VectorizedNodeIR):

    __slots__ = VectorizedNodeIR.__slots__ + ["_inputs"]

    def __init__(self, label: str, operators: OperatorGraph = None, values: dict = None, template: str = None):

        if not operators:
            operators = {}
            values = {}

        super().__init__(label=label, operators=operators, values=values, template=template)

        self._inputs = None
        self.length = 1

    @property
    def inputs(self):
        """Detect input variables of edge. This also references the operator
         the variable belongs to.

         Note: As of `0.9.0` multiple source/input variables are allowed per edge."""

        inputs = self.get_inputs()

        if len(inputs) == 0:
            return None
        else:
            return inputs

    @property
    def n_inputs(self):
        """Computes number of input variables that need to be informed from outside the edge (meaning from some node).
        """
        inputs = self.get_inputs()
        return len(inputs)

    def get_inputs(self):
        """Find all input variables of operators that need to be mapped to source variables in a source node."""
        try:
            len(self._inputs)
        except TypeError:
            # find inputs
            # noinspection PyTypeChecker
            # in_op = [op for op, in_degree in self.op_graph.in_degree if in_degree == 0]  # type: List[str]
            inputs = dict()
            # alternative:
            for op, op_data in self.op_graph.nodes(data=True):
                for var, var_data in op_data["inputs"].items():
                    if len(var_data["sources"]) == 0:
                        key = f"{op}/{var}"
                        try:
                            inputs[var].append(key)
                        except KeyError:
                            inputs[var] = [key]

            self._inputs = inputs
        return self._inputs

    @property
    def output(self):
        """Detect output variable of edge, assuming only one output variable exists."""

        # try to find single output variable
        # noinspection PyTypeChecker
        out_op = [op for op, out_degree in self.op_graph.out_degree if out_degree == 0]  # type: List[str]

        # only one single output operator allowed
        if len(out_op) == 1:
            out_var = self[out_op[0]]['output']
            return f"{self.label}/{out_op[0]}/{out_var}"
        elif len(out_op) == 0:
            return None
        else:
            raise PyRatesException("Too many or too little output operators found. Exactly one output operator and "
                                   "associated output variable is required per edge.")

    def __hash__(self):
        raise NotImplementedError

