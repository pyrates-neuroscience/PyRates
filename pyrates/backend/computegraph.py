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
from networkx import MultiDiGraph, DiGraph
from sympy import Symbol, Expr
from copy import deepcopy

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


class ComputeGraph(MultiDiGraph):
    """Creates a compute graph where nodes are all constants and variables of the network and edges are the mathematical
    operations linking those variables/constants together to form equations.
    """

    def __init__(self):

        self._eval_nodes = []
        super().__init__()

    def add_var(self, label: str, symbol: Union[Symbol, Expr], value: Any, vtype: str, **kwargs):

        unique_label = self._generate_unique_label(label)
        super().add_node(unique_label, symbol=symbol, value=value, vtype=vtype, **kwargs)
        return unique_label, self.nodes[unique_label]

    def add_op(self, inputs: Union[list, tuple], label: str, expr: str, func: Callable, vtype: str, **kwargs):

        # add target node that contains result of operation
        unique_label = self._generate_unique_label(label)
        super().add_node(unique_label, expr=expr, func=func, vtype=vtype, **kwargs)

        # add edges from source nodes to target node
        for i, v in enumerate(inputs):
            super().add_edge(v, unique_label, key=i)

        # TODO: add automatic broadcasting of operator inputs via backend functions here

        return unique_label, self.nodes[unique_label]

    def eval(self):

        return [self._eval_node(n) for n in self._eval_nodes]

    def to_str(self) -> str:

        # TODO: generate function string from compute graph
        return ""

    def compile(self, lambdify: bool = True, to_file: bool = False, in_place: bool = True):

        G = self if in_place else deepcopy(self)

        # remove unconnected nodes and constants from graph
        G._prune()

        # evaluate constant-based operations
        self._eval_nodes = [node for node, out_degree in G.out_degree if out_degree == 0]
        for node in self._eval_nodes:

            # process inputs of node
            for inp in G.predecessors(node):
                if G.nodes[inp]['vtype'] == 'constant':
                    G.eval_subgraph(inp)

            # evaluate node if all its inputs are constants
            if all([G.nodes[inp]['vtype'] == 'constant' for inp in G.predecessors(node)]):
                G.eval_subgraph(node)

        # TODO: lambdify graph
        if lambdify:
            pass

        # TODO: write graph to function file
        if to_file:
            func_str = self.to_str()

        return G

    def eval_subgraph(self, n):

        inputs = []
        for inp in self.predecessors(n):
            inputs.append(self.eval_subgraph(inp))
            self.remove_node(inp)

        node = self.nodes[n]
        if inputs:
            node['value'] = node['func'](*tuple(inputs))

        return node['value']

    def remove_subgraph(self, n):

        for inp in self.predecessors(n):
            self.remove_subgraph(inp)
        self.remove_node(n)

    def _eval_node(self, n):

        inputs = [self._eval_node(inp) for inp in self.predecessors(n)]
        if inputs:
            return self.nodes[n]['func'](*tuple(inputs))
        return self.nodes[n]['value']

    def _prune(self):

        # remove all subgraphs that contain constants only
        for n in [node for node, out_degree in self.out_degree if out_degree == 0]:
            if self.nodes[n]['vtype'] == 'constant':
                self.remove_subgraph(n)

        # remove all unconnected nodes
        for n in [node for node, out_degree in self.out_degree if out_degree == 0]:
            if self.in_degree(n) == 0:
                self.remove_node(n)

    def _generate_unique_label(self, label: str):

        if label in self.nodes:
            label_split = label.split('_')
            try:
                new_label = "_".join(label_split[:-1] + [f"{int(label_split[-1])+1}"])
            except ValueError:
                new_label = f"{label}_0"
            return self._generate_unique_label(new_label)
        else:
            return label
