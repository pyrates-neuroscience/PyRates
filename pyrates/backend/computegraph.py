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
from typing import Any, Callable, Union
from networkx import MultiDiGraph
from sympy import Symbol, Expr, Function
from copy import deepcopy
import numpy as np

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


#########################
# compute graph classes #
#########################


class ComputeGraph(MultiDiGraph):
    """Creates a compute graph where nodes are all constants and variables of the network and edges are the mathematical
    operations linking those variables/constants together to form equations.
    """

    def __init__(self, **kwargs):

        self._out_nodes = []
        super().__init__(**kwargs)

    def add_var(self, label: str, value: Any, vtype: str, **kwargs):

        unique_label = self._generate_unique_label(label)
        symbol = Symbol(unique_label)
        super().add_node(unique_label, symbol=symbol, value=value, vtype=vtype, **kwargs)
        return unique_label, self.nodes[unique_label]

    def add_op(self, inputs: Union[list, tuple], label: str, expr: Expr, func: Callable, vtype: str, **kwargs):

        # add target node that contains result of operation
        unique_label = self._generate_unique_label(label)
        symbol = Symbol(unique_label)
        super().add_node(unique_label, expr=expr, func=func, vtype=vtype, symbol=symbol, **kwargs)

        # add edges from source nodes to target node
        for i, v in enumerate(inputs):
            super().add_edge(v, unique_label, key=i)

        return unique_label, self.nodes[unique_label]

    def eval_nodes(self, nodes: list):

        return [self.eval_node(n) for n in nodes]

    def compile(self, in_place: bool = True):

        G = self if in_place else deepcopy(self)

        # remove unconnected nodes and constants from graph
        #G._prune()

        # evaluate constant-based operations
        self._out_nodes = [node for node, out_degree in G.out_degree if out_degree == 0]
        for node in self._out_nodes:

            # process inputs of node
            for inp in G.predecessors(node):
                if G.nodes[inp]['vtype'] == 'constant':
                    G.eval_subgraph(inp)

            # evaluate node if all its inputs are constants
            if all([G.nodes[inp]['vtype'] == 'constant' for inp in G.predecessors(node)]):
                G.eval_subgraph(node)

        # broadcast all variable shapes to a common number of dimensions
        for node in self._out_nodes:
            G.broadcast_op_inputs(node, squeeze=False)

        return G

    def eval_subgraph(self, n):

        inputs = []
        input_nodes = [node for node in self.predecessors(n)]
        for inp in input_nodes:
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

    def broadcast_op_inputs(self, n, squeeze: bool = False, target_shape: tuple = None):

        try:

            # attempt to perform graph operation
            if target_shape:
                raise ValueError
            return self.eval_node(n)

        except ValueError:

            # collect inputs to operator node
            inputs, nodes, shapes = [], [], []
            for inp in self.predecessors(n):

                # ensure the whole operation tree of input is broadcasted to matching shapes
                inp_eval = self.broadcast_op_inputs(inp)

                inputs.append(inp_eval)
                nodes.append(inp)
                shapes.append(len(inp_eval.shape) if hasattr(inp_eval, 'shape') else 1)

            # broadcast shapes of inputs
            if not target_shape:
                target_shape = inputs[np.argmax(shapes)].shape
            for i in range(len(inputs)):

                inp_eval = inputs[i]

                # get new shape of input
                new_shape = self._broadcast_shapes(target_shape, inp_eval.shape, squeeze=squeeze)

                # reshape input
                if new_shape != inp_eval.shape:
                    if 'func' in self.nodes[nodes[i]]:
                        self.broadcast_op_inputs(nodes[i], squeeze=squeeze, target_shape=new_shape)
                    else:
                        inp_eval = inp_eval.reshape(new_shape)
                        self.nodes[nodes[i]]['value'] = inp_eval

            return self.broadcast_op_inputs(n, squeeze=True)

    def node_to_expr(self, n: str, **kwargs) -> tuple:

        expr_args = []
        try:
            expr = self.nodes[n]['expr']
            expr_info = [(self.nodes[inp]['symbol'], self.node_to_expr(inp, **kwargs)) for inp in self.predecessors(n)]
            for expr_old, (args, expr_new) in expr_info:
                expr = expr.replace(expr_old, expr_new)
                expr_args.extend(args)
            for expr_old, expr_new in kwargs.items():
                expr = expr.replace(Function(expr_old), Function(expr_new))
            return expr_args, expr
        except KeyError:
            if self.nodes[n]['vtype'] in ['constant', 'input']:
                expr_args.append(n)
            return expr_args, self.nodes[n]['symbol']

    def eval_node(self, n):

        inputs = [self.eval_node(inp) for inp in self.predecessors(n)]
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

    @staticmethod
    def _broadcast_shapes(s1: tuple, s2: tuple, squeeze: bool):

        new_shape = []
        for j, s in enumerate(s1):
            if squeeze:
                try:
                    if s2[j] == s1[j] or s2[j] == 1:
                        new_shape.append(s1[j])
                    else:
                        new_shape.append(1)
                except IndexError:
                    pass
            else:
                try:
                    if s2[j] == s1[j] or s1[j] == 1:
                        new_shape.append(s2[j])
                    else:
                        new_shape.append(1)
                except IndexError:
                    new_shape.append(1)

        return tuple(new_shape)
