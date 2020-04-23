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
from copy import deepcopy

from pyrates.ir.circuit import CircuitIR
from pyrates.ir.node import NodeIR
from pyrates.ir.edge import EdgeIR
from pyrates.ir.operator import OperatorIR

__author__ = "Daniel Rose"
__status__ = "Development"


def from_circuit(circuit: CircuitIR, deep: bool = False):
    """Reformat graph structure into a dictionary that can be saved as YAML template. The current implementation assumes
    that nodes and edges are given by as templates.

    Parameters
    ----------
    circuit
        An instance of CircuitIR to transform to a dictionary
    deep
        Toggles whether to only include references to node/edge templates (`deep=False`) or also transform these to
        dictionaries as well (`deep=True`).

    """

    node_dict = {}
    node_templates = {}
    for node_key, node_data in circuit.nodes(data=True):
        node = node_data["node"]
        if deep:
            template_name, node = node.to_dict(deep)
            node_dict[node_key] = template_name
            node_templates[template_name] = node
        else:
            node_dict[node_key] = node.template

    edge_list = []
    for source, target, edge_data in circuit.edges(data=True):
        edge_data = deepcopy(edge_data)
        edge = edge_data.pop("edge_ir")
        if deep:
            edge = edge.to_dict(deep)
        else:
            edge = edge.template

        source = f"{source}/{edge_data['source_var']}"
        target = f"{target}/{edge_data['target_var']}"
        edge_list.append((source, target, edge, dict(weight=edge_data["weight"],
                                                     delay=edge_data["delay"])))

    # use Python template as base, since inheritance from YAML templates is ambiguous for circuits
    base = "CircuitTemplate"

    return dict(nodes=node_dict, edges=edge_list, base=base)  # , description=circuit.__doc__)


def from_node(node: NodeIR, deep: bool = False):
    """Transform NodeIR instance into a dictionary that can be used to create a NodeTemplate.

    Parameters
    ----------
    node
        The node parse.
    deep
        Whether or not to also parse included operators. If 'False', then operators will just be included by name.
    """

    node_dict = dict(base="NodeTemplate")
    # description=node.__doc__)

    if node.template:
        name

    operators = []


def from_edge(edge: EdgeIR):
    pass


def from_operator(op: OperatorIR):
    pass
