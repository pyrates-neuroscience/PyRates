
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
from pyrates.frontend import CircuitTemplate, NodeTemplate, EdgeTemplate, OperatorTemplate
from typing import Union

__author__ = "Daniel Rose, Richard Gast"
__status__ = "Development"


def from_circuit(circuit: CircuitTemplate, base: str = 'CircuitTemplate'):
    """Reformat graph structure into a dictionary that can be saved as YAML template. The current implementation assumes
    that nodes and edges are given by as templates."""

    return_dict = {circuit.name: {'base': base, 'circuits': {}, 'nodes': {}, 'edges': []}}

    # collect template definitions of operators, nodes, edges and circuits
    if circuit.circuits:

        for key, c in circuit.circuits.items():
            return_dict[circuit.name]['circuits'][key] = c.name
            return_dict.update(from_circuit(c))

    else:

        for key, n in circuit.nodes.items():
            return_dict[circuit.name]['nodes'][key] = n.name
            return_dict.update(from_node(n))

    for edge in circuit.edges:
        return_dict[circuit.name]['edges'].append(edge)
        edge_temp = edge[2]  # type: EdgeTemplate
        return_dict.update(from_edge(edge_temp))

    return return_dict


def from_node(node: Union[NodeTemplate, EdgeTemplate], base: str = 'NodeTemplate'):
    """Reformat operator structure into a dictionary that can be saved as YAML template.
    """

    return_dict = {node.name: {'base': base, 'operators': []}}

    for op, updates in node.operators.items():
        return_dict[node.name]['operators'].append(op.name)
        return_dict.update(from_operator(op=op, updates=updates))

    return return_dict


def from_operator(op: OperatorTemplate, updates, base: str = 'OperatorTemplate'):
    """Reformat operator template into a dictionary that can be saved as YAML template.
    """
    return_dict = {op.name: {'base': base,
                             'equations': deepcopy(op.equations),
                             'variables': deepcopy(op.variables)}}
    return_dict[op.name]['variables'].update(updates)
    return return_dict


def from_edge(edge: EdgeTemplate, base: str = 'EdgeTemplate'):
    """Reformat edge template into a list that can be saved as YAML template.
    """
    if edge is None:
        return {}
    return from_node(edge, base=base)
