
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
from pyrates.frontend import CircuitTemplate, NodeTemplate, EdgeTemplate, OperatorTemplate
from pyrates.backend.parser import get_unique_label
from typing import Union

__author__ = "Daniel Rose, Richard Gast"
__status__ = "Development"


def from_circuit(circuit: CircuitTemplate, return_dict: dict, base: str = 'CircuitTemplate') -> str:
    """Reformat graph structure into a dictionary that can be saved as YAML template. The current implementation assumes
    that nodes and edges are given by as templates."""

    new_dict = {'base': base, 'circuits': {}, 'nodes': {}, 'edges': []}

    if circuit.circuits:

        # collect circuit definitions
        for key, c in circuit.circuits.items():
            ckey_tmp = from_circuit(c, return_dict=return_dict)
            new_dict['circuits'][key] = ckey_tmp

    else:

        # collect node definitions
        for key, n in circuit.nodes.items():
            nkey = from_node(n, return_dict=return_dict)
            new_dict['nodes'][key] = nkey

    # collect edge definitions
    for edge in circuit.edges:
        edge = list(edge)
        edge_temp = edge[2]  # type: EdgeTemplate
        edge[2] = from_edge(edge_temp, return_dict=return_dict)
        new_dict['edges'].append(tuple(edge))

    # add the circuit information to the return dict
    return add_to_dict(circuit, new_dict, return_dict)


def from_node(node: Union[NodeTemplate, EdgeTemplate], return_dict: dict, base: str = 'NodeTemplate') -> str:
    """Reformat operator structure into a dictionary that can be saved as YAML template.
    """

    new_dict = {'base': base, 'operators': []}

    # collect operator definitions
    for op, updates in node.operators.items():
        opkey = from_operator(op=op, updates=updates, return_dict=return_dict)
        new_dict['operators'].append(opkey)

    # add node information to the return dictionary
    return add_to_dict(node, new_dict, return_dict)


def from_operator(op: OperatorTemplate, updates: dict, return_dict: dict, base: str = 'OperatorTemplate') -> str:
    """Reformat operator template into a dictionary that can be saved as YAML template.
    """

    # collect operator attributes
    new_dict = {'base': base, 'equations': op.equations, 'variables': op.variables}
    new_dict['variables'].update(updates)

    # add operator definition to the return dictionary
    return add_to_dict(op, new_dict, return_dict)


def from_edge(edge: EdgeTemplate, return_dict: dict, base: str = 'EdgeTemplate') -> Union[str, None]:
    """Reformat edge template into a list that can be saved as YAML template.
    """
    if edge is None:
        return edge
    return from_node(edge, return_dict=return_dict, base=base)


def add_to_dict(template, template_dict: dict, full_dict: dict):

    temp_key = template.name
    if temp_key in full_dict and full_dict[temp_key] != template_dict:
        temp_key = get_unique_label(temp_key, list(full_dict.keys()))
    full_dict[temp_key] = template_dict
    return temp_key
