
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


__author__ = "Daniel Rose, Richard Gast"
__status__ = "Development"


def from_circuit(circuit: CircuitTemplate):
    """Reformat graph structure into a dictionary that can be saved as YAML template. The current implementation assumes
    that nodes and edges are given by as templates."""

    return_dict = dict()

    # collect circuits, nodes and edges
    if circuit.circuits:
        return_dict['circuits'] = {key: from_circuit(template) for key, template in circuit.circuits.items()}
    else:
        return_dict['nodes'] = {key: from_node(template) for key, template in circuit.nodes.items()}
    return_dict['edges'] = [from_edge(edge) for edge in circuit.edges]

    # use Python template as base, since inheritance from YAML templates is ambiguous for circuits
    return_dict['base'] = "CircuitTemplate"

    return return_dict


def from_node(node: NodeTemplate):
    """Reformat operator structure into a dictionary that can be saved as YAML template.
    """
    operators = {op.name: from_operator(op, updates) for op, updates in node.operators.items()}
    base = "NodeTemplate"
    return dict(operators=operators, base=base, name=node.name)


def from_operator(op: OperatorTemplate, updates):
    """Reformat operator template into a dictionary that can be saved as YAML template.
    """
    equations = deepcopy(op.equations)
    variables = deepcopy(op.variables)
    variables.update(updates)
    base = "OperatorTemplate"
    return dict(equations=equations, variables=variables, base=base)


def from_edge(edge: EdgeTemplate):
    """Reformat edge template into a list that can be saved as YAML template.
    """
    edge_new = list(edge)
    if edge_new[2] is None:
        return tuple(edge_new)
    edge_new[2] = from_node(edge_new[2])
    return tuple(edge_new)
