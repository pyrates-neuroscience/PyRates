
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


#
# def to_node(node_dict: dict):
#
#     order = node_dict["operator_order"]
#     variables = {}
#     # collect all information about variables and separate operator and variable keys
#     for key, var_dict in node_dict["operator_args"].items():
#         op_key, var_key = key.split("/")  # split key as 'op_key/var_key'
#         if op_key not in variables:
#             variables[op_key] = {}
#
#         variables[op_key][var_key] = var_dict
#
#     operators = []
#     for op_key in order:
#         op_dict = node_dict["operators"][op_key]
#         op_dict["name"] = op_key
#         op_dict["path"] = f"custom/{op_key}"
#         op_dict["variables"] = variables[op_key]
#         inputs = op_dict.pop("inputs")
#         for var in inputs:
#             op_dict["variables"][var]["vtype"] = "input"
#         outvar = op_dict.pop("output")
#         op_dict["variables"][outvar]["vtype"] = "output"
#         operators.append(to_operator(op_dict))
#
#     return NodeIR(operators=operators)
#
#
#
# def to_operator(op_dict: dict):
#     from pyrates.frontend import OperatorTemplate
#     template = OperatorTemplate(**op_dict)
#
#     return template.apply()
#


def from_circuit(circuit: CircuitIR):
    """Reformat graph structure into a dictionary that can be saved as YAML template. The current implementation assumes
    that nodes and edges are given by as templates."""

    node_dict = {}
    for node_key, node_data in circuit.nodes(data=True):
        node = node_data["node"]
        if node.template:
            node_dict[node_key] = node.template
        else:
            # if no template is given, build and search deeper for node templates
            pass

    edge_list = []
    for source, target, edge_data in circuit.edges(data=True):
        edge_data = deepcopy(edge_data)
        edge = edge_data.pop("edge_ir")
        source = f"{source}/{edge_data['source_var']}"
        target = f"{target}/{edge_data['target_var']}"
        edge_list.append((source, target, edge.template, dict(weight=edge_data["weight"],
                                                              delay=edge_data["delay"])))

    # use Python template as base, since inheritance from YAML templates is ambiguous for circuits
    base = "CircuitTemplate"

    return dict(nodes=node_dict, edges=edge_list, base=base)
