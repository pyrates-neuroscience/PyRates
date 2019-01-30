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


#############################################################################################################
# Define shortcuts for different types of input that are defined as frontends to provide a common interface
# to all implemented type conversions.
# All functions are renamed explicitly
#############################################################################################################

# template-based interface
from .template import CircuitTemplate, NodeTemplate, EdgeTemplate, OperatorTemplate
from .template import to_circuit as circuit_from_template
from .template import to_node as node_from_template
from .template import to_edge as edge_from_template
from .template import to_operator as operator_from_template

# (Legacy) conversion from networkx.MultiDiGraph to a circuit
from .nxgraph import to_circuit as circuit_from_nxgraph
from .nxgraph import from_circuit as nxgraph_from_circuit

# reading a template from file
# from .file import to_template as template_from_file

# YAML-based interface
from .yaml import to_template_dict as template_dict_from_yaml_file
from .yaml import from_circuit as yaml_from_circuit
from .yaml import to_template as template_from_yaml_file

# dict-based interface
from .dict import from_circuit as dict_from_circuit
from .dict import to_node as node_from_dict
from .dict import to_operator as operator_from_dict


def circuit_from_yaml(path: str):
    """Directly return CircuitIR instance from a yaml file."""
    return template_from_yaml_file(path, CircuitTemplate).apply()


def node_from_yaml(path: str):
    """Directly return NodeIR instance from a yaml file."""
    return template_from_yaml_file(path, NodeTemplate).apply()


def edge_from_yaml(path: str):
    """Directly return EdgeIR instance from a yaml file."""

    return template_from_yaml_file(path, EdgeTemplate).apply()


def operator_from_yaml(path: str):
    """Directly return OperatorIR instance from a yaml file."""

    return template_from_yaml_file(path, OperatorTemplate).apply()
