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
from pyrates.frontend import template
from pyrates.frontend import dict as dict_
from pyrates.frontend import yaml
from pyrates.frontend import nxgraph
from pyrates.frontend.template import CircuitTemplate, NodeTemplate, EdgeTemplate, OperatorTemplate

# By importing the above, all transformation functions (starting with `to_` or `from_`) are registered
# Below these functions are collected and made available from pyrates.frontend following the naming convention
# `{target}_from_{source}` with target and source being respective valid representations in the frontend

from pyrates.frontend._registry import REGISTERED_INTERFACES
import sys

# add all registered functions to main frontend module
this_module = sys.modules[__name__]

for new_name, func in REGISTERED_INTERFACES.items():
    # set new name on current module
    setattr(this_module, new_name, func)


# The following function are shorthands that bridge multiple interface steps

def circuit_from_yaml(path: str):
    """Directly return CircuitIR instance from a yaml file."""
    return CircuitTemplate.from_yaml(path).apply()


def node_from_yaml(path: str):
    """Directly return NodeIR instance from a yaml file."""
    return NodeTemplate.from_yaml(path).apply()


def edge_from_yaml(path: str):
    """Directly return EdgeIR instance from a yaml file."""

    return EdgeTemplate.from_yaml(path).apply()


def operator_from_yaml(path: str):
    """Directly return OperatorIR instance from a yaml file."""

    return OperatorTemplate.from_yaml(path).apply()
