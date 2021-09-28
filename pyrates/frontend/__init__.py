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
from pyrates.frontend.fileio import yaml
from pyrates.frontend.template import CircuitTemplate, NodeTemplate, EdgeTemplate, OperatorTemplate

#external imports
from typing import Union


def clear_frontend_caches(clear_template_cache=True, clear_operator_cache=True):
    """Utility to clear caches in the frontend.

    Parameters
    ----------
    clear_template_cache
        toggles whether or not to clear the template_cache that contains all previously loaded templates
    clear_operator_cache
        toggles whether or not to clear the cache of unique OperatorIR instances
    """
    if clear_template_cache:
        template.clear_cache()

    if clear_operator_cache:
        OperatorTemplate.cache.clear()


# The following function are shorthands that bridge multiple interface steps
def circuit_from_yaml(path: str):
    """Directly return CircuitIR instance from a yaml file."""
    return CircuitTemplate.from_yaml(path)


def simulate(circuit: Union[str, CircuitTemplate], **kwargs):
    """Directly simulate dynamics of a circuit."""
    if type(circuit) is str:
        circuit = circuit_from_yaml(path=circuit)
    results = circuit.run(**kwargs)
    if 'clear' in kwargs and kwargs['clear']:
        clear_frontend_caches()
    return results


def node_from_yaml(path: str):
    """Directly return NodeIR instance from a yaml file."""
    return NodeTemplate.from_yaml(path).apply()


def edge_from_yaml(path: str):
    """Directly return EdgeIR instance from a yaml file."""

    return EdgeTemplate.from_yaml(path).apply()


def operator_from_yaml(path: str):
    """Directly return OperatorIR instance from a yaml file."""

    return OperatorTemplate.from_yaml(path).apply()
