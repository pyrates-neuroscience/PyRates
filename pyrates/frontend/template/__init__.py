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

from .node import NodeTemplate
from .operator import OperatorTemplate
from .edge import EdgeTemplate
from .circuit import CircuitTemplate
from pyrates.frontend._registry import register_interface


# module-level functions for template conversion
# writing them out explicitly
@register_interface
def to_circuit(template: CircuitTemplate):
    """Takes a circuit template and returns a CircuitIR instance from it."""
    return template.apply()


@register_interface
def to_node(template: NodeTemplate):
    """Takes a node template and returns a NodeIR instance from it."""
    return template.apply()


@register_interface
def to_edge(template: EdgeTemplate):
    """Takes a edge template and returns a EdgeIR instance from it."""
    return template.apply()


@register_interface
def to_operator(template: OperatorTemplate):
    """Takes a operator template and returns a OperatorIR instance from it."""
    return template.apply()
