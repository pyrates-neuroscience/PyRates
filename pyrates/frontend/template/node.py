
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
from pyrates.frontend.template.operator_graph import OperatorGraphTemplate
from pyrates.ir.node import cache_func, VectorizedNodeIR


class NodeTemplate(OperatorGraphTemplate):
    """Generic template for a node in the computational backend graph. A single node may encompass several
    different operators. One template defines a typical structure of a given node type."""

    @staticmethod
    def target_ir(label: str, operators: dict, values: dict = None, template: str = None, **kwargs):
        return cache_func(label, operators, values, template, VectorizedNodeIR, **kwargs)
