
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

from pyrates.frontend.template.circuit import CircuitTemplate
from pyrates.frontend.template.node import NodeTemplate
from pyrates.frontend.template.edge import EdgeTemplate

from .nxgraph import to_circuit as circuit_from_nxgraph
from .nxgraph import from_circuit as nxgraph_from_circuit

from .file import to_template as template_from_file

from .yaml import to_template as template_from_yaml
from .yaml import from_circuit as yaml_from_circuit

from .dict import from_circuit as dict_from_circuit
from .dict import to_node as node_from_dict
from .dict import to_operator as operator_from_dict
