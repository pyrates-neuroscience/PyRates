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

import pytest

__author__ = "Daniel Rose, Richard Gast"
__status__ = "Development"


@pytest.mark.skip
def test_ir_vectorization():
    """Test, if apply() function works properly"""

    path = "model_templates.jansen_rit.circuit.JansenRitCircuit"
    from pyrates.ir.circuit import CircuitIR
    from pyrates.frontend import clear_frontend_caches
    clear_frontend_caches()

    circuit = CircuitIR.from_yaml(path)

    # ensure that the interneuron nodes of the JRC have been vectorized
    assert any([node['node'].length > 1 for _, node in circuit._ir.nodes.items()])
    circuit.clear()


@pytest.mark.skip
def test_ir_compilation():
    path = "model_templates.jansen_rit.circuit.JansenRitCircuit"
    from pyrates.ir.circuit import CircuitIR
    from pyrates.frontend import clear_frontend_caches
    clear_frontend_caches()

    circuit = CircuitIR.from_yaml(path)
    circuit._ir.backend.compile()
    circuit.clear()
