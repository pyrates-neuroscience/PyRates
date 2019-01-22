
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

__author__ = "Daniel Rose"
__status__ = "Development"


def test_move_edge_ops_to_nodes():
    """Test, if apply() functions all work properly"""

    path = "pyrates.examples.jansen_rit.circuit.JansenRitCircuit"
    from pyrates.frontend.template.circuit import CircuitTemplate
    from pyrates.ir.circuit import CircuitIR

    template = CircuitTemplate.from_yaml(path)

    circuit = template.apply()  # type: CircuitIR
    circuit2 = circuit.move_edge_operators_to_nodes()

    for source, target, data in circuit2.edges(data=True):
        # check that no operators are left in the edges of the rearranged circuit
        assert len(data["edge_ir"].op_graph) == 0

        # check that operator from previous edges is indeed in target nodes
        # original_edge = circuit.edges[(source, target, 0)]["edge_ir"]
        # original_op = list(original_edge.op_graph.nodes)[0]
        # assert f"{original_op}.0" in circuit2[target]
