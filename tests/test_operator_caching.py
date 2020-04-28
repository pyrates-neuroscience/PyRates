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
Test suite to determine, whether caching of identical operator works properly.
"""

__author__ = "Daniel Rose"
__status__ = "Development"

import pytest


def setup_module():
    print("\n")
    print("================================")
    print("| Test Suite: Operator Caching |")
    print("================================")


def test_op_caching_nodes():
    """Test the case that two operator templates are identical up to variable definitions, where one inherits from the
    other"""
    from pyrates.frontend import CircuitTemplate

    circuit = CircuitTemplate.from_yaml("model_templates.test_resources.test_operator_caching_templates.TestCircuit1").apply()

    # access nodes
    node1 = circuit["node1"]
    node2 = circuit["node2"]

    # verify that operator labels and contents are identical
    op1, op1_dict = next(iter(node1.op_graph.nodes(data=True)))
    op2, op2_dict = next(iter(node2.op_graph.nodes(data=True)))

    assert op1 == op2
    assert op1_dict == op2_dict
    assert op1_dict["operator"] is op2_dict["operator"]

    # verify that values in nodes are indeed different (but refer to same operator)
    assert node1.values["TestOpLin0"]["c"] == 0
    assert node2.values["TestOpLin0"]["c"] == 1

    # now do the vectorization step
    circuit = circuit.optimize_graph_in_place()

# def test_op_caching_edges():
# still need to implement a test for edges, because they behave a little different.








