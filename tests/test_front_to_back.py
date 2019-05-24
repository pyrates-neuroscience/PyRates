
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
""" Tests that go all the way from YAML-based templates to the backend representation and potentially computations."""

__author__ = "Daniel Rose"
__status__ = "Development"

import pytest


def setup_module():
    print("\n")
    print("========================================")
    print("| Test Suite: From Frontend to Backend |")
    print("========================================")


@pytest.mark.xfail
def test_simple_example():
    """Test of a simple self-connecting one-node backend with a linear operator for the full pipeline from YAML
    to simulation."""

    # Step 1: Load Circuit template
    from pyrates.frontend.template.circuit import CircuitTemplate
    path = "../model_templates/test_resources/linear/ExampleCircuit"
    tmp = CircuitTemplate.from_yaml(path)

    # Step 2: Instantiate template to create frontend IR
    circuit = tmp.apply()

    # Step 3: Reformat frontend IR to backend IR
    # ToDo: adapt this step to new frontend-ir-backend structure
    from pyrates.frontend.nxgraph import from_circuit
    net_def = from_circuit(circuit, revert_node_names=True)

    # Step 4: Create tensorflow graph
    from pyrates.backend import ComputeGraph
    net = ComputeGraph(net_def, dt=5e-4, vectorization='none')

    # Step 5: Run simulation
    results, _ = net.run(simulation_time=1.,
                         outputs={"V": ('Node.0', 'LinearOperator.0', 'y')},
                         sampling_step_size=1e-3)

    results.plot()


@pytest.mark.xfail
@pytest.mark.parametrize("vectorize", ["none", "nodes", "ops"])
def test_3_coupled_jansen_rit_circuits(vectorize):
    """Test the simple Jansen-Rit example with three coupled circuits."""

    # Step 1: Load Circuit template
    from pyrates.frontend.template.circuit import CircuitTemplate
    path = "model_templates.jansen_rit.circuit.MultiJansenRitCircuit"
    tmp = CircuitTemplate.from_yaml(path)

    # Step 2: Instantiate template to create frontend IR
    circuit = tmp.apply()

    # Step 3: Reformat frontend IR to backend IR
    # ToDo: adapt this step to new frontend-ir-backend structure
    from pyrates.frontend.nxgraph import from_circuit
    net_def = from_circuit(circuit, revert_node_names=True)

    # Step 4: Create tensorflow graph
    from pyrates.backend import ComputeGraph
    net = ComputeGraph(net_def, dt=5e-4, vectorization=vectorize)
