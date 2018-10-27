""" Tests that go all the way from YAML-based templates to the backend representation and potentially computations."""

__author__ = "Daniel Rose"
__status__ = "Development"

import pytest


def setup_module():
    print("\n")
    print("========================================")
    print("| Test Suite: From Frontend to Backend |")
    print("========================================")


def test_simple_example():
    """Test of a simple self-connecting one-node network with a linear operator for the full pipeline from YAML
    to simulation."""

    # Step 1: Load Circuit template
    from pyrates.frontend.circuit import CircuitTemplate
    path = "pyrates.examples.linear.ExampleCircuit"
    tmp = CircuitTemplate.from_yaml(path)

    # Step 2: Instantiate template to create frontend IR
    circuit = tmp.apply()

    # Step 3: Reformat frontend IR to backend IR
    net_def = circuit.network_def()

    # Step 4: Create tensorflow graph
    from pyrates.network import Network
    net = Network(net_def, dt=5e-4, vectorize='none')

    # Step 5: Run simulation
    results, _ = net.run(simulation_time=1.,
                         outputs={"V": ('all', 'operator_ptr' , 'v')},
                         sampling_step_size=1e-3)

    results.plot()