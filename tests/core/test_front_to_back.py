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
    """Test of a simple self-connecting one-node backend with a linear operator for the full pipeline from YAML
    to simulation."""

    # Step 1: Load Circuit template
    from pyrates.frontend.circuit import CircuitTemplate
    path = "pyrates.examples.linear.ExampleCircuit"
    tmp = CircuitTemplate.from_yaml(path)

    # Step 2: Instantiate template to create frontend IR
    circuit = tmp.apply()

    # Step 3: Reformat frontend IR to backend IR
    net_def = circuit.network_def(revert_node_names=True)

    # Step 4: Create tensorflow graph
    from pyrates.backend import ComputeGraph
    net = ComputeGraph(net_def, dt=5e-4, vectorize='none')

    # Step 5: Run simulation
    results, _ = net.run(simulation_time=1.,
                         outputs={"V": ('Node.0', 'LinearOperator.0', 'y')},
                         sampling_step_size=1e-3)

    results.plot()


@pytest.mark.parametrize("vectorize", ["none", "nodes", "ops"])
def test_3_coupled_jansen_rit_circuits(vectorize):
    """Test the simple Jansen-Rit example with three coupled circuits."""

    # Step 1: Load Circuit template
    from pyrates.frontend.circuit import CircuitTemplate
    path = "pyrates.frontend.circuit.templates.MultiJansenRitCircuit"
    tmp = CircuitTemplate.from_yaml(path)

    # Step 2: Instantiate template to create frontend IR
    circuit = tmp.apply()

    # Step 3: Reformat frontend IR to backend IR
    net_def = circuit.network_def(revert_node_names=True)

    # Step 4: Create tensorflow graph
    from pyrates.backend import ComputeGraph
    net = ComputeGraph(net_def, dt=5e-4, vectorize=vectorize)
