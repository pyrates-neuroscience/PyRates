"""Test suit for saving PyRates objects to or loading them from file."""

__author__ = "Daniel Rose"
__status__ = "Development"

import pytest


def setup_module():
    print("\n")
    print("=======================")
    print("| Test Suite File I/O |")
    print("=======================")


@pytest.mark.xfail
def test_save_to_pickle():
    pass

    path = "model_templates.jansen_rit.circuit.JansenRitCircuit"
    from pyrates.frontend.template.circuit import CircuitTemplate
    from pyrates.ir.circuit import CircuitIR

    template = CircuitTemplate.from_yaml(path)

    circuit = template.apply()  # type: CircuitIR
    # circuit.to_file(filename="resources/jansen_rit.p", mode="pickle")
    circuit.optimize_graph_in_place()
    # circuit.to_file(filename="resources/jansen_rit_vectorized.p", mode="pickle")

    import pickle
    pickle.dump(circuit, open("resources/jansen_rit.p", "wb"))


@pytest.mark.skip
def test_load_from_pickle():
    pass