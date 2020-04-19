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
def test_pickle_ir():
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
    pickle.dump(circuit, open("resources/jansen_rit.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)


@pytest.mark.skip
def test_load_from_pickle():
    pass


def test_pickle_template():
    path = "model_templates.jansen_rit.circuit.JansenRitCircuit"
    from pyrates.frontend.template import from_yaml, clear_cache
    import filecmp
    import os
    clear_cache()
    template = from_yaml(path)
    out_file = "output/jansen_rit_template.p"
    test_file = "resources/jansen_rit_template.p"

    from pyrates.frontend.fileio import pickle

    # pickle.dump(template, open("resources/jansen_rit_template.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(template, out_file)

    if os.path.getsize(out_file) == os.path.getsize(test_file):
        assert filecmp.cmp(out_file, test_file, shallow=False)
    else:
        raise ValueError("File are not the same")

    data = pickle.load(out_file)
    assert data
