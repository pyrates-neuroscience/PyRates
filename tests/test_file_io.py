"""Test suit for saving PyRates objects to or loading them from file."""

__author__ = "Daniel Rose"
__status__ = "Development"

import filecmp
import os

import pytest


def setup_module():
    print("\n")
    print("=======================")
    print("| Test Suite File I/O |")
    print("=======================")


class FileCompareError(Exception):
    pass


def compare_files(filename1, filename2):
    """Helper function to compare files"""

    if os.path.getsize(filename1) == os.path.getsize(filename2):
        if filecmp.cmp(filename1, filename2, shallow=False):
            pass
        else:
            raise FileCompareError("Files have different content.")
    else:
        raise FileCompareError("Files are not of the same size.")


def test_pickle_ir():
    pass

    path = "model_templates.jansen_rit.circuit.JansenRitCircuit"
    from pyrates.frontend.template.circuit import CircuitTemplate
    from pyrates.ir.circuit import CircuitIR
    from pyrates.frontend.fileio import pickle

    template = CircuitTemplate.from_yaml(path)

    # try to pickle non-vectorized circuit
    circuit = template.apply()  # type: CircuitIR
    circuit.to_file(filename="output/jansen_rit_ir.p", mode="pickle")

    # compare to reference pickle
    # compare_files("output/jansen_rit_ir.p", "resources/jansen_rit_ir.p")  # currently does not work
    circuit2 = pickle.load("output/jansen_rit_ir.p")

    # ToDo: compare circuit IR instances at runtime

    # try to pickle vectorized circuit
    circuit.optimize_graph_in_place()
    circuit.to_file(filename="output/jansen_rit_ir_vectorized.p", mode="pickle")

    # compare to reference pickle
    # compare_files("output/jansen_rit_ir_vectorized.p", "resources/jansen_rit_ir_vectorized.p")
    # currently does not work

    circuit3 = pickle.load("output/jansen_rit_ir_vectorized.p")

    # ToDo: compare vectorized circuit IR instances at runtime


def test_pickle_template():
    path = "model_templates.jansen_rit.circuit.JansenRitCircuit"
    from pyrates.frontend.template import from_yaml, clear_cache
    clear_cache()
    template = from_yaml(path)
    out_file = "output/jansen_rit_template.p"
    test_file = "resources/jansen_rit_template.p"

    from pyrates.frontend.fileio import pickle

    # pickle.dump(template, open("resources/jansen_rit_template.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(template, out_file)

    compare_files(out_file, test_file)

    template = pickle.load(out_file)
    assert template
