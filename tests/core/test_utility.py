""" Testing utility functions
"""

import pytest
# import numpy as np

from core.utility.json_filestorage import CustomEncoder, get_attrs

__author__ = "Daniel Rose"
__status__ = "Development"

#########
# Setup #
#########


def setup_module():
    print("\n")
    print("====================================")
    print("| Test Suite 5 : Utility Functions |")
    print("====================================")

#########
# Tests #
#########


# @pytest.mark.xfail
def test_store_circuit_config_as_dict():
    """As title says."""

    from core.circuit import JansenRitCircuit
    import json

    step_size = 1e-4

    circuit = JansenRitCircuit(step_size)

    config_dict = get_attrs(circuit)

    with open("tests/resources/jr_config_target.json", "w") as json_file:
        json.dump(config_dict, json_file, cls=CustomEncoder, indent=4)

    config_dict = json.loads(json.dumps(config_dict, cls=CustomEncoder))

    # TODO: need to properly filter all elements in dict - or simply directly pass to file

    with open("tests/resources/jr_config_target.json", "r") as json_file:
        target_config_dict = json.load(json_file)

    assert config_dict == target_config_dict


@pytest.mark.xfail
def test_store_circuit_config_dict_as_json():
    """As title says."""

    from core.circuit import JansenRitCircuit
    from core.utility import save_circuit_config_to_disk

    step_size = 1e-4

    circuit = JansenRitCircuit(step_size)

    relpath = "../resources/"
    filename = "jr_config_test_result.json"
    save_circuit_config_to_disk(circuit, relpath=relpath, filename=filename)

    # with open("../resources/jr_config_test_result.json", "w") as json_file:
    #     json.dump(config_dict, json_file)

    import filecmp
    from os import path

    result = path.join(relpath, filename)
    target = "../resources/jr_config_target.json"

    assert filecmp.cmp(result, target)

    filecmp.clear_cache()


@pytest.mark.xfail
def test_read_circuit_config_from_file():
    """As title says."""

    from core.circuit import JansenRitCircuit
    from core.utility import read_config_from_file, read_config_from_circuit

    step_size = 1e-4

    target_circuit = JansenRitCircuit(step_size)

    target_config_dict = read_config_from_circuit(target_circuit)

    test_dict = read_config_from_file("../resources/jr_config_target.json")

    assert test_dict == target_config_dict


@pytest.mark.xfail
def test_construct_circuit_from_file_or_dict():
    """As title says."""

    from core.circuit import JansenRitCircuit
    from core.utility import read_config_from_file, construct_circuit_from_dict, read_config_from_circuit, \
        construct_circuit_from_file

    step_size = 1e-4

    target_circuit = JansenRitCircuit(step_size)

    target_config_dict = read_config_from_circuit(target_circuit)

    # Construct circuit from template's dict and compare against template
    #####################################################################

    test_circuit = construct_circuit_from_dict(target_config_dict)

    test_config_dict = read_config_from_circuit(test_circuit)

    assert test_config_dict == target_config_dict

    # Construct from file and test against template
    ###############################################

    test_circuit = construct_circuit_from_file("../resources/jr_config_target.json")

    test_config_dict = read_config_from_circuit(test_circuit)

    assert test_config_dict == target_config_dict

    # test if the resulting circuit actually runs
    #############################################

    synaptic_inputs = test_circuit.stored_synaptic_inputs
    simulation_time = test_circuit.stored_simulation_time
    test_circuit.run(synaptic_inputs, simulation_time)



