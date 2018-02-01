""" Testing utility functions
"""

from pytest import xfail

__author__ = "Daniel Rose"
__status__ = "Development"


@xfail
def test_store_circuit_config_as_dict():
    """As title says."""

    from core.circuit import JansenRitCircuit
    from core.utility import read_config_from_circuit
    import json

    step_size = 1e-4

    circuit = JansenRitCircuit(step_size)

    config_dict = read_config_from_circuit(circuit)

    with open("../resources/jr_config_target.json", "r") as json_file:
        target_config_dict = json.load(json_file)

    assert config_dict == target_config_dict


