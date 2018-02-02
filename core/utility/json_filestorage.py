""" Utility functions to store Circuit configurations and data in JSON files
and read/construct circuit from JSON.
"""

__author__ = "Daniel Rose"
__status__ = "Development"

# from typing import Union, List
from inspect import getsource
import numpy as np
import json
from networkx import MultiDiGraph, node_link_data

# from core.circuit import Circuit
from core.population import Population
from core.axon import Axon
from core.synapse import Synapse


def get_attrs(obj: object) -> dict:
    """Transform meta-data of a given object into a dictionary, ignoring default fields form class <object>."""
    config_dict = dict()
    for key, item in obj.__dict__.items():
        if key not in object.__dict__:
            config_dict[key] = item
    return config_dict


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Population):
            return get_attrs(obj)
        elif isinstance(obj, Synapse):
            attr_dict = get_attrs(obj)
            # remove synaptic_kernel from dict, if kernel_function is specified
            if "kernel_function" in attr_dict:
                attr_dict.pop("synaptic_kernel", None)
            return attr_dict
        elif isinstance(obj, Axon):
            return get_attrs(obj)
        elif isinstance(obj, MultiDiGraph):
            net_dict = node_link_data(obj)
            for population in net_dict["nodes"]:
                population.pop("data", None)
            return net_dict
        elif callable(obj):
            return getsource(obj)
        else:
            return super(CustomEncoder, self).default(obj)


# def read_config_from_axon(axon: Axon) -> dict:
#     """Transform meta-data form an axon into a dictionary."""
#
#     config_dict = dict()
#
#     for key, item in axon.__dict__.items():
#         if key not in object.__dict__:
#             # if key == "transfer_function":
#             #     item = getsource(item)
#             config_dict[key] = item
#
#     return config_dict

    # config_dict["transfer_function"] = getsource(axon.transfer_function)
    #
    # config_dict["axon_type"] = axon.axon_type
    # config_dict["transfer_function_args"] = axon.transfer_function_args


# def read_config_from_population(population: Union[List[Population], Population]) -> List[Population]:
#     """Transform meta-data from a population or list of populations into a list of dictionaries"""
#
#     if isinstance(population, list):
#         pop_config_list = []
#         for pop in population:
#             config_list = read_config_from_population(pop)
#             pop_config_list.extend(config_list)
#
#     else:
#         config_dict = dict()
#         # skip synapses to prevent infinitely deep config file
#         # config_dict["synapses"] = read_config...
#
#         for key, item in population.__dict__.items():
#             if key not in object.__dict__:
#                 # if key == "synapses":
#                 #     item = read_config_from_synapse(item)
#
#                 # if key == "axon":
#                 #     item = read_config_from_axon(item)
#
#                 config_dict[key] = item
#         #
#         #
#         # config_dict["state_variables"] = population.state_variables
#         # config_dict["current_firing_rate"] = population.current_firing_rate
#         # config_dict["synaptic_input"] = population.synaptic_input
#         # config_dict["current_input_idx"] = population.current_input_idx
#         # config_dict["additive_synapse_idx"] = population.additive_synapse_idx
#         # config_dict["modulatory_synapse_idx"] = population.modulatory_synapse_idx
#         # config_dict["synaptic_currents"] = population.synaptic_currents
#         # config_dict["synaptic_modulation"] = population.synaptic_modulation
#         # config_dict["synaptic_modulation_direction"] = population.synaptic_modulation_direction
#         # config_dict["extrinsic_current"] = population.extrinsic_current
#         # config_dict["extrinsic_synaptic_modulation"] = population.extrinsic_synaptic_modulation
#         # config_dict["n_synapses"] = population.n_synapses
#         # config_dict["kernel_lengths"] = population.kernel_lengths
#         # config_dict["max_population_delay"] = population.max_population_delay
#         # config_dict["max_synaptic_delay"] = population.max_synaptic_delay
#         # config_dict["store_state_variables"] = population.store_state_variables
#         # config_dict["tau_leak"] = population.tau_leak
#         # config_dict["step_size"] = population.step_size
#         # config_dict["membrane_capacitance"] = population.membrane_capacitance
#         # config_dict["label"] = population.label
#
#         pop_config_list = [config_dict]
#
#     return pop_config_list





# def read_config_from_synapse(synapse: Union[List[Synapse], Synapse]) -> List[dict]:
#     """Transform meta-data from a synapse or list of synapses into a list of dictionaries"""
#     if isinstance(synapse, list):
#         syn_config_list = []
#         for syn in synapse:
#             config_list = read_config_from_synapse(syn)
#             syn_config_list.extend(config_list)
#
#     else:
#         config_dict = dict()
#         # skip synapses to prevent infinitely deep config file
#         # config_dict["synapses"] = read_config...
#
#         for key, item in synapse.__dict__.items():
#             if key not in object.__dict__:
#                 config_dict[key] = item
#
#         syn_config_list = [config_dict]
#
#     return syn_config_list


# def read_config_from_circuit(circuit: Circuit) -> dict:
#     """Transform all relevant info from a Circuit instance into a dictionary."""
#
#     import networkx
#
#     config_dict = dict()
#
#     for key, item in circuit.__dict__.items():
#         if key not in object.__dict__:
#             # if key == "populations":
#             #     item = read_config_from_population(item)
#
#             if key == "network_graph":
#                 item = networkx.node_link_data(item)
#                 # for node in item["nodes"]:
#                 #     if not isinstance(node["data"], Population):
#                 #         raise TypeError("Expected type <Population> or a subclass of it.")
#                 #     node["data"] = read_config_from_population(node["data"])
#                     # assuming a node's data is of type Population, this fails if the convention is changed.
#
#             config_dict[key] = item
#
#     # config_dict["populations"] = read_config_from_population(circuit.populations)
#     #
#     # config_dict["C"] = circuit.C
#     # config_dict["D"] = circuit.D
#     # config_dict["step_size"] = circuit.step_size
#     # config_dict["N"] = circuit.N
#     # config_dict["n_synapses"] = circuit.n_synapses
#     #
#     # config_dict["active_synapses"] = read_config_from_synapses(circuit.active_synapses)
#     # config_dict["network_graph"] = networkx.node_link_data(circuit.network_graph)
#
#     return config_dict

