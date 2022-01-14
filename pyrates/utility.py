# -*- coding: utf-8 -*-
#
#
# PyRates software framework for flexible implementation of neural
# network model_templates and simulations. See also:
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
"""Functions for performing parameter grid simulations with pyrates model_templates.
"""

# meta infos
__author__ = "Christoph Salomon, Richard Gast"
__status__ = "development"

# external imports
from typing import Union, Optional, Tuple, Dict
import pandas as pd
import numpy as np
from copy import deepcopy
import json
from inspect import getsource
#from networkx import node_link_data
from pandas import DataFrame
import pickle

# pyrates imports
from pyrates.frontend import CircuitTemplate, NodeTemplate, OperatorTemplate, EdgeTemplate, template


###########################
# model loading functions #
###########################


def clear_frontend_caches(clear_template_cache=True, clear_operator_cache=True):
    """Utility to clear caches in the frontend.

    Parameters
    ----------
    clear_template_cache
        toggles whether or not to clear the template_cache that contains all previously loaded templates
    clear_operator_cache
        toggles whether or not to clear the cache of unique OperatorIR instances
    """
    if clear_template_cache:
        template.clear_cache()

    if clear_operator_cache:
        OperatorTemplate.cache.clear()


# The following function are shorthands that bridge multiple interface steps
def circuit_from_yaml(path: str):
    """Directly return CircuitIR instance from a yaml file."""
    return CircuitTemplate.from_yaml(path)


# The following function are shorthands that bridge multiple interface steps
def circuit_from_pickle(path: str, **kwargs):
    """Directly return CircuitIR instance from a yaml file."""
    return pickle.load(path, **kwargs)


def node_from_yaml(path: str):
    """Directly return NodeIR instance from a yaml file."""
    return NodeTemplate.from_yaml(path).apply()


def edge_from_yaml(path: str):
    """Directly return EdgeIR instance from a yaml file."""

    return EdgeTemplate.from_yaml(path).apply()


def operator_from_yaml(path: str):
    """Directly return OperatorIR instance from a yaml file."""

    return OperatorTemplate.from_yaml(path).apply()


########################
# simulation functions #
########################


def integrate(circuit: Union[str, CircuitTemplate], **kwargs):
    """Directly simulate dynamics of a circuit."""
    if type(circuit) is str:
        circuit = circuit_from_yaml(path=circuit)
    results = circuit.run(**kwargs)
    if 'clear' in kwargs and kwargs['clear']:
        clear_frontend_caches()
    return results


#############################
# parameter sweep functions #
#############################

def linearize_grid(grid: dict, permute: bool = False) -> pd.DataFrame:
    """Turns the grid into a grid that can be traversed linearly, i.e. pairwise.

    Parameters
    ----------
    grid
        Parameter grid.
    permute
        If true, all combinations of the parameter values in grid will be created.

    Returns
    -------
    pd.DataFrame
        Resulting linear grid in form of a data frame.

    """

    arg_lengths = [len(arg) for arg in grid.values()]

    if len(list(set(arg_lengths))) == 1 and not permute:
        return pd.DataFrame(grid)
    elif permute:
        vals, keys = [], []
        for key, val in grid.items():
            vals.append(val)
            keys.append(key)
        new_grid = np.stack(np.meshgrid(*tuple(vals)), -1).reshape(-1, len(grid))
        return pd.DataFrame(new_grid, columns=keys)
    else:
        raise ValueError('Wrong number of parameter combinations. If `permute` is False, all parameter vectors in grid '
                         'must have the same number of elements.')


def adapt_circuit(circuit: Union[CircuitTemplate, str], params: dict, param_map: dict) -> CircuitTemplate:
    """Changes the parametrization of a circuit.

    Parameters
    ----------
    circuit
        Circuit instance.
    params
        Key-value pairs of the parameters that should be changed.
    param_map
        Map between the keys in params and the circuit variables.

    Returns
    -------
    CircuitIR
        Updated circuit instance.

    """

    if type(circuit) is str:
        circuit = deepcopy(CircuitTemplate.from_yaml(circuit))
    else:
        circuit = deepcopy(circuit)

    node_updates = {}
    edge_updates = []

    for key in params.keys():

        val = params[key]

        if 'nodes' in param_map[key]:
            for n in param_map[key]['nodes']:
                for v in param_map[key]['vars']:
                    node_updates[f"{n}/{v}"] = val
        else:
            edges = param_map[key]['edges']
            if len(edges[0]) < 3:
                for source, target in edges:
                    for var in param_map[key]['vars']:
                        edge = circuit.get_edge(source=source, target=target, idx=0)
                        edge_updates.append((edge[0], edge[1], {var: val}))
            else:
                for source, target, idx in edges:
                    for var in param_map[key]['vars']:
                        edge = circuit.get_edge(source=source, target=target, idx=idx)
                        edge_updates.append((edge[0], edge[1], {var: val}))

    return circuit.update_var(node_vars=node_updates, edge_vars=edge_updates)


def grid_search(circuit_template: Union[CircuitTemplate, str], param_grid: Union[dict, pd.DataFrame], param_map: dict,
                step_size: float, simulation_time: float, inputs: dict, outputs: dict,
                sampling_step_size: Optional[float] = None, permute_grid: bool = False, **kwargs) -> tuple:
    """Function that runs multiple parametrizations of the same circuit in parallel and returns a combined output.

    Parameters
    ----------
    circuit_template
        Path to the circuit template.
    param_grid
        Key-value pairs for each circuit parameter that should be altered over different circuit parametrizations.
    param_map
        Key-value pairs that map the keys of param_grid to concrete circuit variables.
    step_size
        Simulation step-size in s.
    simulation_time
        Simulation time in s.
    inputs
        Inputs as provided to the `run` method of `:class:ComputeGraph`.
    outputs
        Outputs as provided to the `run` method of `:class:ComputeGraph`.
    sampling_step_size
        Sampling step-size as provided to the `run` method of `:class:ComputeGraph`.
    permute_grid
        If true, all combinations of the provided param_grid values will be realized. If false, the param_grid values
        will be traversed pairwise.
    kwargs
        Additional keyword arguments passed to the `CircuitTemplate.run` call.


    Returns
    -------
    tuple
        Simulation results stored in a multi-index data frame, the mapping between the data frame column names and the
        parameter grid, the simulation time, and the memory consumption.

    """

    # argument pre-processing
    #########################

    vectorization = kwargs.pop('vectorization', True)

    # linearize parameter grid if necessary
    if type(param_grid) is dict:
        param_grid = linearize_grid(param_grid, permute_grid)

    # create grid-structure of network
    ##################################

    # get parameter names and grid length
    param_keys = list(param_grid.keys())
    N = param_grid.shape[0]

    # assign parameter updates to each circuit, combine them to unconnected network and remember their parameters
    circuit_names = []
    circuit = CircuitTemplate(name='top_lvl', path='none')
    for idx in param_grid.index:
        new_params = {}
        for key in param_keys:
            new_params[key] = param_grid[key][idx]
        circuit_tmp = adapt_circuit(circuit_template, new_params, param_map)
        circuit_key = f'{circuit_tmp.name}_{idx}'
        circuit = circuit.update_template(circuits={circuit_key: circuit_tmp})
        circuit_names.append(circuit_key)
    param_grid.index = circuit_names

    # adjust input of simulation to combined network
    for inp_key, inp in inputs.copy().items():
        inputs[f"all/{inp_key}"] = inp
        inputs.pop(inp_key)

    # adjust output of simulation to combined network
    outputs_new = {}
    for key, out in outputs.items():
        outputs_new[key] = f"all/{out}"

    # simulate the circuits behavior
    results = circuit.run(simulation_time=simulation_time,
                          step_size=step_size,
                          sampling_step_size=sampling_step_size,
                          inputs=inputs,
                          outputs=outputs_new,
                          vectorization=vectorization,
                          **kwargs)    # type: pd.DataFrame

    # create dataframe that maps between output names and parameter sets
    data, index = [], []
    for key in results.keys():
        param_key = key[1].split('/')[0]
        data.append(param_grid.loc[param_key, :].values)
    param_map = pd.DataFrame(data=np.asarray(data).T, columns=results.columns, index=param_grid.columns)

    # return results
    if 'profile' in kwargs:
        results, duration = results
        return results, param_map, duration
    return results, param_map


##########################
# file storage functions #
##########################

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):

        # from pyrates.population import Population
        # from pyrates.synapse import Synapse

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, "to_json"):
            return obj.to_dict()
        # elif isinstance(obj, Synapse):
        #     attr_dict = obj.to_dict()
        #     # remove synaptic_kernel from dict, if kernel_function is specified
        #     if "kernel_function" in attr_dict:
        #       JansenRitCircuit  attr_dict.pop("synaptic_kernel", None)
        #     return attr_dict
        # elif isinstance(obj, Axon):
        #     return get_attrs(obj)
        elif callable(obj):
            return getsource(obj)
        else:
            return super().default(obj)


# class RepresentationBase(object):
#     """Class that implements a __repr__ that yields the __init__ function signature with provided arguments in the
#     form 'module.Class(arg1, arg2, arg3)'"""
#
#     def __new__(cls, *args, **kwargs):
#
#         # noinspection PyArgumentList
#         instance = super().__new__(cls)
#         _init_dict = {**kwargs}
#         if args:
#             _init_dict["args"] = args
#
#         instance._init_dict = _init_dict
#
#         return instance
#
#     def __repr__(self) -> str:
#         """Magic method that returns to repr(object). It retrieves the signature of __init__ and maps it to class
#         attributes to return a representation in the form 'module.Class(arg1=x, arg2=y, arg3=3)'. Raises AttributeError,
#         if the a parameter in the signature cannot be found (is not saved). The current implementation """
#
#         # repr_str = f"{self.__module__!r}.{self.__class__!r}("
#         # params = self._get_params()
#         from copy import copy
#         init_dict = copy(self._init_dict)
#
#         # args = ""
#         # if "args" in init_dict:
#         #     args = init_dict.pop("args", None)
#         #     args = ", ".join((f"{value!r}" for value in args))
#         #
#         # kwargs = ", ".join((f"{name}={value!r}" for name, value in init_dict.items()))
#         # param_str = f"{args}, {kwargs}"
#         param_str = "*args, **kwargs"
#         _module = self.__class__.__module__
#         _class = self.__class__.__name__
#         # return f"{_module}.{_class}({param_str})"
#
#         try:
#             return f"{_class} '{self.key}'"
#         except AttributeError:
#             return f"{_class}"
#
#     def _defaults(self) -> Generator[Tuple[str, Any], None, None]:
#         """Inspects the __init__ special method and yields tuples of init parameters and their default values."""
#
#         import inspect
#
#         # retrieve signature of __init__ method
#         sig = inspect.signature(self.__init__)
#
#         # retrieve parameters in the signature
#         for name, param in sig.parameters.items():
#             # check if param has a default value
#             if np.all(param.default != inspect.Parameter.empty):
#                 # yield parameter name and default value
#                 yield name, param.default
#
#     def to_dict(self, include_defaults=False, include_graph=False, recursive=False) -> OrderedDict:
#         """Parse representation string to a dictionary to later convert it to json."""
#
#         _dict = dict()
#         _dict["class"] = {"__module__": self.__class__.__module__, "__name__": self.__class__.__name__}
#         # _dict["module"] = self.__class__.__module__
#         # _dict["class"] = self.__class__.__name__
#
#         # obtain parameters that were originally passed to __init__
#         ###########################################################
#
#         for name, value in self._init_dict.items():
#             _dict[name] = value
#
#         # include defaults of parameters that were not specified when __init__ was called.
#         ##################################################################################
#
#         if include_defaults:
#             default_dict = {}
#             for name, value in self._defaults():
#                 if name not in _dict:  # this fails to get positional arguments that override defaults to keyword args
#                     default_dict[name] = value
#             _dict["defaults"] = default_dict
#
#         # Include graph if desired
#         ##########################
#
#         # check if it quacks like a duck... eh.. has a network_graph like a circuit
#         if include_graph and hasattr(self, "network_graph"):
#             net_dict = node_link_data(self.network_graph)
#             if recursive:
#                 for node in net_dict["nodes"]:
#                     node["data"] = node["data"].to_dict(include_defaults=include_defaults, recursive=True)
#             _dict["network_graph"] = net_dict
#
#         # Apply dict transformation recursively to all relevant objects
#         ###############################################################
#
#         if recursive:
#             for key, item in _dict.items():
#                 # check if it quacks like a duck... eh.. represents like a RepresentationBase
#                 # note, however, that "to_dict" is actually implemented by a number of objects outside this codebase
#                 if hasattr(item, "to_dict"):
#                     _dict[key] = item.to_dict(include_defaults=include_defaults, recursive=True)
#
#         return _dict
#
#     def to_json(self, include_defaults=False, include_graph=False, path="", filename=""):
#         """Parse a dictionary into """
#
#         # from pyrates.utility.json_filestorage import CustomEncoder
#
#         _dict = self.to_dict(include_defaults=include_defaults, include_graph=include_graph, recursive=True)
#
#         if filename:
#             import os
#             filepath = os.path.join(path, filename)
#
#             # create directory if necessary
#             create_directory(filepath)
#
#             if not filepath.lower().endswith(".json"):
#                 filepath = f"{filepath}.json"
#
#             with open(filepath, "w") as outfile:
#                 json.dump(_dict, outfile, cls=CustomEncoder, indent=4)
#
#         # return json as string
#         return json.dumps(_dict, cls=CustomEncoder, indent=2)


def get_simulation_data(circuit, state_variable='membrane_potential', pop_keys: Union[tuple, list] = None,
                        time_window: tuple = None) -> Tuple[dict, DataFrame]:
    """Obtain all simulation data from a circuit, including run parameters"""

    run_info = circuit.run_info
    states = circuit.get_population_states(state_variable=state_variable, population_keys=pop_keys,
                                           time_window=time_window)

    labels = [pop for pop in circuit.populations.keys() if 'dummy' not in pop]

    states = DataFrame(data=states, index=run_info.index, columns=labels)

    return run_info, states


def save_simulation_data_to_file(output_data: DataFrame, run_info: dict,
                                 dirname: str, path: str = "", out_format: str = "csv"):
    """Save simulation output and inputs that were given to the run function to a file."""

    import os

    dirname = os.path.join(path, dirname) + "/"

    # create directory if necessary
    create_directory(dirname)

    # save output data
    ##################

    filename = f"output.{out_format}"
    filepath = os.path.join(dirname, filename)
    if out_format == "json":
        output_data.to_json(filepath, orient="split")
    elif out_format == "csv":
        output_data.to_csv(filepath, sep="\t")
    else:
        raise ValueError(f"Unknown output format '{out_format}'")

    # save run information
    ######################

    if "time_vector" in run_info:
        # throw away time vector, since it is contained in all the other data
        run_info.pop("time_vector")

    for key, item in run_info.items():
        filename = f"{key}.{out_format}"
        filepath = os.path.join(dirname, filename)
        if item is None:
            continue

        if out_format == "json":
            item.to_json(filepath, orient="split")
        else:
            item.to_csv(filepath, sep="\t")

    # TODO: think about float vs. decimal representation: Possibly save float data as hex or fraction
    # TODO: choose time update as exact float (e.g. 0.0005 --> 2**-11=0.00048828125)


def create_directory(path):
    """check if a directory exists and create it otherwise"""

    import os
    import errno

    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def read_simulation_data_from_file(dirname: str, path="", filenames: list = None) -> Dict[str, DataFrame]:
    """Read simulation data from files. The assumed data structure is:
    <path>/<dirname>/
        output.csv
        synaptic_inputs.csv
        extrinsic_current.csv
        extrinsic_modulation.csv

    This is the expected output from 'save_simulation_data_to_file', but if 'names' is specified, other data
    may also be read.
    """

    import os

    if filenames is None:
        filenames = ["output", "synaptic_inputs", "extrinsic_current", "extrinsic_modulation"]
        ignore_missing = True
    else:
        ignore_missing = False

    data = {}
    path = os.path.join(path, dirname)

    for label in filenames:
        if label == "output":
            header = 0
        else:
            header = [0, 1]
        filename = label + ".csv"
        filepath = os.path.join(path, filename)

        try:
            data[label] = pd.read_csv(filepath, sep="\t", header=header, index_col=0)
        except FileNotFoundError:
            if not ignore_missing:
                raise

    return data


def to_pickle(obj, filename):
    """Conserve a PyRates object as pickle."""
    pass
