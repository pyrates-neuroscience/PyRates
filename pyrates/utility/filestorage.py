
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
""" Utility functions to store Circuit configurations and data in JSON files
and read/construct circuit from JSON.
"""

# external packages
from collections import OrderedDict
from typing import Generator, Tuple, Any, Union, Dict
from networkx import node_link_data
from inspect import getsource
import numpy as np
import json
from pandas import DataFrame
import pandas as pd

# meta infos
__author__ = "Daniel Rose"
__status__ = "Development"


# TODO: Update documentations & clean functions from unnecessary comments (i.e. silent code)


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


class RepresentationBase(object):
    """Class that implements a __repr__ that yields the __init__ function signature with provided arguments in the
    form 'module.Class(arg1, arg2, arg3)'"""

    def __new__(cls, *args, **kwargs):

        # noinspection PyArgumentList
        instance = super().__new__(cls)
        _init_dict = {**kwargs}
        if args:
            _init_dict["args"] = args

        instance._init_dict = _init_dict

        return instance

    def __repr__(self) -> str:
        """Magic method that returns to repr(object). It retrieves the signature of __init__ and maps it to class
        attributes to return a representation in the form 'module.Class(arg1=x, arg2=y, arg3=3)'. Raises AttributeError,
        if the a parameter in the signature cannot be found (is not saved). The current implementation """

        # repr_str = f"{self.__module__!r}.{self.__class__!r}("
        # params = self._get_params()
        from copy import copy
        init_dict = copy(self._init_dict)

        # args = ""
        # if "args" in init_dict:
        #     args = init_dict.pop("args", None)
        #     args = ", ".join((f"{value!r}" for value in args))
        #
        # kwargs = ", ".join((f"{name}={value!r}" for name, value in init_dict.items()))
        # param_str = f"{args}, {kwargs}"
        param_str = "*args, **kwargs"
        _module = self.__class__.__module__
        _class = self.__class__.__name__
        # return f"{_module}.{_class}({param_str})"

        try:
            return f"{_class} '{self.key}'"
        except AttributeError:
            return f"{_class}"

    def _defaults(self) -> Generator[Tuple[str, Any], None, None]:
        """Inspects the __init__ special method and yields tuples of init parameters and their default values."""

        import inspect

        # retrieve signature of __init__ method
        sig = inspect.signature(self.__init__)

        # retrieve parameters in the signature
        for name, param in sig.parameters.items():
            # check if param has a default value
            if np.all(param.default != inspect.Parameter.empty):
                # yield parameter name and default value
                yield name, param.default

    def to_dict(self, include_defaults=False, include_graph=False, recursive=False) -> OrderedDict:
        """Parse representation string to a dictionary to later convert it to json."""

        _dict = OrderedDict()
        _dict["class"] = {"__module__": self.__class__.__module__, "__name__": self.__class__.__name__}
        # _dict["module"] = self.__class__.__module__
        # _dict["class"] = self.__class__.__name__

        # obtain parameters that were originally passed to __init__
        ###########################################################

        for name, value in self._init_dict.items():
            _dict[name] = value

        # include defaults of parameters that were not specified when __init__ was called.
        ##################################################################################

        if include_defaults:
            default_dict = {}
            for name, value in self._defaults():
                if name not in _dict:  # this fails to get positional arguments that override defaults to keyword args
                    default_dict[name] = value
            _dict["defaults"] = default_dict

        # Include graph if desired
        ##########################

        # check if it quacks like a duck... eh.. has a network_graph like a circuit
        if include_graph and hasattr(self, "network_graph"):
            net_dict = node_link_data(self.network_graph)
            if recursive:
                for node in net_dict["nodes"]:
                    node["data"] = node["data"].to_dict(include_defaults=include_defaults, recursive=True)
            _dict["network_graph"] = net_dict

        # Apply dict transformation recursively to all relevant objects
        ###############################################################

        if recursive:
            for key, item in _dict.items():
                # check if it quacks like a duck... eh.. represents like a RepresentationBase
                # note, however, that "to_dict" is actually implemented by a number of objects outside this codebase
                if hasattr(item, "to_dict"):
                    _dict[key] = item.to_dict(include_defaults=include_defaults, recursive=True)

        return _dict

    def to_json(self, include_defaults=False, include_graph=False, path="", filename=""):
        """Parse a dictionary into """

        # from pyrates.utility.json_filestorage import CustomEncoder

        _dict = self.to_dict(include_defaults=include_defaults, include_graph=include_graph, recursive=True)

        if filename:
            import os
            filepath = os.path.join(path, filename)

            # create directory if necessary
            create_directory(filepath)

            if not filepath.lower().endswith(".json"):
                filepath = f"{filepath}.json"

            with open(filepath, "w") as outfile:
                json.dump(_dict, outfile, cls=CustomEncoder, indent=4)

        # return json as string
        return json.dumps(_dict, cls=CustomEncoder, indent=2)


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


