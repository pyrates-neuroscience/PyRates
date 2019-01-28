
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
"""Function to construct circuits or other types of instances from a file or dictionary
"""

# external packages
import json
from typing import Union

# pyrates internal imports
from pyrates.axon import Axon
from pyrates.circuit import Circuit
from pyrates.population import Population
from pyrates.synapse import Synapse
# from pyrates.utility.json_filestorage import RepresentationBase

# meta infos
__author__ = "Daniel Rose"
__status__ = "Development"

# TODO: Update docstrings


def construct_circuit_from_file(filename: str, path: str="") -> Circuit:
    """Load a JSON file and construct a circuit from it"""

    import os

    filepath = os.path.join(path, filename)

    if not filepath.lower().endswith(".json"):
        filepath = f"{filepath}.json"

    with open(filepath, "r") as json_file:
        config_dict = json.load(json_file)

    circuit = construct_instance_from_dict(config_dict)

    return circuit


def construct_instance_from_dict(config_dict: dict, include_defaults=False) -> Union[Population, Axon, Synapse, Circuit]:
    """Construct a class instance from a dictionary that describes class name and input"""

    from importlib import import_module
    import numpy as np

    cls_dict = config_dict.pop("class")

    module = import_module(cls_dict["__module__"])
    cls = getattr(module, cls_dict["__name__"])

    if "network_graph" in config_dict:
        raise NotImplementedError("Haven't implemented the construction of a circuit from a graph yet.")

    # parse items
    #############

    # obtain positional arguments
    args = ()
    if "args" in config_dict:
        args = config_dict.pop("args", None)
        for i, arg in enumerate(args):
            if isinstance(arg, list):
                args[i] = np.asarray(arg)
                # this maps every list to a numpy array... might not always be the desired behavior

    # obtain default values
    defaults = {}
    if "defaults" in config_dict:
        if include_defaults:
            defaults = config_dict.pop("defaults")
        else:
            # ignore defaults in construction
            config_dict.pop("defaults")

    # obtain keyword arguments
    for key, item in config_dict.items():
        if isinstance(item, list):
            config_dict[key] = np.array(item)

    return cls(*args, **config_dict, **defaults)
