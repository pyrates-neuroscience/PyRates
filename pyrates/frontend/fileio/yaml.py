# -*- coding: utf-8 -*-
#
#
# PyRates software framework for flexible implementation of neural 
# network models and simulations. See also: 
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
""" Some utility functions for parsing YAML-based definitions of circuits and components.
"""


__author__ = "Daniel Rose"
__status__ = "Development"


def dict_from_yaml(path: str):
    """Load a template from YAML and return the resulting dictionary.

    Parameters
    ----------

    path
        (str) path to YAML template of the form `path.to.template_file.template_name` or
        path/to/template_file/template_name.TemplateName. The dot notation refers to a path that can be found
        using python's import functionality. That means it needs to be a module (a folder containing an `__init__.py`)
        located in the Python path (e.g. the current working directory). The slash notation refers to a file in an
        absolute or relative path from the current working directory. In either case the second-to-last part refers to
        the filename without file extension and the last part refers to the template name.
    """
    from pyrates.frontend.file import parse_path

    template_name, filename, directory = parse_path(path)

    # test if file can be found (and potentially add extension)
    import os

    if "." in filename:
        filepath = os.path.join(directory, filename)
    else:
        # this is actually the default case for the internal interface
        for ext in ["yaml", "yml"]:
            filepath = os.path.join(directory, ".".join((filename, ext)))
            if os.path.exists(filepath):
                break
        else:
            raise FileNotFoundError(f"Could not identify file with name {filename} in directory {directory}.")

    # load as yaml file
    from ruamel.yaml import YAML

    yaml = YAML(typ="safe", pure=True)

    with open(filepath, "r") as file:
        file_dict = yaml.load(file)

    if template_name in file_dict:
        template_dict = file_dict[template_name]
        template_dict["path"] = path
        template_dict["name"] = template_name
    else:
        raise AttributeError(f"Could not find {template_name} in {filepath}.")

    return template_dict


def dump_to_yaml(circuit, path: str, **kwargs) -> None:
    """Interface to dump a `CircuitTemplate` instance to YAML.

    Parameters
    ----------
    circuit
    path

    Returns
    -------
    None
    """
    from pyrates.frontend.dict import from_circuit as dict_from_circuit

    dict_repr = {}
    dict_from_circuit(circuit, dict_repr)

    from ruamel.yaml import YAML
    yaml = YAML()

    from pyrates.utility import create_directory
    create_directory(path)
    from pathlib import Path
    path = Path(path)
    yaml.dump(dict_repr, path, **kwargs)
