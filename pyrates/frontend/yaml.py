
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
from typing import Type

__author__ = "Daniel Rose"
__status__ = "Development"


def to_template_dict(path: str):
    """Load a template from YAML and return the resulting dictionary.

    Parameters
    ----------

    path
        string containing path of YAML template of the form path.to.template or path/to/template.file.TemplateName.
        The dot notation refers to a path that can be found using python's import functionality. The slash notation
        refers to a file in an absolute or relative path from the current working directory.
    """
    from pyrates.frontend.file import parse_path

    name, filename, directory = parse_path(path)
    from ruamel.yaml import YAML
    import os

    yaml = YAML(typ="safe", pure=True)

    if not filename.endswith(".yaml"):
        filename = f"{filename}.yaml"

    filepath = os.path.join(directory, filename)

    with open(filepath, "r") as file:
        file_dict = yaml.load(file)

    if name in file_dict:
        template_dict = file_dict[name]
        template_dict["path"] = path
        template_dict["name"] = name
    else:
        raise AttributeError(f"Could not find {name} in {filepath}.")

    return template_dict


class TemplateLoader:
    """Class that loads templates from YAML and returns an OperatorTemplate class instance"""

    cache = {}  # dictionary that keeps track of already loaded templates

    def __new__(cls, path: str, template_cls: type):
        """Load template recursively and return OperatorTemplate class.

        Parameters
        ----------

        path
            string containing path of YAML template of the form path.to.template
        template_cls
            class that the loaded template will be instantiated with
        """

        if path in cls.cache:
            template = cls.cache[path]
        else:
            template_dict = to_template_dict(path)
            try:
                base_path = template_dict.pop("base")
            except KeyError:
                raise KeyError(f"No 'base' defined for template {path}. Please define a "
                               f"base to derive the template from.")
            if base_path == template_cls.__name__:
                # if base refers to the python representation, instantiate here
                template = template_cls(**template_dict)
            else:
                # load base if needed
                if "." in base_path:
                    # reference to template in different file
                    # noinspection PyCallingNonCallable
                    template = cls(base_path)
                else:
                    # reference to template in same file
                    base_path = ".".join((*path.split(".")[:-1], base_path))
                    # noinspection PyCallingNonCallable
                    template = cls(base_path)
                template = cls.update_template(template, **template_dict)
                # may fail if "base" is present but empty

            cls.cache[path] = template

        return template

    @classmethod
    def update_template(cls, *args, **kwargs):
        """Updates the template with a given list of arguments."""
        raise NotImplementedError


def from_circuit(circuit, path: str, name: str):

    from pyrates.frontend.dict import from_circuit
    dict_repr = {name: from_circuit(circuit)}

    from ruamel.yaml import YAML
    yaml = YAML()

    from pyrates.utility.filestorage import create_directory
    create_directory(path)
    from pathlib import Path
    path = Path(path)
    yaml.dump(dict_repr, path)


def to_template(path: str, template_cls):
    return template_cls.from_yaml(path)
