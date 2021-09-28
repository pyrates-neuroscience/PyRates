
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
"""
"""
import importlib

from pyrates.ir.circuit import PyRatesException
from pyrates.frontend.fileio import yaml as _yaml

__author__ = "Daniel Rose"
__status__ = "Development"

file_loader_mapping = {"yaml": _yaml.dict_from_yaml,
                       "yml": _yaml.dict_from_yaml}


# def to_template(filepath: str, template_name: str):
#     """Draft for generic template interface. Currently not in use."""
#     name, file, abspath = parse_path(filepath)
#     filename, extension = file.split(".")
#     try:
#         loader = file_loader_mapping[extension]
#     except KeyError:
#         raise PyRatesException(f"Could not find loader for file extension {extension}.")
#
#     return loader(filepath, template_name)


def parse_path(path: str):
    """Parse a path of form path.to.template_file.template_name or path/to/template_file/template_name,
    returning a tuple of (name, file, abspath)."""

    if "/" in path or "\\" in path:
        import os

        # relative or absolute path of form:
        # path/to/file/TemplateName
        file, template_name = os.path.split(path)
        dirs, file = os.path.split(file)
        abspath = os.path.abspath(dirs)
    elif "." in path:
        *modules, file, template_name = path.split(".")

        # look for pyrates library and return absolute path
        parentdir = ".".join(modules)
        # let Python figure out where to look for the module
        try:
            module = importlib.import_module(parentdir)
        except ModuleNotFoundError:
            raise PyRatesException(f"Could not find Python (module) directory associated to path "
                                   f"`{parentdir}` of Template `{path}`.")
        try:
            abspath = module.__path__  # __path__ returns a list[str] or _NamespacePath
            abspath = abspath[0] if type(abspath) is list else abspath._path[0]
        except TypeError:
            raise PyRatesException(f"Something is wrong with the given YAML template path `{path}`.")

    else:
        raise NotImplementedError(f"Was base specified in template '{path}', but left empty?")
        # this should only happen, if "base" is specified, but empty

    return template_name, file, abspath
