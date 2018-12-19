
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

from pyrates import PyRatesException
from pyrates.frontend.parser.yaml import load_template_from_yaml

__author__ = "Daniel Rose"
__status__ = "Development"

file_loader_mapping = {"yaml": load_template_from_yaml,
                       "yml": load_template_from_yaml}


def load_template_from_file(filepath: str, template_name: str):
    file, abspath = parse_path(filepath)
    filename, extension = file.split(".")
    try:
        loader = file_loader_mapping[extension]
    except KeyError:
        raise PyRatesException(f"Could not find loader for file extension {extension}.")

    return loader(filepath, template_name)


def parse_path(path: str):
    """Parse a path of form path.to.template, returning a tuple of (name, file, abspath)."""
    # ToDo: Add parent path as required argument, to better locate errors in YAML templates.

    if "/" in path:
        *dirs, file = path.split("/")
        dirs = "/".join(dirs)
        *file, name = file.split(".")
        file = ".".join(file)
        import os
        abspath = os.path.abspath(dirs)
    else:
        if "." in path:
            parts = path.split(".")
            name = parts[-1]

            if parts[0] == "pyrates":
                # look for pyrates library and return absolute path
                file = parts[-2]
                parentdir = ".".join(parts[:-2])
                # let Python figure out where to look for the module
                try:
                    module = importlib.import_module(parentdir)
                except ModuleNotFoundError:
                    raise PyRatesException(f"Could not find Python (module) directory associated to path "
                                           f"`{parentdir}` of Template `{path}`.")
                try:
                    abspath = module.__path__[0]  # __path__ returns a list[str]
                except TypeError:
                    raise PyRatesException(f"Something is wrong with the given YAML template path `{path}`.")

            else:
                # import custom defined backend with relative or absolute path
                import os
                file = os.path.join(*parts[:-1])
                abspath = ""  # empty filepath

        else:
            raise NotImplementedError(f"Was base specified in template '{path}', but left empty?")
            # this should only happen, if "base" is specified, but empty

    return name, file, abspath
