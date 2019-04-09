
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
""" Abstract base classes
"""
from importlib import import_module

# from pyrates.ir.abc import AbstractBaseIR

__author__ = "Daniel Rose"
__status__ = "Development"


class AbstractBaseTemplate:
    """Abstract base class for templates"""

    target_ir = None  # placeholder for template-specific intermediate representation (IR) target class

    def __init__(self, name: str, path: str, description: str = "A template."):
        self.name = name
        self.path = path
        self.__doc__ = description  # overwrite class-specific doc with user-defined description

    def __repr__(self):
        return f"<{self.__class__.__name__} '{self.path}'>"

    def _format_path(self, path):
        """Check if path contains a folder structure and prepend own path, if it doesn't"""
        # ToDo: rename to something more meaningful like _prepend_parent_path or _check_path_prepend_parent

        if "." not in path:
            if "/" in self.path or "\\" in self.path:
                import os
                basedir, _ = os.path.split(self.path)
                path = os.path.normpath(os.path.join(basedir, path))
            else:
                path = ".".join((*self.path.split('.')[:-1], path))
        return path

    @classmethod
    def from_yaml(cls, path):
        """Convenience method that looks for a loader class for the template type and applies it, assuming
        the class naming convention '<template class>Loader'.

        Parameters:
        -----------
        path
            Path to template in YAML file of form 'directories.file.template'
        """
        # ToDo: add AbstractBaseTemplate._format_path functionality here
        module = import_module(cls.__module__)
        loader = getattr(module, f"{cls.__name__}Loader")
        return loader(path)

    def apply(self, *args, **kwargs):

        raise NotImplementedError


