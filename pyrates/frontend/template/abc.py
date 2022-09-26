
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

__author__ = "Daniel Rose"
__status__ = "Development"


class AbstractBaseTemplate:
    """Abstract base class for templates"""

    target_ir = None  # placeholder for template-specific intermediate representation (IR) target class

    def __init__(self, name: str, path: str = None, description: str = "A template."):
        """Basic initialiser for template classes, requires template name and path that it is loaded from. For custom
        templates that are not loaded from a file, the path can be set arbitrarily."""
        self.name = name
        self.path = path
        self.__doc__ = description  # overwrite class-specific doc with user-defined description

    def __repr__(self):
        """Defines how an instance identifies itself when called with `str()` or `repr()`, e.g. when shown in an
        interactive terminal. Shows Class name and path that was used to construct the class."""
        return f"<{self.__class__.__name__} '{self.name}'>"

    def __getitem__(self, item):
        raise NotImplementedError

    @classmethod
    def from_yaml(cls, path):
        """Load a template from yaml file. After importing the template, this method also checks whether
        the resulting template is actually an instance of the class that this method was called from. This is done to
        ensure any cls.from_yaml() produces only instances of that class and not other classes for consistency.
        Templates are cached by path. Depending on the 'base' key of the yaml template,
        either a template class is instantiated or the function recursively loads base templates until it hits a known
        template class.

        Parameters
        ----------
        path
            (str) path to YAML template of the form `path.to.template_file.template_name` or
            path/to/template_file/template_name.TemplateName. The dot notation refers to a path that can be found
            using python's import functionality. That means it needs to be a module (a folder containing an
            `__init__.py`) located in the Python path (e.g. the current working directory). The slash notation refers to
            a file in an absolute or relative path from the current working directory. In either case the second-to-last
            part refers to the filename without file extension and the last part refers to the template name.

        Returns
        -------

        """
        from pyrates.frontend.template import from_yaml
        tpl = from_yaml(path)

        if isinstance(tpl, cls):
            return tpl
        else:
            raise TypeError(f"The template associated with '{path}' is not of type {cls}.")

    def to_yaml(self, path, **kwargs) -> None:
        """Saves template to YAML file."""
        raise NotImplementedError

    def update_template(self, *args, **kwargs):
        """Updates the template with a given list of arguments and returns a new instance of the template class."""
        raise NotImplementedError

    def apply(self, *args, **kwargs):
        """Converts the template into its corresponding intermediate representation class."""
        raise NotImplementedError


