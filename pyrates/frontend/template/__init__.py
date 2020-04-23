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
from ._io import _complete_template_path
from .node import NodeTemplate
from .operator import OperatorTemplate
from .edge import EdgeTemplate
from .circuit import CircuitTemplate

known_template_classes = dict()

template_cache = dict()


def register_template_class(name, cls):
    """Register a given template class to the module attribute `_known_template_classes`. This way new template classes
     can be registered by users. Could also be used to overwrite existing template classes."""

    if name in known_template_classes:
        raise UserWarning(f"Overwriting existing map from name `{name}` to template class `{cls}`.")

    known_template_classes[name] = cls


register_template_class("OperatorTemplate", OperatorTemplate)
register_template_class("NodeTemplate", NodeTemplate)
register_template_class("EdgeTemplate", EdgeTemplate)
register_template_class("CircuitTemplate", CircuitTemplate)


def from_file(path: str, mode: str = "yaml"):
    """Generic file loader function that looks for correct template class"""

    if mode == "yaml":
        loader = from_yaml

    else:
        raise ValueError(f"Unknown file loading mode '{mode}'.")

    return loader(path)


def from_yaml(path):
    """Load template from yaml file. Templates are cached by path. Depending on the 'base' key of the yaml template,
    either a template class is instantiated or the function recursively loads base templates until it hits a known
    template class.

    Parameters:
    -----------
    path
        (str) path to YAML template of the form `path.to.template_file.template_name` or
        path/to/template_file/template_name.TemplateName. The dot notation refers to a path that can be found
        using python's import functionality. That means it needs to be a module (a folder containing an `__init__.py`)
        located in the Python path (e.g. the current working directory). The slash notation refers to a file in an
        absolute or relative path from the current working directory. In either case the second-to-last part refers to
        the filename without file extension and the last part refers to the template name.
    """

    if path in template_cache:
        # if we have loaded this template in the past, return what has been cached
        template = template_cache[path]
    else:
        # if it has not been cached yet, load the file and parse into dict
        from pyrates.frontend.fileio.yaml import dict_from_yaml
        template_dict = dict_from_yaml(path)

        try:
            base = template_dict.pop("base")
        except KeyError:
            raise KeyError(f"No 'base' defined for template {path}. Please define a "
                           f"base to derive the template from.")

        # figure out which type of template this is by analysing the "base" key
        try:
            # If the base key coincides with any known template class name, fetch the class
            cls = known_template_classes[base]

        except KeyError:
            # class not known, so the base must refer to a parent template. Then let's recursively load that one until
            # we hit a known template class.
            base = _complete_template_path(base, path)

            base_template = from_yaml(base)
            template = base_template.update_template(**template_dict)
            # may fail if "base" is present but empty
        else:
            # instantiate template class
            template = cls(**template_dict)

        template_cache[path] = template

    return template


def clear_cache():
    """Shorthand to clear template cache for whatever reason."""
    template_cache.clear()


def _select_template_class():
    pass


# module-lvl functions for template conversion
# writing them out explicitly

def to_circuit(template: CircuitTemplate):
    """Takes a circuit template and returns a CircuitIR instance from it."""
    return template.apply()



def to_node(template: NodeTemplate):
    """Takes a node template and returns a NodeIR instance from it."""
    return template.apply()



def to_edge(template: EdgeTemplate):
    """Takes a edge template and returns a EdgeIR instance from it."""
    return template.apply()



def to_operator(template: OperatorTemplate):
    """Takes a operator template and returns a OperatorIR instance from it."""
    return template.apply()
