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
"""Functionality to register functions that are used to transform between different data types in the frontend.
"""

__author__ = "Daniel Rose"
__status__ = "Development"

REGISTERED_INTERFACES = dict()


def register_interface(func, name=""):
    """Register a transformation function (interface) between two representations of models.

    Parameters
    ----------
    func
        Function to be registered. Needs to start with "from_" or "to_" to signify the direction of transformation
    name
        (Optional) String that defines the name under which the function should be registered. If left empty,
        the name will be formatted in the form {target}_from_{source}, where target and source are representations to
        transform from or to."""
    if name is "":

        # get interface name from module name
        module_name = func.__module__.split(".")[-1]
        # parse to_ and from_ functions
        func_name = func.__name__
        if func_name.startswith("from_"):
            target = module_name
            source = func_name[5:]  # crop 'from_'
        elif func_name.startswith("to_"):
            source = module_name
            target = func_name[3:]  # crop 'to_'
        else:
            raise ValueError(f"Function name {func_name} does not adhere to convention to start "
                             f"with either `to_` or `from_`.")  # ignore any other functions
        new_name = f"{target}_from_{source}"
    else:
        new_name = name

    if new_name in REGISTERED_INTERFACES:
        raise ValueError(f"Interface {new_name} already exist. Cannot add {func}.")
    else:
        REGISTERED_INTERFACES[new_name] = func

    return func
