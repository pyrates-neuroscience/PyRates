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
"""
"""
from collections import namedtuple as _namedtuple
from copy import deepcopy
from typing import List, Iterator

from pyrates.ir.abc import AbstractBaseIR

__author__ = "Daniel Rose"
__status__ = "Development"

Variable = _namedtuple("Variable", ["vtype", "dtype", "shape"])


class ProtectedVariableDict:
    """Hashable dictionary-like object as container for variable definitions. It does not support item assignment after
    creation, but is strictly speaking not immutable. There may also be faster implementations, but this works for
    now."""

    __slots__ = ["_hash", "_d", "_parsed"]

    def __init__(self, variables: List[tuple]):
        variables = tuple(variables)
        self._hash = hash(variables)
        self._d = {vname: dict(vtype=vtype, dtype=dtype, shape=shape)
                   for vname, vtype, dtype, shape in variables}

    def add_parsed_variable(self, key, props):
        """Add parsed representation to a variable from compilation."""
        self._parsed[key] = props

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def __hash__(self):
        return self._hash

    def items(self):
        return self._d.items()

    def keys(self):
        return self._d.keys()

    def values(self):
        return self._d.values()

    def to_dict(self):
        return deepcopy(self._d)

# class ProtectedVariableDict(dict):
#     """This is an unsafe hack to provide an immutable and hashable dict. Unsafe means, that checks against isinstance
#     or
#     issubtype might lead to wrong conclusions about the mutability. There may also be faster implementations, but this
#     works for now."""
#
#     def __init__(self, variables: List[tuple]):
#         variables = tuple(variables)
#         self._h = hash(variables)
#
#         named_variables = {vname: Variable(vtype, dtype, shape)
#                            for vname, vtype, dtype, shape in variables}
#
#         super().__init__(**named_variables)
#
#     def __setitem__(self, key, value):
#         raise TypeError("'VariableDict' object does not support item assignment")
#
#     # def __setattr__(self, key, value):
#     #     raise AttributeError(f"'VariableDict' object has not attribute '{key}'")
#
#     def __hash__(self):
#         return self._h


class OperatorIR(AbstractBaseIR):
    """This implementation of the Operator IR is aimed to be hashable and immutable. Following Python standards, we
    assume that users are consenting adults. Objects are thus not actually immutable, just slightly protected.
    This might change in the future."""
    __slots__ = ["_equations", "_variables", "_inputs", "_output"]

    def __init__(self, equations: List[str], variables: List[tuple], inputs: List[str], output: str,
                 template: str = None):

        super().__init__(template)
        # define hash

        self._equations = tuple(equations)

        self._variables = ProtectedVariableDict(variables)
        self._inputs = tuple(inputs)
        self._output = output
        self._h = hash((self._equations, self._variables, self._inputs, self._output))

    @property
    def variables(self):
        return self._variables

    @property
    def equations(self):
        return self._equations

    @property
    def inputs(self):
        return self._inputs

    @property
    def output(self):
        return self._output

    def getitem_from_iterator(self, key: str, key_iter: Iterator[str]):
        """
        Checks if a variable named by key exists in an equations.
        Parameters
        ----------
        key
        key_iter

        Returns
        -------
        key
        """

        for equation in self.equations:
            if key in equation:
                return key
        else:
            raise KeyError(f"Variable `{key}` not found in equations {self.equations}")

    def __str__(self):

        return f"<{self.__class__.__name__}({self.equations}), hash = {hash(self)} >"
