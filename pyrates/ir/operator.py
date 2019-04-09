
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
import re
from copy import deepcopy
from typing import List, Union, Iterator

from pyrates import PyRatesException
from pyrates.frontend.utility import deep_freeze
from pyrates.ir.abc import AbstractBaseIR

__author__ = "Daniel Rose"
__status__ = "Development"


class OperatorIR(AbstractBaseIR):

    def __init__(self, equations: List[str], inputs: list, output: str, template: str=None):

        super().__init__(template)
        self.output = output
        self.inputs = inputs
        self.equations = equations

    # @staticmethod
    # def _reduce_ode_order(equations: str, variables):
    #     """Checks if a 2nd-order ODE is present and reduces it to two coupled first-order ODEs.
    #     Currently limited to special case of the form '(d/dt + a)^2 * x = b'.
    #
    #     Parameters
    #     ----------
    #     equations
    #         string of form 'a = b'
    #     """
    #
    #     # matches pattern of form `(d/dt + a)^2 * y` and extracts `a` and `y`
    #     match = re.match(r"\(\s*d\s*/\s*dt\s*[+-]\s*([\d]*[.]?[\d]*/?[a-zA-Z]\w*)\s*\)\s*\^2\s*\*\s*([a-zA-Z]\w*)",
    #                      equations)
    #
    #     if match:
    #         # assume the entire lhs was matched, fails if there is something remaining on the lhs
    #         lhs, rhs = equations.split("=")
    #         a, var = match.groups()  # returns coefficient `a` and variable `y`
    #         eq1 = f"d/dt * {var} = {var}_t"
    #         eq2 = f"d/dt * {var}_t = {rhs} - ({a})^2 * {var} - 2. * {a} * {var}_t"
    #
    #         variables[f"{var}_t"] = {"dtype": "float32",
    #                                  "description": "integration variable",
    #                                  "vtype": "state_var",
    #                                  "value": variables[var]['value'] if variables[var]['value'] else 0.}
    #
    #         return eq1, eq2, variables
    #     else:
    #         return equations, variables

    def copy(self):

        return self.__class__(deepcopy(self.equations), deepcopy(self.inputs), deepcopy(self.output))

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