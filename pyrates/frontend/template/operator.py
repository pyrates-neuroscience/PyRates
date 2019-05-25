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
# external imports
import re
from copy import deepcopy
from typing import Union

# pyrates internal imports
from pyrates import PyRatesException
from pyrates.frontend.template.abc import AbstractBaseTemplate
from pyrates.frontend.utility import deep_freeze

# meta infos
from pyrates.ir.operator import OperatorIR

__author__ = " Daniel Rose"
__status__ = "Development"


class OperatorTemplate(AbstractBaseTemplate):
    """Generic template for an operator with a name, equations, variables and possible
    initialization conditions. The template can be used to create variations of a specific
    equations or variables."""

    cache = {}  # tracks all unique instances of applied operator templates
    target_ir = OperatorIR
    key_map = {}
    key_counter = {}

    # key_map = {}  # tracks renaming of of operator keys to have shorter unique keys

    def __init__(self, name: str, path: str, equations: Union[list, str], variables: dict,
                 description: str = "An operator template."):
        """For now: only allow single equations in operator template."""

        super().__init__(name, path, description)

        if isinstance(equations, str):
            self.equations = [equations]
        else:
            self.equations = equations
        self.variables = variables

    def update_template(self, name: str, path: str, equations: Union[str, list, dict] = None,
                        variables: dict = None, description: str = None):
        """Update all entries of the Operator template in their respective ways."""

        if equations:
            # if it is a string, just replace
            if isinstance(equations, str):
                equations = [equations]
            elif isinstance(equations, list):
                pass  # pass equations string to constructor
            # else, update according to predefined rules, assuming dict structure
            elif isinstance(equations, dict):
                equations = [_update_equation(eq, **equations) for eq in self.equations]

            else:
                raise TypeError("Unknown data type for attribute 'equations'.")
        else:
            # copy equations from parent template
            equations = self.equations

        if variables:
            variables = _update_variables(self.variables, variables)
        else:
            variables = self.variables

        rogue_variables = set()
        for var in variables:
            # remove variables that are not present in the equations
            found = False
            for eq in equations:
                if var in eq:
                    found = True

            if not found:
                # save entries in list, since dictionary must not change size during iteration
                rogue_variables.add(var)

        for var in rogue_variables:
            variables.pop(var)

        if not description:
            description = self.__doc__  # or do we want to enforce documenting a template?

        return self.__class__(name=name, path=path, equations=equations, variables=variables,
                              description=description)

    def apply(self, return_key=False, values: dict = None):
        """Returns the non-editable but unique, cashed definition of the operator."""

        # figure out key name of given combination of template and assigned values
        name = self.name
        if values:
            # if values are given, figure out, whether combination is known
            frozen_values = deep_freeze(values)
            try:
                key = self.key_map[(name, frozen_values)]
            except KeyError:
                # not known, need to assign new key and register key in key_map
                if name in self.key_counter:
                    key = f"{name}.{self.key_counter[name] + 1}"
                    self.key_counter[name] += 1
                else:
                    key = f"{name}.1"
                    self.key_counter[name] = 1
                self.key_map[(name, frozen_values)] = key

        else:  # without additional values specified, default name is op_name.0
            key = f"{name}.0"

        try:
            instance, variables = self.cache[key]
            # instance = defrost(instance)
            # variables = defrost(variables)
            # instance, variables = dict(instance), dict(variables)
        except KeyError:
            # get variable definitions and specified default values
            # ToDo: remove variable separation: instead pass variables detached from equations?
            variables, inputs, output = self._separate_variables(self)

            # reduce order of ODE if necessary
            # this step is currently skipped to streamline equation interface
            # instead equations need to be given as either non-differential equations or first-order
            # linear differential equations
            # *equations, variables = cls._reduce_ode_order(template.equations, variables)
            equations = self.equations

            # operator instance is invoked as a dictionary of equations and variable definition
            # this may be subject to change

            if values:
                for vname, update in values.items():
                    # if isinstance(update, dict):
                    #     if "value" in update:
                    #         variables[vname["value"]] = update["value"]
                    #     if "vtype" in
                    # only values are updated, assuming all other specs remain the same
                    variables[vname]["value"] = values[vname]
                    # should fail, if variable is unknown

            instance = self.target_ir(equations=equations, inputs=inputs, output=output)
            self.cache[key] = (instance, variables)

        if return_key:
            return instance.copy(), deepcopy(variables), key
        else:
            return instance.copy(), deepcopy(variables)

    @classmethod
    def _separate_variables(cls, template):
        """
        Return variable definitions and the respective values.

        Returns
        -------
        variables
        inputs
        output
        """
        # this part can be improved a lot with a proper expression parser

        variables = {}
        inputs = {}
        output = None
        for variable, properties in template.variables.items():
            var_dict = deepcopy(properties)
            # default shape is scalar
            if "shape" not in var_dict:
                var_dict["shape"] = (1,)

            # identify variable type and data type
            # note: this assume that a "default" must be given for every variable
            try:
                var_dict.update(cls._parse_vprops(var_dict.pop("default")))
            except KeyError:
                raise PyRatesException("Variables need to have a 'default' (variable type, data type and/or value) "
                                       "specified.")

            # separate in/out specification from variable type specification
            if var_dict["vtype"] == "input":
                inputs[variable] = dict(sources=[], reduce_dim=True)  # default to True for now
                var_dict["vtype"] = "state_var"
            elif var_dict["vtype"] == "output":
                if output is None:
                    output = variable  # for now assume maximum one output is present
                else:
                    raise PyRatesException("More than one output specification found in operator. "
                                           "Only one output per operator is supported.")
                var_dict["vtype"] = "state_var"

            variables[variable] = var_dict

        return variables, inputs, output

    @staticmethod
    def _parse_vprops(expr: Union[str, int, float]):
        """Naive version of a parser for the default key of variables in a template. Returns data type,
        variable type and default value of the variable."""

        value = 0.  # Setting initial conditions for undefined variables to 0. Is that reasonable?
        if isinstance(expr, int):
            vtype = "constant"
            value = expr
            # dtype = "int32"
            dtype = "float32"  # default to float for maximum compatibility
        elif isinstance(expr, float):
            vtype = "constant"
            value = expr
            dtype = "float32"
            # restriction to 32bit float for consistency. May not be reasonable at all times.
        else:
            # set vtype
            if expr.startswith("input"):
                vtype = "input"
            elif expr.startswith("output"):
                vtype = "output"
            elif expr.startswith("variable"):
                vtype = "state_var"
            elif expr.startswith("constant"):
                vtype = "constant"
            elif expr.startswith("placeholder"):
                vtype = "placeholder"
            else:
                try:
                    # if "." in expr:
                    value = float(expr)  # default to float
                    # else:
                    #     value = int(expr)
                    vtype = "constant"
                except ValueError:
                    raise ValueError(f"Unable to interpret variable type in default definition {expr}.")

            # set dtype and value
            if expr.endswith("(float)"):
                dtype = "float32"  # why float32 and not float64?
            elif expr.endswith("(int)"):
                # dtype = "int32"
                dtype = "float32"  # default to float for maximum compatibility
            elif "." in expr:
                dtype = "float32"
                value = float(re.search("[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)", expr).group())
                # see https://stackoverflow.com/questions/12643009/regular-expression-for-floating-point-numbers
            elif re.search("[0-9]+", expr):
                # dtype = "int32"
                dtype = "float32"  # default to float for maximum compatibility
                value = float(re.search("[0-9]+", expr).group())
            else:
                dtype = "float32"  # base assumption

        return dict(vtype=vtype, dtype=dtype, value=value)


def _update_variables(variables: dict, updates: dict):
    updated = deepcopy(variables)

    for var, var_dict in updates.items():
        if var in updated:
            # update dictionary defining single variable
            updated[var].update(var_dict)
        else:
            # copy new variable into variables dictionary
            updated.update({var: var_dict})

    return updated


def _update_equation(equation: str,  # original equation
                     replace: dict = False,  # replace parts of the string
                     remove: Union[list, tuple] = False,  # remove parts of the string
                     append: str = False,  # append to the end of the string
                     prepend: str = False,  # add to beginning of string
                     ):
    # replace existing terms by new ones
    if replace:
        for old, new in replace.items():
            equation = equation.replace(old, new)
            # this might fail, if multiple replacements refer or contain the same variables
            # is it possible to call str.replace with tuples?

    # remove terms
    if remove:
        if isinstance(remove, str):
            equation = equation.replace(remove, "")
        else:
            for old in remove:
                equation = equation.replace(old, "")

    # append terms at the end of the equation string
    if append:
        # only allowing single append per update
        equation = f"{equation} {append}"

        # prepend terms at the beginning of the equation string
    if prepend:
        # only allowing single prepend per update
        equation = f"{prepend} {equation}"

    return equation
