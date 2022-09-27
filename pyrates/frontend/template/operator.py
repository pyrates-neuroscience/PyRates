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
# external _imports
from copy import deepcopy
from sys import intern
from typing import Union
from warnings import warn

# pyrates internal _imports
from pyrates.ir.circuit import PyRatesException, PyRatesWarning
from pyrates.frontend.template.abc import AbstractBaseTemplate
from pyrates.backend.parser import replace as eq_replace

# meta infos
from pyrates.ir.operator import OperatorIR

__author__ = " Daniel Rose, Richard Gast"
__status__ = "Development"


class OperatorTemplate(AbstractBaseTemplate):
    """Generic template for an operator with a name, equations, variables and possible
    initialization conditions. The template can be used to create variations of a specific
    equations or variables."""

    cache = {}  # tracks all unique instances of applied operator templates
    target_ir = OperatorIR

    def __init__(self, name: str, equations: Union[list, str], variables: dict, path: str = None,
                 description: str = "An operator template."):
        """For now: only allow single equations in operator template."""

        super().__init__(name, path, description)

        if isinstance(equations, str):
            self.equations = [intern(equations)]
        else:
            self.equations = [intern(eq) for eq in equations]
        self.variables = variables

    def __getitem__(self, item):
        """Attempts to return the variable with name `item`.
        """
        try:
            return self.variables[item]
        except KeyError:
            warn(PyRatesWarning(f"Variable with name {item} was not found on {self.name}."))
            return

    def update_template(self, name: str = None, path: str = None, equations: Union[str, list, dict] = None,
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
                new_eqs = equations.pop('add', [])
                equations = [_update_equation(eq, **equations) for eq in self.equations] + new_eqs
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
        if not name:
            name = self.name
        if not path:
            path = self.path

        # decide whether template requires a new key or not
        return self.__class__(name=name, path=path, equations=equations, variables=variables,
                              description=description)

    def apply(self, return_key=False, values: dict = None):
        """Returns the non-editable but unique, cashed definition of the operator."""

        # key for global operator cache is the frozen list of equation strings.
        key = self.name
        if values is None:
            values = {}

        try:
            instance, default_values = self.cache[key]

            for vname, value in default_values.items():
                if vname not in values:
                    values[vname] = value

        except KeyError:
            # get variable definitions and specified default values
            variables = []
            default_values = dict()
            inputs = []
            output = None
            for vname, vtype, dtype, shape, value in _separate_variables(self.variables):

                # check whether an invalid variable name was passed
                vtype = check_vname(vname, vtype)

                # if no new value is given for a variable, get the default
                if vname not in values:
                    values[vname] = value

                default_values[vname] = value

                if vtype == "input":
                    inputs.append(vname)  # = dict(sources=[], reduce_dim=True)  # default to True for now
                    vtype = "input"
                elif vtype == "output":
                    if output is None:
                        output = vname  # for now assume maximum one output is present
                    else:
                        raise PyRatesException("More than one output specification found in operator. "
                                               "Only one output per operator is supported.")
                    vtype = "state_var"
                # pack variable defining properties into tuple
                variables.append((vname, vtype, dtype, shape))

            equations = self.equations
            instance = self.target_ir(equations=equations, variables=variables, inputs=inputs, output=output,
                                      template=self)
            self.cache[key] = (instance, default_values)

        if return_key:
            return instance, values, key
        else:
            return instance, values


def check_vname(v: str, vtype: str):

    disallowed_names = ['y', 'dy', 'source_idx', 'target_idx', 'pi', 'I', 'E']
    disallowed_name_parts = ['_buffer', '_delays', '_maxdelay', '_idx', '_hist']

    if v == 't' and vtype != 'state_var':
        vtype = 'state_var'

    if v in disallowed_names:
        raise PyRatesException(f'The variable name {v} is reserved for internal variables created by PyRates. '
                               f'Please choose a different variable name.')

    for dpart in disallowed_name_parts:
        if dpart in v:
            raise PyRatesException(f'The variable name {v} contains the sub-string {dpart} which is reserved for '
                                   f'internal variables created by PyRates. Please choose a different variable name.')

    return vtype


def _update_variables(variables: dict, updates: dict):
    updated = variables.copy()
    updated.update(updates)
    # for var, val in updates.items():
    #     if var in updated:
    #         # update dictionary defining single variable
    #         updated[var] = val
    #     else:
    #         # copy new variable into variables dictionary
    #         updated.update({var: var_dict})

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
            equation = eq_replace(equation, old, new)
            #equation = intern(equation.replace(old, new))
            # this might fail, if multiple replacements refer or contain the same variables
            # is it possible to call str.replace with tuples?

    # remove terms
    if remove:
        if isinstance(remove, str):
            equation = intern(eq_replace(equation, remove, ""))
        else:
            for old in remove:
                equation = intern(eq_replace(equation, old, ""))

    # append terms at the end of the equation string
    if append:
        # only allowing single append per update
        equation = intern(f"{equation} {append}")

        # prepend terms at the beginning of the equation string
    if prepend:
        # only allowing single prepend per update
        equation = intern(f"{prepend} {equation}")

    return equation


def _parse_defaults(expr: Union[str, int, float]) -> dict:
    """Naive version of a parser for the default key of variables in a template. Returns data type,
    variable type and default value of the variable."""

    value = 0.
    shape = (1,)
    if isinstance(expr, int):
        vtype = "constant"
        value = expr
        dtype = "int"
    elif isinstance(expr, float):
        vtype = "constant"
        value = expr
        dtype = "float"
    elif isinstance(expr, complex):
        vtype = "constant"
        value = expr
        dtype = "complex"
    else:
        # set vtype
        if expr.startswith("input"):
            vtype = "input"
        elif expr.startswith("output"):
            vtype = "output"
        elif expr.startswith("variable"):
            vtype = "state_var"
        else:
            vtype = "constant"
            try:
                value = float(expr)
            except ValueError:
                try:
                    value = complex(expr)
                except ValueError:
                    raise ValueError(f"Unable to interpret variable type in default definition {expr}.")

        dtype, value, shape = _get_variable_info(expr, value)

    return {'vtype': vtype, 'dtype': dtype, 'value': value, 'shape': shape}


def _separate_variables(variables: dict):
    """
    Return variable definitions and the respective values.

    Returns
    -------
    variables
    inputs
    output
    """
    # this part can be improved a lot with a proper expression parser

    for vname, vinfo in variables.items():

        # identify variable type and data type
        default_vals = vinfo.copy() if type(vinfo) is dict else _parse_defaults(vinfo)
        yield vname, default_vals['vtype'], default_vals['dtype'], default_vals['shape'], default_vals['value']


def _get_variable_info(expr: Union[str, int, float], value: Union[int, complex, float]) -> tuple:
    # set dtype and value and shape
    shape = (1,)
    if expr.endswith("(float)"):
        dtype = "float"
    elif expr.endswith("(complex)"):
        dtype = "complex"
        if not isinstance(value, complex):
            value = complex(value, value)
    elif "(" in expr:
        expr = expr.split("(")[1][:-1]
        if "," in expr:
            expr, var_len = expr.split(",")
            var_len = int(var_len)
            dtype, value, _ = _get_variable_info(expr, value)
            value = [value for _ in range(var_len)]
            shape = (var_len,)
        else:
            dtype, value, shape = _get_variable_info(expr, value)
    else:
        try:
            value = float(expr)
            dtype = "float"
        except ValueError:
            try:
                value = complex(expr)
                dtype = "complex"
            except ValueError:
                dtype = "float"  # base assumption

    return dtype, value, shape
