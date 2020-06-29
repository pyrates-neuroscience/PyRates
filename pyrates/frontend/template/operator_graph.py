
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
from typing import Union, List, Type, Dict

from pyrates import PyRatesException
from pyrates.frontend.template import _complete_template_path
from pyrates.frontend.template.abc import AbstractBaseTemplate
from pyrates.ir.operator_graph import OperatorGraph
from pyrates.frontend.template.operator import OperatorTemplate


class OperatorGraphTemplate(AbstractBaseTemplate):
    target_ir = None

    def __init__(self, name: str, path: str, operators: Union[str, List[str], dict, List[AbstractBaseTemplate]],
                 description: str = "A node or an edge.", label: str = None):
        """For now: only allow single equation in operator template."""

        super().__init__(name, path, description)

        if label:
            self.label = label
        else:
            self.label = self.name

        self.operators = {}  # dictionary with operator path as key and variations to the template as values
        if isinstance(operators, str):
            operator_template = self._load_operator_template(operators)
            self.operators[operator_template] = {}  # single operator path with no variations
        elif isinstance(operators, list):
            for op in operators:
                try:
                    operator_template = self._load_operator_template(op)
                except TypeError:
                    operator_template = op
                self.operators[operator_template] = {}  # multiple operator paths with no variations
        elif isinstance(operators, dict):
            for op, variations in operators.items():
                try:
                    operator_template = self._load_operator_template(op)
                except TypeError:
                    operator_template = op
                self.operators[operator_template] = variations
        # for op, variations in operators.items():
        #     if "." not in op:
        #         op = f"{path.split('.')[:-1]}.{op}"
        #     self.operators[op] = variations

    def _load_operator_template(self, path: str) -> OperatorTemplate:
        """Load an operator template based on a path"""
        path = _complete_template_path(path, self.path)
        return OperatorTemplate.from_yaml(path)

    def update_template(self, name: str, path: str, label: str,
                        operators: Union[str, List[str], dict] = None,
                        description: str = None):
        """Update all entries of a base edge template to a more specific template."""

        if operators:
            _update_operators(self.operators, operators)
        else:
            operators = self.operators

        if not description:
            description = self.__doc__  # or do we want to enforce documenting a template?

        return self.__class__(name=name, path=path, label=label, operators=operators,
                              description=description)

    def apply(self, values: dict = None):
        """ Apply template to gain a node or edge intermediate representation.

        Parameters
        ----------
        values
            dictionary with operator/variable as keys and values to update these variables as items.

        Returns
        -------

        """

        value_updates = {}
        if values:
            for key, value in values.items():
                op_name, var_name = key.split("/")
                if op_name not in value_updates:
                    value_updates[op_name] = {}
                value_updates[op_name][var_name] = value

        operators = {}
        all_values = {}  # # type: Dict[OperatorIR, Dict]

        for template, variations in self.operators.items():
            values_to_update = variations

            if values_to_update is None:
                values_to_update = {}
            # if a value for this particular variation has been passed, overwrite the previous value
            if template.name in value_updates:
                values_to_update.update(value_updates.pop(template.name))
            # apply operator template to get OperatorIR and associated default values and label
            operator, op_values, key = template.apply(return_key=True, values=values_to_update)
            operators[key] = operator
            all_values[key] = op_values

        # fail gracefully, if any variables remain in `values` which means, that there is some typo
        if value_updates:
            raise PyRatesException(
                "Found value updates that did not fit any operator by name. This may be due to a "
                "typo in specifying the operator or variable to update. Remaining variables:"
                f"{value_updates}")
        return self.target_ir(operators=operators, values=all_values, template=self.path)


def _update_operators(base_operators: dict, updates: Union[str, List[str], dict]):
    """Update operators of a given template. Note that currently, only the new information is
    propagated into the operators dictionary. Comparing or replacing operators is not implemented.

    Parameters:
    -----------

    base_operators:
        Reference to one or more operators in the base class.
    updates:
        Reference to one ore more operators in the child class
        - string refers to path or name of single operator
        - list refers to multiple operators of the same class
        - dict contains operator path or name as key
    """
    # updated = base_operators.copy()
    updated = {}
    if isinstance(updates, str):
        updated[updates] = {}  # single operator path with no variations
    elif isinstance(updates, list):
        for path in updates:
            updated[path] = {}  # multiple operator paths with no variations
    elif isinstance(updates, dict):
        for path, variations in updates.items():
            updated[path] = variations
        # dictionary with operator path as key and variations as sub-dictionary
    else:
        raise TypeError("Unable to interpret type of operator updates. Must be a single string,"
                        "list of strings or dictionary.")
    # # Check somewhere, if child operators have same input/output as base operators?
    #
    return updated




