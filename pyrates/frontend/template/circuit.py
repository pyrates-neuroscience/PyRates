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
"""Basic neural mass backend class plus derivations of it.

This module includes the base circuit class that manages the set-up and simulation of population networks. Additionally,
it features various sub-classes that act as circuit constructors and allow to build circuits based on different input
arguments. For more detailed descriptions, see the respective docstrings.

"""

# external packages
from typing import List, Union, Dict, Optional
from copy import deepcopy

# pyrates internal imports
from pyrates import PyRatesException
from pyrates.frontend.template._io import _complete_template_path
from pyrates.frontend.template.abc import AbstractBaseTemplate
from pyrates.frontend.template.edge import EdgeTemplate
from pyrates.frontend.template.node import NodeTemplate

# from pyrates.frontend.operator import OperatorTemplate

# meta infos
from pyrates.ir.circuit import CircuitIR
from pyrates.ir.edge import EdgeIR

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


class CircuitTemplate(AbstractBaseTemplate):
    target_ir = CircuitIR

    def __init__(self, name: str, path: str, description: str = "A circuit template.", label: str = "circuit",
                 circuits: dict = None, nodes: dict = None, edges: List[tuple] = None):

        super().__init__(name, path, description)

        self.nodes = {}  # type: Dict[str, NodeTemplate]
        if nodes:
            for key, node in nodes.items():
                if isinstance(node, str):
                    path = _complete_template_path(node, self.path)
                    self.nodes[key] = NodeTemplate.from_yaml(path)
                else:
                    self.nodes[key] = node

        self.circuits = {}
        if circuits:
            for key, circuit in circuits.items():
                if isinstance(circuit, str):
                    path = _complete_template_path(circuit, self.path)
                    self.circuits[key] = CircuitTemplate.from_yaml(path)
                else:
                    self.circuits[key] = circuit

        if edges:
            self.edges = self._get_edge_templates(edges)
        else:
            self.edges = []

        self.label = label

    def update_template(self, name: str, path: str, description: str = None,
                        label: str = None, circuits: dict = None, nodes: dict = None,
                        edges: List[tuple] = None):
        """Update all entries of the circuit template in their respective ways."""

        if not description:
            description = self.__doc__

        if not label:
            label = self.label

        if nodes:
            nodes = update_dict(self.nodes, nodes)
        else:
            nodes = self.nodes

        if circuits:
            circuits = update_dict(self.circuits, circuits)
        else:
            circuits = self.circuits

        if edges:
            edges = update_edges(self.edges, edges)
        else:
            edges = self.edges

        return self.__class__(name=name, path=path, description=description,
                              label=label, circuits=circuits, nodes=nodes,
                              edges=edges)

    def apply(self, label: str = None, node_values: dict = None, edge_values: dict = None):
        """Create a Circuit graph instance based on the template


        Parameters
        ----------
        label
            (optional) Assign a label that is saved as a sort of name to the circuit instance. This is particularly
            relevant, when adding multiple circuits to a bigger circuit. This way circuits can be identified by their
            given label.
        node_values
            (optional) Dictionary containing values (and possibly other variable properties) that overwrite defaults in
            specific nodes/operators/variables. Values must be given in the form: {'node/op/var': value}
        edge_values
            (optional) Dictionary containing source and target variable pairs as items and value dictionaries as values
            (e.g. {('source/op1/var1', 'target/op1/var2'): {'weight': 0.3, 'delay': 1.0}}). Can be used to overwrite
            default values defined in template.

        Returns
        -------

        """
        if not label:
            label = self.label
        if not edge_values:
            edge_values = {}

        # reformat node templates to NodeIR instances
        if node_values is None:
            nodes = {key: temp.apply() for key, temp in self.nodes.items()}
        else:
            values = dict()
            for key, value in node_values.items():
                node, op, var = key.split("/")
                if node not in values:
                    values[node] = dict()

                values[node]["/".join((op, var))] = value
            nodes = {key: temp.apply(values.get(key, None)) for key, temp in self.nodes.items()}

        # reformat edge templates to EdgeIR instances
        edges = []
        for (source, target, template, values) in self.edges:

            # get edge template and instantiate it
            values = deepcopy(values)

            # update edge template default values with passed edge values, if source and target are simple variable keys
            if type(source) is str and type(target) is str and (source, target) in edge_values:
                values.update(edge_values[(source, target)])
            weight = values.pop("weight", 1.)

            # get delay
            delay = values.pop("delay", None)
            spread = values.pop("spread", None)

            # treat additional source assignments in values dictionary
            extra_sources = {}
            keys_to_remove = []
            for key, value in values.items():
                try:
                    _, _, _ = value.split("/")
                except AttributeError:
                    # not a string
                    continue
                except ValueError as e:
                    raise ValueError(f"Wrong format of source specifier. Expected form: `node/op/var`. "
                                     f"Actual form: {value}")
                else:
                    # was actually able to split that string? Then it is an additional source specifier.
                    # Let's treat it as such
                    try:
                        op, var = key.split("/")
                    except ValueError as e:
                        if e.args[0].startswith("not enough"):
                            # No operator specified: assume that it is a known input variable. If not, we will notice
                            # later.
                            var = key
                            # Need to remove the key, but need to wait until after the iteration finishes.
                            keys_to_remove.append(key)
                        else:
                            raise e
                    else:
                        # we know which variable in which operator, so let us set its type to "input"
                        values[key] = "input"

                    # assuming, the variable has been set to "input", we can omit any operator description and only
                    # pass the actual variable name
                    extra_sources[var] = value

            for key in keys_to_remove:
                values.pop(key)

            # treat empty dummy edge templates as not existent templates
            if template and len(template.operators) == 0:
                template = None
            if template is None:
                edge_ir = None
                if values:
                    # should not happen. Putting this just in case.
                    raise PyRatesException("An empty edge IR was provided with additional values. "
                                           "No way to figure out where to apply those values.")

            else:
                edge_ir = template.apply(values=values)  # type: Optional[EdgeIR] # edge spec

            # check whether multiple source variables are defined
            try:
                # if source is a dictionary, pass on its values as source_var
                source_var = list(source.values())[0]  # type: dict
                source = list(source.keys())[0]
            except AttributeError:
                # no dictionary? only singular source definition present. go on as planned.
                edge_dict = dict(edge_ir=edge_ir,
                                 weight=weight,
                                 delay=delay,
                                 spread=spread)

            else:
                # oh source was indeed a dictionary. go pass source information as separate entry
                edge_dict = dict(edge_ir=edge_ir,
                                 weight=weight,
                                 delay=delay,
                                 spread=spread,
                                 source_var=source_var)

            # now add extra sources, if there are some
            if extra_sources:
                edge_dict["extra_sources"] = extra_sources

            edges.append((source, target,  # edge_unique_key,
                          edge_dict
                          ))

        return CircuitIR(label, self.circuits, nodes, edges, self.path)

    def _get_edge_templates(self, edges: List[Union[tuple, dict]]):
        """
        Reformat edges from [source, target, template_path, variables] to
        [source, target, template_object, variables]

        Parameters
        ----------
        edges

        Returns
        -------
        edges_with_templates
        """
        edges_with_templates = []
        for edge in edges:

            if isinstance(edge, dict):
                try:
                    source = edge["source"]
                    target = edge["target"]
                    template = edge["template"]
                    variables = edge["variables"]
                except KeyError as e:
                    raise TypeError(f"Wrong edge configuration. Unable to find key {e.args[0]}")

            else:
                source, target, template, variables = edge

            # "template" is EdgeTemplate, just use it
            # also just leave it as None, if so it be.
            if isinstance(template, EdgeTemplate) or template is None:
                pass

            # if not, try to load template path from yaml
            else:
                path = _complete_template_path(template, self.path)
                template = EdgeTemplate.from_yaml(path)

            edges_with_templates.append((source, target, template, variables))
        return edges_with_templates


# def to_circuit(template):
#     return template.apply()


def update_edges(base_edges: List[tuple], updates: List[Union[tuple, dict]]):
    """Add edges to list of edges. Removing or altering is currently not supported."""

    updated = deepcopy(base_edges)
    for edge in updates:
        if isinstance(edge, dict):
            if "variables" in edge:
                edge = [edge["source"], edge["target"], edge["template"], edge["variables"]]
            else:
                edge = [edge["source"], edge["target"], edge["template"]]
        elif not 3 <= len(edge) <= 4:
            raise PyRatesException("Wrong edge data type or not enough arguments")
        updated.append(edge)

    return updated


def update_dict(base_dict: dict, updates: dict):
    updated = deepcopy(base_dict)

    updated.update(updates)

    return updated
