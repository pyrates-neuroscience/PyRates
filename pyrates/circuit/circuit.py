"""Basic neural mass model class plus derivations of it.

This module includes the base circuit class that manages the set-up and simulation of population networks. Additionally,
it features various sub-classes that act as circuit constructors and allow to build circuits based on different input
arguments. For more detailed descriptions, see the respective docstrings.

"""

# external packages
from typing import List

# pyrates internal imports
from pyrates.abc import AbstractBaseTemplate
from pyrates.node import NodeTemplate

# meta infos
from pyrates.utility.yaml_parser import TemplateLoader
from pyrates.operator.operator import OperatorTemplate

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"
















class CircuitTemplate(AbstractBaseTemplate):

    def __init__(self, name: str, path: str, description: str, label: str,
                 nodes: dict=None, coupling: dict=None, edges: List[list]=None,
                 options: dict=None):

        super().__init__(name, path, description)

        self.nodes = {}
        if nodes:
            for key, path in nodes.items():
                if isinstance(path, str):
                    self.nodes[key] = NodeTemplate.from_yaml(path)

        self.coupling = {}
        if coupling:
            for key, path in coupling.items():
                if isinstance(path, str):
                    self.coupling[key] = OperatorTemplate.from_yaml(path)

        if edges:
            self.edges = edges
        else:
            self.edges = []

        self.label = label

        if options:
            self.options = options
        else:
            self.options = {}


class CircuitTemplateLoader(TemplateLoader):

    def __new__(cls, path):

        return super().__new__(cls, path, CircuitTemplate)

    @classmethod
    def update_template(cls, base, name: str, path: str, description: str = None,
                        label: str = None, nodes: dict=None, coupling: dict=None,
                        edges: List[list]=None, options: dict = None):
        """Update all entries of the circuit template in their respective ways."""

        if not description:
            description = base.__doc__

        if not label:
            label = base.label

        if nodes:
            nodes = cls.update_nodes(base.nodes, nodes)
        else:
            nodes = base.nodes

        if coupling:
            coupling = cls.update_coupling(base.coupling, coupling)
        else:
            coupling = base.coupling

        if edges:
            edges = cls.update_edges(base.edges, edges)
        else:
            edges = base.edges

        if options:
            options = cls.update_options(base.options, options)
        else:
            # copy old options dict
            options = base.options

        return CircuitTemplate(name=name, path=path, description=description,
                               label=label, nodes=nodes, coupling=coupling,
                               edges=edges, options=options)

    @staticmethod
    def update_nodes(base_nodes: dict, updates: dict):

        updated = base_nodes.copy()

        # for key, value in updates.items():
        #     if isinstance(value, str):
        #         updated[key] = OperatorTemplate.from_yaml(value)
        #     else:
        #         raise TypeError("Node specifier must a string.")
        #         # could be a dict, but not implemented
        updated.update(updates)

        return updated

    @staticmethod
    def update_coupling(base_coupling: dict, updates: dict):

        updated = base_coupling.copy()

        updated.update(updates)

        return updated

    @staticmethod
    def update_edges(base_edges: list, updates: list):
        """Add edges to list of edges. Removing or altering is currently not supported."""

        updated = base_edges.copy()
        updated.extend(updates)

        return updated
