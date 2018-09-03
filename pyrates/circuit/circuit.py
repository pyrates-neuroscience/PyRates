"""Basic neural mass model class plus derivations of it.

This module includes the base circuit class that manages the set-up and simulation of population networks. Additionally,
it features various sub-classes that act as circuit constructors and allow to build circuits based on different input
arguments. For more detailed descriptions, see the respective docstrings.

"""

# external packages
from typing import List, Dict
from networkx import MultiDiGraph

# pyrates internal imports
from pyrates.abc import AbstractBaseTemplate
from pyrates.node import NodeTemplate

from pyrates.utility.yaml_parser import TemplateLoader
from pyrates.operator.operator import OperatorTemplate

# meta infos
__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


class Circuit(MultiDiGraph):
    """Custom graph datastructure that represents a network of nodes and edges with associated equations
    and variables."""

    def __init__(self, label: str, nodes: dict=None,
                 coupling: Dict[str, OperatorTemplate]=None,
                 edges=None, template: str=None, **attr):
        """
        Parameters:
        -----------
        label
            string label, used if circuit is part of other circuits
        nodes
            dictionary of nodes of form {node_label:NodeTemplate instance}
        operators
            dictionary of coupling operators of form {op_label:OperatorTemplate}
        edges
            list of tuples (source:str, target:str, coupling_operator:str, coupling_values:dict)
        attr
            attribute keyword arguments that are passed to networkx graph constructor.
        """

        super().__init__(**attr)
        self.label = label
        self.template = template

        self.add_nodes_from(nodes, from_templates=True)

        coupling = {key: op.apply() for (key, op) in coupling.items()}

        self.add_edges_from(edges, coupling)

    def add_nodes_from(self, nodes: Dict[str, NodeTemplate], from_templates=False, **attr):
        """Add multiple nodes"""

        if from_templates:
            node_list = []
            for label, template in nodes.items():
                node_list.append((label, {"node": template.apply()}))
                # assuming unique labels
            super().add_nodes_from(node_list, **attr)
        else:  # default networkx behaviour
            super().add_nodes_from(nodes, **attr)

    def add_node(self, label: str, template: NodeTemplate=None, options: dict=None, **attr):
        """Add single node"""

        if template:
            super().add_node(label, node=template.apply(options), **attr)
        else:  # default networkx behaviour
            super().add_node(label, **attr)

    def add_edges_from(self, edges, coupling: dict=None, **attr):
        """Add multiple edges"""

        if coupling:
            edge_list = []
            for (source, target, key, values) in edges:
                try:
                    defaults = dict(coupling[key][1])
                    defaults.update(values)
                    values = defaults
                except TypeError as e:
                    if e.args[0].startswith("'NoneType'"):
                        pass
                    else:
                        raise e
                edge_list.append((source, target, {"coupling": coupling[key][0], "values": values}))
            super().add_edges_from(edge_list, **attr)
        else:  # default networkx behaviour
            super().add_edges_from(edges, **attr)

    # def __repr__(self):
    #     return f"Circuit '{self.label}'"


class CircuitTemplate(AbstractBaseTemplate):

    def __init__(self, name: str, path: str, description: str, label: str="circuit",
                 nodes: dict=None, coupling: dict=None, edges: List[tuple]=None,
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

    def apply(self, label: str=None, **options):
        """Create a Circuit graph instance based on the template, possibly applying predefined options"""

        if options:
            raise NotImplementedError("Unrecognised keyword argument. "
                                      "Note: The use of options is not implemented yet.")
        if not label:
            label = self.label

        return Circuit(label, self.nodes, self.coupling, self.edges, self.path)


class CircuitTemplateLoader(TemplateLoader):

    def __new__(cls, path):

        return super().__new__(cls, path, CircuitTemplate)

    @classmethod
    def update_template(cls, base, name: str, path: str, description: str = None,
                        label: str = None, nodes: dict=None, coupling: dict=None,
                        edges: List[tuple]=None, options: dict = None):
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
    def update_edges(base_edges: List[tuple], updates: List[tuple]):
        """Add edges to list of edges. Removing or altering is currently not supported."""

        updated = base_edges.copy()
        updated.extend(updates)

        return updated
