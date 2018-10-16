"""Basic neural mass model class plus derivations of it.

This module includes the base circuit class that manages the set-up and simulation of population networks. Additionally,
it features various sub-classes that act as circuit constructors and allow to build circuits based on different input
arguments. For more detailed descriptions, see the respective docstrings.

"""

# external packages
from typing import List, Dict, Any, Union, Tuple
from networkx import MultiDiGraph, DiGraph, NetworkXNoCycle, find_cycle
from copy import deepcopy

# pyrates internal imports
from pyrates.frontend.abc import AbstractBaseTemplate, AbstractBaseIR
from pyrates.frontend.edge import EdgeTemplate, EdgeIR
from pyrates.frontend.node import NodeTemplate
from pyrates import PyRatesException
from pyrates.frontend.yaml_parser import TemplateLoader

# from pyrates.frontend.operator import OperatorTemplate

# meta infos
__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


class CircuitIR(AbstractBaseIR):
    """Custom graph datastructure that represents a network of nodes and edges with associated equations
    and variables."""

    def __init__(self, label: str, nodes: dict = None,
                 edges: list = None, template: str = None, **attr):
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
            list of tuples (source:str, target:str, dict(template=edge_template, variables=coupling_values))
        attr
            attribute keyword arguments that are passed to networkx graph constructor.
        """

        super().__init__(**attr)
        self.label = label
        self.label_counter = {}
        self.template = template

        self.graph = MultiDiGraph()

        label_map = self.add_nodes_from(nodes, from_templates=True)

        self.add_edges_from(edges, label_map)

    # def add_nodes_and_edges(self, nodes: Dict[str, NodeTemplate],
    #                         edges: list,):
    #
    #     label_map = {}
    #     edge_list = []
    #     for label, template in nodes.items():
    #         # ensure label uniqueness
    #         label_map[label] = self._get_unique_label(label)
    #         self.add_node(label_map[label], node=template.apply())

    def add_nodes_from(self, nodes: Union[Dict[str, NodeTemplate], Dict[str, dict]],
                       from_templates=True, **attr) -> dict:
        """ Add multiple nodes to circuit. Allows networkx-style adding if from_templates is set to False.

        Parameters
        ----------
        nodes
            Dictionary with node label as key. The item is a template to derive the node instance from.
        from_templates
            Defaults to `True`. Setting it to `False` invokes `add_nodes_from` of `networkx.MultiDiGraph`. Use this
            option, if you want to add node instances instead of templates. If `True`, nodes are assumed to be defined
            by templates and these templates are applied to create node instances.
        attr
            additional keyword attributes that can be added to the node data. (default `networkx` syntax.)

        Returns
        -------
        label_map
        """
        label_map = {}
        for label in nodes:
            label_map[label] = self._get_unique_label(label)

        if from_templates:
            node_list = []
            for label, template in nodes.items():
                # ensure label uniqueness
                self.graph.add_node(label_map[label], node=template.apply())
        else:  # default networkx behaviour
            for old, new in label_map.items():
                nodes[new] = nodes.pop(old)
            self.graph.add_nodes_from(nodes, **attr)
        return label_map

    def add_node(self, label: str, template: NodeTemplate = None,
                 node: dict = None, **attr) -> str:
        """Add single node"""

        label = self._get_unique_label(label)

        if template:

            self.graph.add_node(label, node=template.apply(), **attr)
        else:  # default networkx behaviour
            self.graph.add_node(label, node=node, **attr)

        return label

    def add_edges_from(self, edges, label_map: dict = None, **attr):
        """ Add multiple edges.

        Parameters
        ----------
        edges
            list of edges, each of shape [source/op/var, target/op/var, edge_template, variables]
        label_map
        attr

        Returns
        -------
        """

        edge_list = []
        for (source, target, template, values) in edges:
            # get edge template and instantiate it
            edge_ir = template.apply()  # type: EdgeIR # edge spec

            # get weight
            weight = values.pop("weight", 1)
            # get delay
            delay = values.pop("delay", 0)

            # ToDo: Implement source/op/var syntax
            # take apart source and target strings
            # for circuit from circuit:
            # for ... in circuits
            # for ... in nodes...

            # test, if variables at source and target exist and reference them properly
            source, target = self._identify_sources_targets(source, target, edge_ir, label_map)

            edge_list.append((source[0], target[0],  # edge_unique_key,
                              {"edge_ir": edge_ir,
                               # "target_operator": target_operator,
                               "weight": weight,
                               "delay": delay,
                               "source_var": "/".join(source[-2:]),
                               "target_var": "/".join(target[-2:])
                               }))
        self.graph.add_edges_from(edge_list, **attr)

    def _get_unique_label(self, label: str) -> str:
        """

        Parameters
        ----------
        label

        Returns
        -------
        unique_label
        """
        # define counter
        if label in self.label_counter:
            self.label_counter[label] += 1
        else:
            self.label_counter[label] = 0

        # set label
        unique_label = f"{label}.{self.label_counter[label]}"

        return unique_label

    # noinspection PyUnresolvedReferences
    def _identify_sources_targets(self, source: str, target: str,
                                  edge_ir: EdgeIR, label_map: dict = None):

        input_var = edge_ir.input
        output_var = edge_ir.output
        # separate source and target specifiers
        *source_path, source_op, source_var = source.split("/")
        *target_path, target_op, target_var = target.split("/")
        source_node = label_map[source_path[-1]]
        target_node = label_map[target_path[-1]]

        # ignore circuits for now:
        source_path = "/".join((source_node, source_op, source_var))
        target_path = "/".join((target_node, target_op, target_var))
        # check if path is valid
        try:
            _ = self[source_path]
            _ = self[target_path]
        except KeyError:
            raise PyRatesException(f"Could not find either `{source_path}` or `{target_path}` in network graph.")

        source = (source_node, source_op, source_var)
        target = (target_node, target_op, target_var)
        return source, target

    def _getter(self, key):
        return self.graph.nodes[key]["node"]

    @property
    def nodes(self):
        return self.graph.nodes

    @property
    def edges(self):
        return self.graph.edges

    @classmethod
    def from_template(cls, template):

        return cls(template.label, template.nodes, template.edges, template.path)

    def network_def(self):
        return BackendIRFormatter.network_def(self)

class CircuitTemplate(AbstractBaseTemplate):

    def __init__(self, name: str, path: str, description: str, label: str = "circuit",
                 nodes: dict = None, edges: List[tuple] = None):

        super().__init__(name, path, description)

        self.nodes = {}
        if nodes:
            for key, path in nodes.items():
                if isinstance(path, str):
                    path = self._format_path(path)
                    self.nodes[key] = NodeTemplate.from_yaml(path)

        # self.edge_templates = {}
        # if edge_templates:
        #     for key, path in edge_templates.items():
        #         if isinstance(path, str):
        #             path = self._format_path(path)
        #             self.edge_templates[key] = EdgeTemplate.from_yaml(path)

        if edges:
            self.edges = self._get_edge_templates(edges)
        else:
            self.edges = []

        self.label = label

    def apply(self, label: str = None):
        """Create a Circuit graph instance based on the template"""

        if not label:
            label = self.label

        return CircuitIR(label, self.nodes, self.edges, self.path)

    def _get_edge_templates(self, edges: List[tuple]):
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
            path = edge[2]
            path = self._format_path(path)
            temp = EdgeTemplate.from_yaml(path)
            edges_with_templates.append((*edge[0:2], temp, *edge[3:]))
        return edges_with_templates


class CircuitTemplateLoader(TemplateLoader):

    def __new__(cls, path):

        return super().__new__(cls, path, CircuitTemplate)

    @classmethod
    def update_template(cls, base, name: str, path: str, description: str = None,
                        label: str = None, nodes: dict = None,
                        edges: List[tuple] = None):
        """Update all entries of the circuit template in their respective ways."""

        if not description:
            description = base.__doc__

        if not label:
            label = base.label

        if nodes:
            nodes = cls.update_nodes(base.nodes, nodes)
        else:
            nodes = base.nodes

        # removed.
        # if edge_templates:
        #     edge_templates = cls.update_nodes(base.edge_templates, edge_templates)
        #     # note: edge templates have the same properties as node templates, could also call them graph entities
        # else:
        #     edge_templates = base.edge_templates

        if edges:
            edges = cls.update_edges(base.edges, edges)
        else:
            edges = base.edges

        return CircuitTemplate(name=name, path=path, description=description,
                               label=label, nodes=nodes,
                               edges=edges)

    @staticmethod
    def update_nodes(base_nodes: dict, updates: dict):

        updated = deepcopy(base_nodes)

        updated.update(updates)

        return updated

    # @staticmethod
    # def update_edge_templates(base_edge_templates: dict, updates: dict):
    #
    #     updated = deepcopy(base_edge_templates)
    #
    #     updated.update(updates)
    #
    #     return updated

    @staticmethod
    def update_edges(base_edges: List[tuple], updates: List[Union[tuple, dict]]):
        """Add edges to list of edges. Removing or altering is currently not supported."""

        updated = base_edges.copy()
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


class BackendIRFormatter:

    @classmethod
    def network_def(cls, circuit: CircuitIR):
        """A bit of a workaround to connect interfaces of frontend and backend.
        TODO: clean up/simplify interface"""
        # import re

        network_def = MultiDiGraph()

        edge_list = []
        node_dict = {}

        # reorganize node to conform with backend API
        #############################################
        for node_key, data in circuit.graph.nodes(data=True):
            node = data["node"]
            # reformat all node internals into operators + operator_args
            node_dict[node_key] = {}  # type: Dict[str, Union[list, dict]]
            node_dict[node_key] = dict(cls._nd_reformat_operators(node.op_graph))
            op_order = cls._nd_get_operator_order(node.op_graph)  # type: list
            # noinspection PyTypeChecker
            node_dict[node_key]["operator_order"] = op_order

        # reorganize edge to conform with backend API
        #############################################
        for source, target, data in circuit.graph.edges(data=True):
            edge = data["edge_ir"]
            # reformat all edge internals into operators + operator_args
            op_data = cls._nd_reformat_operators(edge.op_graph)
            op_order = cls._nd_get_operator_order(edge.op_graph)  # type: List[str]

            # move edge operators to nodes
            # using dictionary update method, assuming no conflicts in operator names
            # this fails, if multiple edges on one source node use the same coupling operator
            # todo: implement conflict management for multiple edges with same operators per source node
            node_dict[source]["operators"].update(op_data["operators"])
            node_dict[source]["operator_args"].update(op_data["operator_args"])

            for op in op_order:
                if op not in node_dict[source]["operator_order"]:
                    # noinspection PyUnresolvedReferences
                    node_dict[source]["operator_order"].append(op)

            # simplify edges and save into edge_list
            # find single output operator to save new reference to source variable after reformatting
            op_graph = edge.op_graph
            out_op = [op for op, out_degree in op_graph.out_degree if out_degree == 0]
            out_var = edge[out_op[0]].output
            source_var = f"{out_op[0]}/{out_var}"

            edge_list.append((source, target, {"source_var": source_var,
                                               "target_var": data["target_var"],
                                               "weight": data["weight"],
                                               "delay": data["delay"]}))

        # network_def.add_nodes_from(node_dict)
        for key, node in node_dict.items():
            network_def.add_node(key, **node)
        network_def.add_edges_from(edge_list)

        return network_def  # return MultiDiGraph as needed by Network class

    @staticmethod
    def _nd_reformat_operators(op_graph: DiGraph):
        operator_args = dict()
        operators = dict()

        for op_key, op_dict in op_graph.nodes(data=True):
            op_cp = deepcopy(op_dict)  # duplicate operator info
            var_dict = op_cp.pop("variables")

            for var_key, var_props in var_dict.items():
                var_prop_cp = deepcopy(var_props)  # duplicate variable properties

                # if var_key in op_dict["variables"]:  # workaround to get values back into operator_args
                #     var_prop_cp["value"] = deepcopy(op_dict["values"][var_key])

                var_prop_cp["shape"] = ()  # default to scalars for now
                var_prop_cp.pop("unit", None)
                var_prop_cp.pop("description", None)
                var_prop_cp.pop("name", None)
                # var_prop_cp["name"] = f"{new_op_key}/{var_key}"  # has been thrown out
                operator_args[f"{op_key}/{var_key}"] = deepcopy(var_prop_cp)

            op_cp["equations"] = op_cp["operator"].equations
            op_cp["inputs"] = op_cp["operator"].inputs
            op_cp["output"] = op_cp["operator"].output
            # op_cp.pop("values", None)
            op_cp.pop("operator", None)
            operators[op_key] = op_cp

        reformatted = dict(operator_args=operator_args,
                           operators=operators,
                           inputs={})
        return reformatted

    @staticmethod
    def _nd_get_operator_order(op_graph: DiGraph) -> list:
        """

        Parameters
        ----------
        op_graph

        Returns
        -------
        op_order
        """
        # check, if cycles are present in operator graph (which would be problematic
        try:
            find_cycle(op_graph)
        except NetworkXNoCycle:
            pass
        else:
            raise PyRatesException("Found cyclic operator graph. Cycles are not allowed for operators within one node.")

        op_order = []
        graph = op_graph.copy()  # type: DiGraph
        while graph.nodes:
            # noinspection PyTypeChecker
            primary_nodes = [node for node, in_degree in graph.in_degree if in_degree == 0]
            op_order.extend(primary_nodes)
            graph.remove_nodes_from(primary_nodes)

        return op_order
