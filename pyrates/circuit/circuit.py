"""Basic neural mass model class plus derivations of it.

This module includes the base circuit class that manages the set-up and simulation of population networks. Additionally,
it features various sub-classes that act as circuit constructors and allow to build circuits based on different input
arguments. For more detailed descriptions, see the respective docstrings.

"""

# external packages
from typing import List, Dict, Any, Union
from networkx import MultiDiGraph, DiGraph, NetworkXNoCycle, find_cycle
from copy import deepcopy

# pyrates internal imports
from pyrates.abc import AbstractBaseTemplate
from pyrates.edge.edge import EdgeTemplate
from pyrates.node import NodeTemplate
from pyrates import PyRatesException
from pyrates.utility.yaml_parser import TemplateLoader
from pyrates.operator.operator import OperatorTemplate

# meta infos
__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


class Circuit(MultiDiGraph):
    """Custom graph datastructure that represents a network of nodes and edges with associated equations
    and variables."""

    label_cache = {}

    def __init__(self, label: str, nodes: dict = None,
                 edge_templates: Dict[str, EdgeTemplate] = None,
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
            list of tuples (source:str, target:str, coupling_operator:str, coupling_values:dict)
        attr
            attribute keyword arguments that are passed to networkx graph constructor.
        """

        super().__init__(**attr)
        self.label = label
        self.template = template

        label_map = self.add_nodes_from(nodes, from_templates=True)

        self.add_edges_from(edges, edge_templates, label_map)

    def add_nodes_and_edges(self, nodes: Dict[str, NodeTemplate],
                            edges: list,
                            edge_templates: Dict[str, EdgeTemplate]):

        label_map = {}
        edge_list = []
        for label, template in nodes.items():
            # ensure label uniqueness
            label_map[label] = self._get_unique_label(label)
            self.add_node(label_map[label], node=template.apply())

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
                node_list.append((label_map[label], {"node": template.apply()}))
                # assuming unique labels
            super().add_nodes_from(node_list, **attr)
        else:  # default networkx behaviour
            for old, new in label_map.items():
                nodes[new] = nodes.pop(old)
            super().add_nodes_from(nodes, **attr)
        return label_map

    def add_node(self, label: str, template: NodeTemplate = None, options: dict = None,
                 node: dict = None, **attr) -> str:
        """Add single node"""

        label = self._get_unique_label(label)

        if template:

            super().add_node(label, node=template.apply(options), **attr)
        else:  # default networkx behaviour
            super().add_node(label, node=node, **attr)

        return label

    def add_edges_from(self, edges, edge_templates: dict = None, label_map: dict = None, **attr):
        """ Add multiple edges.

        Parameters
        ----------
        edges
            list of edges, each of shape [source, target, edge_type_key, target_operator_name, **values]
        edge_templates
        label_map
        attr

        Returns
        -------

        """

        if edge_templates:
            edge_list = []
            for (source, target, template_key, target_operator, *values) in edges:

                edge_type = edge_templates[template_key].apply()  # type: dict # edge spec

                weight = values[0]
                try:
                    delay = values[1]
                except IndexError:
                    delay = 0

                # coupling_op = edge_type["operators"]
                # self._ensure_io_consistency(source, target, target_operator,
                #                             coupling_op["inputs"], coupling_op["output"])
                if label_map:
                    source = label_map[source]
                    target = label_map[target]

                # test, if variables at source and target exist and reference them properly
                source_var, target_var = self._get_edge_source_target_vars(source, target, target_operator,
                                                                           edge_type["operators"])

                edge_list.append((source, target,  # edge_unique_key,
                                  {"edge_type": edge_type,
                                   # "target_operator": target_operator,
                                   "weight": weight,
                                   "delay": delay,
                                   "source_var": source_var,
                                   "target_var": target_var}))
            super().add_edges_from(edge_list, **attr)
        else:  # default networkx behaviour
            super().add_edges_from(edges, **attr)

    def _get_unique_label(self, label: str) -> str:
        """

        Parameters
        ----------
        label
        max_count

        Returns
        -------
        unique_label
        """
        # define counter
        if label in self.label_cache:
            self.label_cache[label] += 1
        else:
            self.label_cache[label] = 0

        # set label
        unique_label = f"{label}:{self.label_cache[label]}"

        return unique_label

    # noinspection PyUnresolvedReferences
    def _get_edge_source_target_vars(self, source: str, target: str, target_operator: str,
                                     op_graph: DiGraph) -> (str, str):

        # 1: get reference for source variable
        ######################################

        # noinspection PyTypeChecker
        in_op = [op for op, in_degree in op_graph.in_degree if in_degree == 0]  # type: List[dict]

        # multiple input operations are possible, as long as they require the same singular input variable
        in_var = set()
        for op_key in in_op:
            for var in op_graph.nodes[op_key]["operator"]["inputs"]:
                in_var.add(var)

        if len(in_var) != 1:
            raise PyRatesException("Too many or too little input variables found. Exactly one input variable is "
                                   "required per edge.")
        else:
            source_var = in_var.pop()

        # collect all output variables
        source_out_vars = {}  # type: Dict[str, list]
        source_op_graph = self.nodes[source]["node"]["operators"]
        for op_key, data in source_op_graph.nodes(data=True):
            out_var = data["operator"]["output"]
            if out_var not in source_out_vars:
                source_out_vars[out_var] = []

            source_out_vars[out_var].append(op_key)

        # check for uniqueness of output variable
        if len(source_out_vars[source_var]) > 1:
            raise PyRatesException(f"Too many operators found for source variable `{source_var}`. The source variable "
                                   f"of an edge needs to be a unique output in the source node.")

        # assign output variable and operator as source_var
        source_out_op = source_out_vars[source_var][0]
        source_path = f"{source_out_op}/{source_var}"
        # note source variable and operator as input in input operators
        for op in in_op:
            op_graph.nodes[op]["operator"]["inputs"][source_var]["source"].append(source_path)
            # this should be transmitted back as a side effect, since op_graph references the actual DiGraph instance

        # 2: get reference for target variable
        ######################################
        # simplification: assume target operator is defined by name:0

        # try to find single output variable
        # noinspection PyTypeChecker
        out_op = [op for op, out_degree in op_graph.out_degree if out_degree == 0]  # type: List[dict]

        # only one single output operator allowed
        if len(out_op) != 1:
            raise PyRatesException("Too many or too little output operators found. Exactly one output operator and "
                                   "associated output variable is required per edge.")

        target_var = op_graph.nodes[out_op[0]]["operator"]["output"]
        target_op_graph = self.nodes[target]["node"]["operators"]
        target_op_name = f"{target_operator.split('.')[-1]}:0"
        target_op = target_op_graph.nodes[target_op_name]["operator"]

        if target_var not in target_op["inputs"]:
            raise PyRatesException(f"Could not find target variable {target_var} in target operator {target_op_name}  "
                                   f"of node {target}.")

        target_path = f"{target_op_name}/{target_var}"

        return source_path, target_path

    # def __repr__(self):
    #     return f"Circuit '{self.label}'"

    # def add_edge(self, *args, **kwargs):
    #
    #     raise NotImplementedError("Adding single edges is not implemented.")

    # def _ensure_io_consistency(self, source: str, target: str, target_operator: str,
    #                            input_var: List[str], output_var: str):
    #     """Test, if inputs and output of coupling operator are present as output/input of source/target,
    #     respectively.
    #
    #     Parameters
    #     ----------
    #     source
    #     target
    #     target_operator
    #     input_var
    #     output_var
    #
    #     Returns
    #     -------
    #     None
    #     """
    #     if len(input_var) > 1:
    #         raise ValueError("Too many input variables defined in `input_var`. "
    #                          "Edges only accept one input variable.")
    #     # step 1: find source output variable
    #     assert self.nodes[source]["node"]["output"] == input_var[0]
    #
    #     # step 2: find target input variable
    #     target_node = self.nodes[target]["node"]
    #     # hard-coded variant of operator instance key with no additional options applied
    #     target_op = target_node["operators"][(target_operator, None)]["operator"]
    #     assert output_var in target_op["inputs"]  # could be replaced by target_node["inputs"],
    #     # if it contained info about operators

    def network_def(self):
        """A bit of a workaround to connect interfaces of frontend and backend.
        TODO: clean up/simplify interface"""
        # import re

        network_def = MultiDiGraph()

        edge_list = []
        node_dict = {}

        # reorganize node to conform with backend API
        #############################################
        for node_key, data in self.nodes(data=True):
            # reformat all node internals into operators + operator_args
            node_dict[node_key] = {}  # type: Dict[str, Union[dict, list]]
            node_dict[node_key] = dict(self._nd_reformat_operators(data["node"]))
            op_order = self._nd_get_operator_order(data["node"]["operators"])  # type: list
            node_dict[node_key]["operator_order"] = op_order

        # reorganize edge to conform with backend API
        #############################################
        for source, target, data in self.edges(data=True):
            # reformat all edge internals into operators + operator_args
            op_data = self._nd_reformat_operators(data["edge_type"])
            op_order = self._nd_get_operator_order(data["edge_type"]["operators"])

            # move edge operators to nodes
            # using dictionary update method, assuming no conflicts in operator names
            # this fails, if multiple edges on one source node use the same coupling operator
            # todo: implement conflict management for multiple edges with same operators per source node
            node_dict[source]["operators"].update(op_data["operators"])
            node_dict[source]["operator_args"].update(op_data["operator_args"])

            for op in op_order:
                if op not in node_dict[source]["operator_order"]:
                    node_dict[source]["operator_order"].append(op)

            # simplify edges and save into edge_list
            # find single output operator to save new reference to source variable after reformatting
            op_graph = data["edge_type"]["operators"]
            out_op = [op for op, out_degree in op_graph.out_degree if out_degree == 0]
            out_var = op_graph.nodes[out_op[0]]["operator"]["output"]
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
    def _nd_reformat_operators(data):
        operator_args = dict()
        operators = dict()

        for op_key, op_dict in data["operators"].nodes(data=True):
            op_cp = deepcopy(op_dict["operator"])  # duplicate operator info

            var_dict = op_cp.pop("variables")

            for var_key, var_props in var_dict.items():
                var_prop_cp = deepcopy(var_props)  # duplicate variable properties

                if var_key in op_dict["values"]:  # workaround to get values back into operator_args
                    var_prop_cp["value"] = op_dict["values"][var_key]

                var_prop_cp["vtype"] = var_prop_cp.pop("variable_type")
                var_prop_cp["dtype"] = var_prop_cp.pop("data_type")
                var_prop_cp["shape"] = ()  # default to scalars for now
                var_prop_cp.pop("unit", None)
                var_prop_cp.pop("description", None)
                var_prop_cp.pop("name", None)
                # var_prop_cp["name"] = f"{new_op_key}/{var_key}"  # has been trown out
                operator_args[f"{op_key}/{var_key}"] = deepcopy(var_prop_cp)

            op_cp["equations"] = op_cp.pop("equation")
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
        graph = op_graph.copy()
        while graph.nodes:
            # noinspection PyTypeChecker
            primary_nodes = [node for node, in_degree in graph.in_degree if in_degree == 0]
            op_order.extend(primary_nodes)
            graph.remove_nodes_from(primary_nodes)

        return op_order


class CircuitTemplate(AbstractBaseTemplate):

    def __init__(self, name: str, path: str, description: str, label: str = "circuit",
                 nodes: dict = None, edge_templates: dict = None, edges: List[tuple] = None,
                 options: dict = None):

        super().__init__(name, path, description)

        self.nodes = {}
        if nodes:
            for key, path in nodes.items():
                if isinstance(path, str):
                    self.nodes[key] = NodeTemplate.from_yaml(path)

        self.edge_templates = {}
        if edge_templates:
            for key, path in edge_templates.items():
                if isinstance(path, str):
                    self.edge_templates[key] = EdgeTemplate.from_yaml(path)

        if edges:
            self.edges = edges
        else:
            self.edges = []

        self.label = label

        if options:
            self.options = options
        else:
            self.options = {}

    def apply(self, label: str = None, **options):
        """Create a Circuit graph instance based on the template, possibly applying predefined options"""

        if options:
            raise NotImplementedError("Unrecognised keyword argument. "
                                      "Note: The use of options is not implemented yet.")
        if not label:
            label = self.label

        return Circuit(label, self.nodes, self.edge_templates, self.edges, self.path)


class CircuitTemplateLoader(TemplateLoader):

    def __new__(cls, path):

        return super().__new__(cls, path, CircuitTemplate)

    @classmethod
    def update_template(cls, base, name: str, path: str, description: str = None,
                        label: str = None, nodes: dict = None,
                        edge_templates: Dict[str, OperatorTemplate] = None,
                        edges: List[tuple] = None, options: dict = None):
        """Update all entries of the circuit template in their respective ways."""

        if not description:
            description = base.__doc__

        if not label:
            label = base.label

        if nodes:
            nodes = cls.update_nodes(base.nodes, nodes)
        else:
            nodes = base.nodes

        if edge_templates:
            edge_templates = cls.update_nodes(base.edge_templates, edge_templates)
            # note: edge templates have the same properties as node templates, could also call them graph entities
        else:
            edge_templates = base.edge_templates

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
                               label=label, nodes=nodes, edge_templates=edge_templates,
                               edges=edges, options=options)

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
    def update_edges(base_edges: List[tuple], updates: List[tuple]):
        """Add edges to list of edges. Removing or altering is currently not supported."""

        updated = base_edges.copy()
        updated.extend(updates)

        return updated
