"""Basic neural mass model class plus derivations of it.

This module includes the base circuit class that manages the set-up and simulation of population networks. Additionally,
it features various sub-classes that act as circuit constructors and allow to build circuits based on different input
arguments. For more detailed descriptions, see the respective docstrings.

"""

# external packages
from typing import List, Dict, Any, Union
from networkx import MultiDiGraph
from copy import deepcopy

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

    def __init__(self, label: str, nodes: dict = None,
                 coupling: Dict[str, OperatorTemplate] = None,
                 edges=None, template: str = None, **attr):
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
        """ Add multiple nodes to circuit. Allows networkx-style adding if from_templates is set to False.

        Parameters
        ----------
        nodes
            Dictionary with node label as key. The item is a template to derive the node instance from.
        from_templates
            Defaults to `False` which invokes `add_nodes_from` of `networkx.MultiDiGraph`. Use this option, if you
            want to add node instances instead of templates. If `True`, nodes are assumed to be defined by templates and
            these templates are applied to create node instances.
        attr
            additional keyword attributes that can be added to the node data. (default `networkx` syntax.)

        Returns
        -------
        None
        """

        if from_templates:
            node_list = []
            for label, template in nodes.items():
                node_list.append((label, {"node": template.apply()}))
                # assuming unique labels
            super().add_nodes_from(node_list, **attr)
        else:  # default networkx behaviour
            super().add_nodes_from(nodes, **attr)

    def add_node(self, label: str, template: NodeTemplate = None, options: dict = None, **attr):
        """Add single node"""

        if template:
            super().add_node(label, node=template.apply(options), **attr)
        else:  # default networkx behaviour
            super().add_node(label, **attr)

    def add_edges_from(self, edges, coupling: dict = None, **attr):
        """ Add multiple edges.

        Parameters
        ----------
        edges
            list of edges, each of shape [source, target, coupling_operator_key, target_operator_name, **values]
        coupling
        attr

        Returns
        -------

        """

        if coupling:
            edge_list = []
            for (source, target, co_key, target_operator, *values) in edges:
                coupling_op = coupling[co_key][0]  # type: dict  # operator spec

                defaults = dict(coupling[co_key][1])  # default values of variables
                # try:
                defaults.update(*values)
                # except TypeError as e:
                #     if e.args[0].startswith("'NoneType'"):
                #         pass  # just propagate "None" as values
                #     else:
                #         raise e
                # else:
                values = defaults
                # test, if variables at source and target exist
                self._ensure_io_consistency(source, target, target_operator,
                                            coupling_op["inputs"], coupling_op["output"])

                # TODO: allow multiple operator definitions per edge
                edge_list.append((source, target, {"operators": {co_key: coupling[co_key][0]},
                                                   "values": values,
                                                   "target_operator": target_operator}))
            super().add_edges_from(edge_list, **attr)
        else:  # default networkx behaviour
            super().add_edges_from(edges, **attr)

    # def __repr__(self):
    #     return f"Circuit '{self.label}'"

    def _ensure_io_consistency(self, source: str, target: str, target_operator: str,
                               input_var: List[str], output_var: str):
        """Test, if inputs and output of coupling operator are present as output/input of source/target,
        respectively.

        Parameters
        ----------
        source
        target
        target_operator
        input_var
        output_var

        Returns
        -------
        None
        """
        if len(input_var) > 1:
            raise ValueError("Too many input variables defined in `input_var`. "
                             "Edges only accept one input variable.")
        # step 1: find source output variable
        assert self.nodes[source]["node"]["output"] == input_var[0]

        # step 2: find target input variable
        target_node = self.nodes[target]["node"]
        # hard-coded variant of operator instance key with no additional options applied
        target_op = target_node["operators"][(target_operator, None)]["operator"]
        assert output_var in target_op["inputs"]  # could be replaced by target_node["inputs"],
        # if it contained info about operators

    def network_def(self):
        """A bit of a workaround to connect interfaces of frontend and backend.
        TODO: clean up/simplify interface"""
        # import re

        network_def = MultiDiGraph()

        # reorganize node and operator data to conform with backend API
        ###############################################################

        op_key_map = {}
        for node_key, node in list(self.nodes(data=True)):
            node_cp = deepcopy(node["node"])  # duplicate node info
            node_cp = self._nd_reformat_node_operators(node_cp, op_key_map)
            node_cp.pop("template", None)
            network_def.add_node(f"{node_key}:0", **node_cp)

        # add edges and reformat them a little
        ######################################
        # edge_counter = 0

        for source, target, edge_data in list(self.edges(data=True)):
            edge_cp = deepcopy(edge_data)
            operators, operator_args = self._nd_reformat_edge_operators(edge_cp, op_key_map)

            network_def.add_edge(f"{source}:0", f"{target}:0",
                                 # key=edge_counter,
                                 operators=operators,
                                 operator_args=operator_args)
            # edge_counter += 1

        return network_def  # return MultiDiGraph as needed by Network class

    @staticmethod
    def _nd_create_short_unique_key(key_map, old_key):
        """ Workaround method to create new shorter unique keys for unneccessarily long keys

        Parameters
        ----------
        key_map
        old_key

        Returns
        -------
        new_key
        """
        if old_key in key_map:  # fetch already known key from map
            new_key = key_map[old_key]
        else:  # create new unique short key
            base_name = old_key[0].split(".")[-1]

            # ensure uniqueness
            for counter in range(1000):  # max 1000 iterations
                new_key = f"{base_name}:{counter}"
                if new_key in key_map.values():
                    continue  # increment counter
                else:  # use current new op_key
                    key_map[old_key] = new_key
                    break
            else:
                raise RecursionError("Maximum number of iterations (=1000) reached. This is a feature that should be "
                                     "removed, once the IR interface overhaul is completed.")

        return new_key

    def _nd_reformat_node_operators(self, node, op_key_map):
        operator_args = dict()
        inputs = {}  # type: Dict[str, list]

        for op_key, op_dict in node["operators"].items():
            op_cp = deepcopy(op_dict["operator"])  # duplicate operator info
            # op_cp["original_key"] = op_key  # copy original key to be able to change it later

            # create shortened unique key
            new_op_key = self._nd_create_short_unique_key(op_key_map, op_key)

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
                operator_args[f"{new_op_key}/{var_key}"] = deepcopy(var_prop_cp)

                vname = var_key.split("/")[-1]
                # create mapping between node-wide input variable names and operator-level var names
                if vname in node["inputs"]:
                    if vname in inputs:
                        inputs[vname].append(f"{op_key}/{var_key}")
                    else:
                        inputs[vname] = [f"{op_key}/{var_key}"]
                node.pop("output", None)  # remove output specifier from node
                # elif vname in node_cp["output"]:
                #     node_cp["output"] = {vname: f"{op_key}/{var_key}"}
            op_cp["equations"] = op_cp.pop("equation")
            node["operators"][op_key] = op_cp

        # rename all operators with short names
        for old_key, new_key in op_key_map.items():
            if old_key in node["operators"]:
                node["operators"][new_key] = node["operators"].pop(old_key)

        node["operator_args"] = operator_args
        node["inputs"] = inputs

        return node

    @staticmethod
    def _nd_reformat_edge_operators(edge_data, op_key_map):
        """

        Parameters
        ----------
        edge_data
        op_key_map

        Returns
        -------
        operators
        operator_args
        """
        operator_args = {}
        edge_data = deepcopy(edge_data)
        operators = edge_data["operators"]
        for op_key, op in operators.items():
            variables = op.pop("variables")
            target_op = edge_data["target_operator"]

            # transfer edge-specific values back to variable info
            values = edge_data["values"]

            for var_key, value in values.items():
                variables[var_key]["value"] = value

            for var_key, var_props in variables.items():
                var_prop_cp = var_props
                var_prop_cp["vtype"] = var_prop_cp.pop("variable_type")
                var_prop_cp["dtype"] = var_prop_cp.pop("data_type")
                if var_key in op["inputs"]:
                    var_prop_cp["vtype"] = "source_var"
                    base_name = target_op.split(".")[-1]
                    var_prop_cp["name"] = f"{base_name}:0/{var_key}"
                    var_prop_cp.pop("dtype", None)
                elif var_key == op["output"]:
                    var_prop_cp["vtype"] = "target_var"
                    base_name = target_op.split(".")[-1]
                    # assume unique instances, no options applied
                    var_prop_cp["name"] = f"{base_name}:0/{var_key}"
                    var_prop_cp.pop("dtype", None)
                else:
                    var_prop_cp["shape"] = ()
                    var_prop_cp.pop("name", None)

                var_prop_cp.pop("unit", None)
                var_prop_cp.pop("description", None)
                operator_args[f"{op_key}/{var_key}"] = deepcopy(var_prop_cp)

        return operators, operator_args


class CircuitTemplate(AbstractBaseTemplate):

    def __init__(self, name: str, path: str, description: str, label: str = "circuit",
                 nodes: dict = None, coupling: dict = None, edges: List[tuple] = None,
                 options: dict = None):

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

    def apply(self, label: str = None, **options):
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
                        label: str = None, nodes: dict = None, coupling: dict = None,
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
