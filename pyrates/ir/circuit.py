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

# external imports
from typing import Union, Dict, Iterator, Optional, List, Tuple
from warnings import filterwarnings
from copy import deepcopy
from networkx import MultiDiGraph, subgraph, DiGraph
from pandas import DataFrame
import numpy as np

# pyrates-internal imports
from pyrates import PyRatesException
from pyrates.ir.node import NodeIR, VectorizedNodeIR
from pyrates.ir.edge import EdgeIR
from pyrates.ir.abc import AbstractBaseIR
from pyrates.backend.parser import parse_equations, is_diff_eq, replace

__author__ = "Daniel Rose, Richard Gast"
__status__ = "Development"


class CircuitIR(AbstractBaseIR):
    """Custom graph data structure that represents a backend of nodes and edges with associated equations
    and variables."""

    # _node_label_grammar = Word(alphanums+"_") + Suppress(".") + Word(nums)
    __slots__ = ["label", "label_map", "graph", "sub_circuits", "_reference_map", "_buffered",
                 "_first_run", "_vectorized", "_compiled", "_backend", "step_size", "solver", "_edge_idx_counter"]

    def __init__(self, label: str = "circuit", circuits: dict = None, nodes: Dict[str, NodeIR] = None,
                 edges: list = None, template: str = None):
        """
        Parameters:
        -----------
        label
            String label, could be used as fallback when subcircuiting this circuit. Currently not used, though.
        circuits
            Dictionary of sub-circuits to be added. Keys are string labels for circuits that serve as namespaces for the
            subcircuits. Items must be `CircuitIR` instances.
        nodes
            Dictionary of nodes of form {node_label: `NodeIR` instance}.
        edges
            List of tuples (source:str, target:str, edge_dict). `edge_dict` should contain the key "edge_ir" with an
            `EdgeIR` instance as item and optionally entries for "weight" and "delay". `source` and `target` should be
            formatted as "node/op/var" (with optionally prepended circuits).
        template
            optional string reference to path to template that this circuit was loaded from. Leave empty, if no template
            was used.
        """

        super().__init__(template)
        self.label = label
        self.label_map = {}

        self.graph = MultiDiGraph()
        self.sub_circuits = set()

        self._reference_map = {}

        if circuits:
            for key, temp in circuits.items():
                self.add_circuit(key, temp)

        if nodes:
            self.add_nodes_from(nodes)

        if edges:
            self.add_edges_from(edges)

        self._first_run = True
        self._vectorized = False
        self._compiled = False
        self._backend = None
        self._buffered = False
        self.solver = None
        self.step_size = None
        self._edge_idx_counter = 0

    def _collect_references(self, edge_or_node):
        """Collect all references of nodes or edges to unique operator_graph instances in local `_reference_map`.
        References are collected as a list, because nodes and edges are (currently) not hashable."""

        try:
            op_graph = edge_or_node.op_graph
        except AttributeError:
            op_graph = None
        try:
            self._reference_map[op_graph].append(edge_or_node)
        except KeyError:
            self._reference_map[op_graph] = [edge_or_node]

        # for key, data in edge_or_node:
        #     op = data["operator"]
        #     try:
        #         self._reference_map[op].add(op_graph)
        #     except KeyError:
        #         self._reference_map[op] = {op_graph}

    def add_nodes_from(self, nodes: Dict[str, NodeIR], **attr):
        """ Add multiple nodes to circuit. Allows networkx-style adding of nodes.

        Parameters
        ----------
        nodes
            Dictionary with node label as key. The item is a NodeIR instance. Note that the item type is not tested
            here, but passing anything that does not behave like a `NodeIR` may cause problems later.
        attr
            additional keyword attributes that can be added to the node data. (default `networkx` syntax.)
        """

        # get unique labels for nodes  --> deprecated and removed.
        # for label in nodes:
        #     self.label_map[label] = self._get_unique_label(label)

        # collect references to op_graphs in nodes
        for node in nodes.values():
            self._collect_references(node)

        # assign NodeIR instances as "node" keys in a separate dictionary, because networkx saves node attributes into
        # a dictionary
        # reformat dictionary to tuple/generator, since networkx does not parse dictionary correctly in add_nodes_from
        nodes = ((key, {"node": node}) for key, node in nodes.items())
        self.graph.add_nodes_from(nodes, **attr)

    def add_node(self, label: str, node: NodeIR, **attr):
        """Add single node

        Parameters
        ----------
        label
            String to identify node by. Is tested for uniqueness internally, and renamed if necessary. Renamed labels
            are stored in the `CircuitIR` instance attribute `label_map`.
        node
            Instance of `NodeIR`. Will be added with the key "node" to the node dictionary.
        attr
            Additional attributes (keyword arguments) that can be added to the node data. (Default `networkx` syntax.)
        """
        self.graph.add_node(label, node=node, **attr)

        # collect references to op_graph in node
        self._collect_references(node)

    def add_edges_from(self, edges: list, **attr):
        """ Add multiple edges. This method explicitly assumes, that edges are given in edge_templates instead of
        existing instances of `EdgeIR`.

        Parameters
        ----------
        edges
            List of edges, each of shape [source/op/var, target/op/var, edge_dict]. The edge_dict must contain the
            keys "edge_ir", and optionally "weight" and "delay".
        attr
            Additional attributes (keyword arguments) that can be added to the edge data. (Default `networkx` syntax.)


        Returns
        -------
        """

        edge_list = []
        for (source, target, edge_dict) in edges:
            self.add_edge(source, target,  # edge_unique_key,
                          **edge_dict, **attr)

    def add_edges_from_matrix(self, source_var: str, target_var: str, nodes: list, weight=None, delay=None,
                              template=None, **attr) -> None:
        """Adds all possible edges between the `source_var` and `target_var` of all passed `nodes`. `Weight` and `Delay`
        need to be arrays containing scalars for each of those edges.

        Parameters
        ----------
        source_var
            Pointer to a variable on the source nodes ('op/var').
        target_var
            Pointer to a variable on the target nodes ('op/var').
        nodes
            List of node names that should be connected to each other
        weight
            Optional N x N matrix with edge weights (N = number of nodes). If not passed, all edges receive a weight of
            1.0.
        delay
            Optional N x N matrix with edge delays (N = number of nodes). If not passed, all edges receive a delay of
            0.0.
        template
            Can be link to edge template that should be used for each edge.
        attr
            Additional edge attributes. Can either be N x N matrices or other scalars/objects.

        Returns
        -------
        None

        """

        # construct edge attribute dictionary from arguments
        ####################################################

        # weights and delays
        if weight is None:
            weight = 1.0
        edge_attributes = {'weight': weight, 'delay': delay}

        # template
        if template:
            edge_attributes['edge_ir'] = template if type(template) is EdgeIR else template.apply()

        # add rest of the attributes
        edge_attributes.update(attr)

        # construct edges list
        ######################

        # find out which edge attributes have been passed as matrices
        matrix_attributes = {}
        for key, attr in edge_attributes.copy().items():
            if hasattr(attr, 'shape') and len(attr.shape) >= 2:
                matrix_attributes[key] = edge_attributes.pop(key)

        # create edge list
        edges = []

        for i, source in enumerate(nodes):
            for j, target in enumerate(nodes):

                edge_attributes_tmp = {}

                # extract edge attribute value from matrices
                for key, attr in matrix_attributes.items():
                    edge_attributes_tmp[key] = attr[i, j]

                # add remaining attributes
                edge_attributes_tmp.update(edge_attributes.copy())

                # add edge to list
                source_key, target_key = f"{source}/{source_var}", f"{target}/{target_var}"

                if edge_attributes_tmp['weight'] and source_key in self and target_key in self:
                    edges.append((source_key, target_key, edge_attributes_tmp))

        # add edges to network
        self.add_edges_from(edges)

    def add_edge(self, source: str, target: str, edge_ir: EdgeIR = None, weight: float = 1., delay: float = None,
                 spread: float = None, **data):
        """
        Parameters
        ----------
        source
        target
        edge_ir
        weight
        delay
        spread
        data
            If no template is given, `data` is assumed to conform to the format that is needed to add an edge. I.e.,
            `data` needs to contain fields for `weight`, `delay`, `edge_ir`, `source_var`, `target_var`.

        Returns
        -------

        """
        # step 1: parse and verify source and target specifiers

        source_node, source_var = self._parse_edge_specifier(source, data, "source_var")

        target_node, target_var = self._parse_edge_specifier(target, data, "target_var")

        # step 2: parse source variable specifier (might be single string or dictionary for multiple source variables)

        # ToDo: treat extra_sources properly, possibly by mapping inputs and sources at this point
        source_vars, extra_sources = self._parse_source_vars(source_node, source_var, edge_ir,
                                                             data.pop("extra_sources", None))
        # step 3: verify complete source/target paths (safeguard, might be unnecessary)

        # step 4: add edges

        # temporary workaround to make sure source/target variable/operator and nodes are defined properly

        attr_dict = dict(edge_ir=edge_ir,
                         weight=weight,
                         delay=delay,
                         spread=spread,
                         source_var=source_vars,
                         target_var=target_var,
                         extra_sources=extra_sources,
                         **data)

        # ToDo: make sure multiple source variables are understood down the road

        self.graph.add_edge(source_node, target_node, **attr_dict)

        # collect references to op_graph in edge ir
        self._collect_references(edge_ir)

    def _parse_edge_specifier(self, specifier: str, data: dict, var_string: str) -> Tuple[str, Union[str, dict]]:
        """Parse source or target specifier for an edge.

        Parameters
        ----------
        specifier
            String that defines either a specific node or complete variable path of source or target for an edge.
            Format: *circuits/node/op/var
        data
            dictionary containing additional information about the edge. This function looks for a variable specifier
            as specified in `var_string`
        var_string
            String that points to an optional key of the `data` dictionary. Should be either 'source_var' or
            'target_var'

        Returns
        -------
        (node, var)

        """
        # step 1: try to get source and target variables from data dictionary, if not available, get them from
        # source/target  string

        try:
            # try to get source variable info from data dictionary
            var = data.pop(var_string)  # type: Union[str, dict]
        except KeyError:
            # not found, assume variable info is contained in `source`
            # also means that there is only one source variable (on the main source node) to take care of
            *node, op, var = specifier.split("/")
            node = "/".join(node)
            var = "/".join((op, var))
        else:
            # source_var was in data, so `source` contains only info about source node
            node = specifier  # type: str

        # step 2: verify node paths, rename if necessary
        node = self._verify_rename_node(node)  # type: str

        return node, var

    def _verify_rename_node(self, node: str) -> str:
        """Verify that a given node specifier exists in the circuit. First tries to rename it accordings to the internal
        label map and then verifies the existence of the result.

        Parameters
        ----------
        node
            String that specifies a node in this circuit. Can either be prepended with sub-circuit labels. Format
            '*circuit_labels/node_label'.

        Returns
        -------
        node
        """

        # re-reference node labels, if necessary
        # this syntax yields `node` back as default if it is not in label_map
        node = self.label_map.get(node, node)  # type: str

        # check if node path is valid
        if node not in self:
            raise PyRatesException(f"Could not find node with path `{node}`.")

        return node

    def _parse_source_vars(self, source_node: str, source_var: Union[str, dict], edge_ir, extra_sources: dict = None
                           ) -> Tuple[Union[str, dict], dict]:
        """Parse is source variable specifications. This tests, whether a single or more source variables and verifies
        all given paths.

        Parameters
        ----------
        source_node
            String that specifies a single node as source of an edge.
        source_var
            Single variable specifier string or dictionary of form `{source_op/source_var: edge_op/edge_var
        edge_ir
            Instance of an EdgeIR that contains information about the internal structure of an edge.

        Returns
        -------
        source_var
        """

        # step 1: figure out, whether only one or more source variables are defined

        try:
            # try to treat source_var as dictionary
            n_source_vars = len(source_var.keys())
        except AttributeError:
            # not a dictionary, so must be a string
            n_source_vars = 1
        else:
            # was a dictionary, treat case that it only has length 1
            if n_source_vars == 1:
                source_var = next(iter(source_var))

        if n_source_vars == 1:
            _, _ = source_var.split("/")  # should be op, var, but we do not need them here
            self._verify_path(source_node, source_var)
        else:
            # verify that number of source variables matches number of input variables in edge
            if extra_sources is not None:
                n_source_vars += len(extra_sources)

            if n_source_vars != edge_ir.n_inputs:
                raise PyRatesException(f"Mismatch between number of source variables ({n_source_vars}) and "
                                       f"inputs ({edge_ir.n_inputs}) in an edge with source '{source_node}' and source"
                                       f"variables {source_var}.")
            for node_var, edge_var in source_var.items():
                self._verify_path(source_node, node_var)

            # ToDo: Get all input variables from all operators and properly map them at this stage?
            #  note: at this stage source_var is not manipulated at all

        if extra_sources is not None:
            for edge_var, source in extra_sources.items():
                node, op, var = source.split("/")
                node = self._verify_rename_node(node)
                source = "/".join((node, op, var))
                self._verify_path(source)
                extra_sources[edge_var] = source

        return source_var, extra_sources

    def _verify_path(self, *parts: str):
        """

        Parameters
        ----------
        parts
            One or more parts of a path string

        Returns
        -------

        """

        path = "/".join(parts)

        # check if path is valid
        if path not in self:
            raise PyRatesException(f"Could not find object with path `{path}`.")

    def getitem_from_iterator(self, key: str, key_iter: Iterator[str]):

        if key in self.sub_circuits:
            item = SubCircuitView(self, key)
        else:
            item = self.graph.nodes[key]["node"]

        return item

    @property
    def nodes(self):
        """Shortcut to self.graph.nodes. See documentation of `networkx.MultiDiGraph.nodes`."""
        return self.graph.nodes

    @property
    def edges(self):
        """Shortcut to self.graph.edges. See documentation of `networkx.MultiDiGraph.edges`."""
        return self.graph.edges

    @classmethod
    def from_circuits(cls, label: str, circuits: dict):
        """Circuit creation method that takes multiple circuits (templates or instances of `CircuitIR`) as inputs to
        create one larger circuit out of these. With additional `connectivity` information, these circuit can directly
        be interlinked.

        Parameters
        ----------
        label
            Name of new circuit. Should not collide with any circuit label given in `circuits`.
        circuits
            Dictionary with unique circuit labels as keys and circuits as items. Circuits may either be instances of
            `CircuitTemplate` or `CircuitIR`. Alternatively, a circuit template may also be given via a sub-dictionary
            with keys `template` and `values`, where `values` is a dictionary of variable value updates for the given
            template.

        Returns
        -------
        circuit
            instance of `CircuitIR`
        """
        # ToDo: Rewrite doc to account for assumption, that only CircuitIR instances are allowed

        circuit = cls(label, nodes={}, edges=[])
        for name, circ in circuits.items():
            circuit.add_circuit(name, circ)
        return circuit

    def add_circuit(self, label: str, circuit):
        """ Add a single circuit (with its own nodes and edges) to this circuit (like a subgraph in a graph).

        Parameters
        ----------
        label
            Assigned name of the circuit. If this name is already in use, the label will be renamed in the form
            `label.idx`.
        circuit
            Instance of `CircuitIR` or `CircuitTemplate` or a dictionary, where the key 'template' refers to a
            `CircuitTemplate` instance and 'values' refers to updates that should be applied to the template.
        Returns
        -------

        """
        # ToDo: disallow usage of templates here

        # parse data type of circuit
        if isinstance(circuit, dict):
            circuit = circuit["template"].apply(circuit["values"])  # type: CircuitIR
        else:
            try:
                # if it is a template, apply it
                circuit = circuit.apply()  # type: CircuitIR
            except AttributeError:
                # assume circuit already is a circuitIR or similarly structured construct
                pass

        # check if given circuit label already exists in this circuit
        if label in self.sub_circuits:
            raise PyRatesException(f"Circuit label {label} already exists in this circuit. Please specify a unique "
                                   f"circuit label.")
            # may change to a rule to rename circuits (like circuit.0, circuit.1, circuit.2...) with label map and
            # counter

        # add circuit nodes, node by node, appending circuit label to node name
        for name, data in circuit.nodes(data=True):
            self.add_node(f"{label}/{name}", **data)

        # add circuit reference to sub_circuits set. Needs to be done before adding edges
        self.sub_circuits.add(label)
        for sc in circuit.sub_circuits:
            self.sub_circuits.add(f"{label}/{sc}")

        # add sub circuit label map items to local label map
        for old, new in circuit.label_map.items():
            self.label_map[f"{label}/{old}"] = f"{label}/{new}"

        # add edges
        for source, target, data in circuit.edges(data=True):
            # source_var = data.pop("source_var")
            # target_var = data.pop("target_var")
            self.add_edge(f"{label}/{source}", f"{label}/{target}", verify_paths=False, **data)

    @staticmethod
    def from_yaml(path):
        from pyrates.frontend import circuit_from_yaml
        return circuit_from_yaml(path)

    def to_dict(self):
        """Transform this object into a dictionary."""
        from pyrates.frontend.dict import from_circuit
        return from_circuit(self)

    def optimize_graph_in_place(self, max_node_idx: int = 100000, vectorize: bool = True, dde_approx: float = 0.0,
                                verbose: bool = True):
        """Restructures network graph to collapse nodes and edges that share the same operator graphs. Variable values
        get an additional vector dimension. References to the respective index is saved in the internal `label_map`."""

        if verbose:
            print("Starting automatic optimization of the network graph:")
        self._compiled = True

        # node vectorization
        old_nodes = self._vectorize_nodes_in_place(max_node_idx)
        self._vectorize_edges_in_place(max_node_idx)
        nodes = (node for node, data in old_nodes)
        self.graph.remove_nodes_from(nodes)
        if verbose:
            print("    ...nodes in the network have been vectorized.")

        # edge vectorization
        if vectorize:
            for source in self.nodes:
                for target in self.nodes:
                    self._vectorize_edges(source, target)
        if verbose:
            print("    ...edges in the network have been vectorized.")

        # go through nodes and create buffers for delayed outputs and mappings for their inputs
        for node_name in self.nodes:

            node_outputs = self.graph.out_edges(node_name, keys=True)
            node_outputs = self._sort_edges(node_outputs, 'source_var')
            node_inputs = self.graph.in_edges(node_name, keys=True)
            node_inputs = self._sort_edges(node_inputs, 'target_var')

            # loop over ouput variables of node
            for i, (out_var, edges) in enumerate(node_outputs.items()):

                # extract delay info from variable projections
                op_name, var_name = out_var.split('/')
                delays, spreads, nodes, add_delay = self._collect_delays_from_edges(edges)

                # add synaptic buffer to output variables with delay
                if add_delay:
                    self._add_edge_buffer(node_name, op_name, var_name, edges=edges, delays=delays,
                                          nodes=nodes, spreads=spreads,
                                          dde_approx=dde_approx)

            # loop over input variables of node
            for i, (in_var, edges) in enumerate(node_inputs.items()):

                # extract delay info from input variable connections
                n_inputs = len(edges)
                op_name, var_name = in_var.split('/')

                # add synaptic input collector to the input variables
                for j in range(n_inputs):
                    if n_inputs > 1:
                        self._add_edge_input_collector(node_name, op_name, var_name, idx=j, edge=edges[j])

        if verbose:
            print("    ...all edges have been connected to nodes.")

        return self

    def _vectorize_nodes_in_place(self, max_node_idx):

        # 1: collapse all nodes that use the same operator graph into one node
        ######################################################################

        node_op_graph_map = {}  # maps each unique op_graph to a collapsed node
        # node_counter = 1  # counts different unique types of nodes
        name_idx = 0  # this is a safeguard to prevent overlap of newly created node names with previous nodes

        # collect all node data, because networkx' node views update when the graph is changed.

        old_nodes = [(node_key, data["node"]) for node_key, data in self.nodes(data=True)]

        for node_key, node in old_nodes:
            op_graph = node.op_graph
            try:
                # get reference to a previously created node
                new_name, collapsed_node = node_op_graph_map[op_graph]

                # extend vectorized node by this node
                collapsed_node.extend(node)

                # refer node key to new node and respective list index of its values
                # format: (nodeX, Z) with X = node index and Z = list index for values
                self.label_map[node_key] = (new_name, len(collapsed_node) - 1)

            except KeyError:
                # if it does not exist, create a new one and save its reference in the map
                collapsed_node = VectorizedNodeIR(node)

                # create unique name and add node to local graph
                while name_idx <= max_node_idx:
                    new_name = f"vector_node{name_idx}"
                    if new_name in self.nodes:
                        name_idx += 1
                        continue
                    else:
                        break
                else:
                    raise PyRatesException(
                        "Too many nodes with generic name 'node{counter}' exist. Aborting vectorization."
                        "Consider not using this naming scheme for your own nodes as it is used for "
                        "vectorization. This problem will also occur, when more unique operator graphs "
                        "exist than the maximum number of iterations allows (default: 100k). You can "
                        "increase this number by setting `max_node_idx` to a larger number.")

                # add new node directly to node graph, bypassing external interface
                # this is the "in_place" way to do this. Otherwise we would create an entirely new CircuitIR instance
                self.graph.add_node(new_name, node=collapsed_node)
                node_op_graph_map[op_graph] = (new_name, collapsed_node)

                # now save the reference to the new node name with index number to label_map
                self.label_map[node_key] = (new_name, 0)

            # TODO: decide, whether reference collecting for operator_graphs in `_reference_map` is actually necessary
            #   and if we thus need to remove these reference again after vectorization.

        return old_nodes

    def _vectorize_edges_in_place(self, max_node_idx):
        """

        Parameters
        ----------
        max_node_idx
        """
        # 2: move all operators from edges to respective coupling nodes and reference labels accordingly
        ################################################################################################

        # we shall assume that there is no overlap between operator_graphs in edges and nodes that is supposed to be
        # accounted for in vectorization.

        node_op_graph_map = {}  # maps each unique op_graph to a collapsed node
        # node_counter = 1  # counts different unique types of nodes
        node_sizes = {}  # counts current size of vectorized nodes
        name_idx = 0  # this is a safeguard to prevent overlap of newly created node names with previous nodes

        # collect all node data, because networkx' node views update when the graph is changed.

        old_edges = [(source, target, key, data) for source, target, key, data in self.edges(data=True, keys=True)]

        for source, target, edge_key, data in old_edges:
            specifier = (source, target, edge_key)
            weight = data["weight"]
            delay = data["delay"]
            spread = data["spread"]
            edge_ir = data["edge_ir"]
            source_var = data["source_var"]  # type: Union[str, dict]
            target_var = data["target_var"]
            extra_sources = data["extra_sources"]

            if weight is None or weight != 0:

                if edge_ir is None:
                    # if the edge is empty, just add one with remapped names
                    source, source_idx = self.label_map[source]
                    target, target_idx = self.label_map[target]
                    # make sure simple edges do have only one source variable defined
                    try:
                        # try to calculate length
                        n_vars = len(source_var.keys())
                    except AttributeError:
                        # no dictionary --> we should be fine (assume string)
                        pass
                    else:
                        if n_vars > 1:
                            raise PyRatesException("Too many source variables defined. Edges with no operators allow only"
                                                   "one source variable to be defined.")
                            # could actually do this earlier

                    # add edge from source to the new node
                    self.graph.add_edge(source, target,
                                        source_var=source_var, source_idx=[source_idx],
                                        target_var=target_var, target_idx=[target_idx],
                                        weight=weight, delay=delay, spread=spread
                                        )
                else:
                    op_graph = edge_ir.op_graph

                    try:
                        # get reference to a previously created node
                        new_name, collapsed_node = node_op_graph_map[op_graph]
                        # add values to respective lists in collapsed node
                        collapsed_node.extend(edge_ir)
                        # for op_key, value_dict in edge_ir.values.items():
                        #     for var_key, value in value_dict.items():
                        #         collapsed_node.extend([f"{op_key}/{var_key}"]["value"].append(value)

                        # note current index of node
                        coupling_vec_idx = node_sizes[op_graph]
                        # increment op_graph size counter
                        node_sizes[op_graph] += 1

                    except KeyError:
                        # if it does not exist, create a new one and save its reference in the map
                        collapsed_node = VectorizedNodeIR(edge_ir)

                        # create unique name and add node to local graph
                        while name_idx <= max_node_idx:
                            new_name = f"vector_coupling{name_idx}"
                            if new_name in self.nodes:
                                name_idx += 1
                                continue
                            else:
                                break
                        else:
                            raise PyRatesException(
                                "Too many nodes with generic name 'vector_coupling{counter}' exist. Aborting vectorization."
                                "Consider not using this naming scheme for your own nodes as it is used for "
                                "vectorization. This problem will also occur, when more unique operator graphs "
                                "exist than the maximum number of iterations allows (default: 100k). You can "
                                "increase this number by setting `max_node_idx` to a larger number.")

                        # add new node directly to node graph, bypassing external interface
                        # this is the "in_place" way to do this. Otherwise we would create an entirely new CircuitIR
                        # instance
                        self.graph.add_node(new_name, node=collapsed_node)
                        node_op_graph_map[op_graph] = (new_name, collapsed_node)

                        # set current index to 0
                        coupling_vec_idx = 0
                        # and set size of this node to 1
                        node_sizes[op_graph] = 1

                    # refer node key to new node and respective list index of its values
                    # format: "nodeX[Z]" with X = node index and Z = list index for values
                    # self.label_map[specifier] = f"{new_name}[{coupling_vec_idx}]"

                    # get new reference for source/target nodes
                    # new references should have format "vector_node{node_idx}[{vector_idx}]"
                    # the following raises an error, if the format is wrong for some reason
                    source, source_idx = self.label_map[source]
                    target, target_idx = self.label_map[target]

                    source_vars = []

                    try:
                        n_vars = len(source_var.keys())
                    except AttributeError:
                        # simple/legacy case: only one input present. Unclear whether this also works with multiple
                        # operators that use the same input variable
                        n_vars = 1
                        input_var = next(iter(edge_ir.inputs.values()))[0]
                        # add edge from source to the new node
                        self.graph.add_edge(source, new_name,
                                            source_var=source_var, source_idx=[source_idx],
                                            target_var=input_var, target_idx=[coupling_vec_idx],
                                            weight=1, delay=None, spread=None
                                            )

                        # add edge from new node to target
                        self.graph.add_edge(new_name, target,
                                            source_var=edge_ir.output_var, source_idx=[coupling_vec_idx],
                                            target_var=target_var, target_idx=[target_idx],
                                            weight=weight, delay=delay, spread=spread
                                            )
                    else:
                        for in_var, op_vars in edge_ir.inputs.items():

                            input_var = op_vars[0]  # simple solution for now: take only first reference
                            # need to rewrite this, if problems arise from operators in a node referencing the same
                            # node-wide variable

                            # now fetch the source variable connected to this input variable
                            # should fail, if there is a mismatch between assigned input variable and actual inputs
                            # could check for this during edge creation in the case of multiple source variables.
                            try:
                                single_source = next((key for key, value in source_var.items() if value == in_var))
                            except StopIteration:
                                if in_var in extra_sources.keys():
                                    continue
                                else:
                                    raise PyRatesException(f"Failed to divide edge with multiple source variables into many "
                                                           f"edges with single source variable, because there is a mismatch "
                                                           f"between assigned input variables in the source definition "
                                                           f"{source_var} and inputs as defined by internal operator graph "
                                                           f"{edge_ir.inputs}. This happened in an edge between {source} and "
                                                           f"{target}.")
                            # add edge from source to the new node

                            self.graph.add_edge(source, new_name,
                                                source_var=single_source, source_idx=[source_idx],
                                                target_var=input_var, target_idx=[coupling_vec_idx],
                                                weight=1, delay=None, spread=None
                                                )

                        # add edge from new node to target
                        self.graph.add_edge(new_name, target,
                                            source_var=edge_ir.output_var, source_idx=[coupling_vec_idx],
                                            target_var=target_var, target_idx=[target_idx],
                                            weight=weight, delay=delay, spread=spread
                                            )

                    # # add edge from source to the new node
                    # self.graph.add_edge(source, new_name,
                    #                     source_var=source_var, source_idx=[source_idx],
                    #                     target_var=edge_ir.input_var, target_idx=[coupling_vec_idx],
                    #                     weight=1, delay=None, spread=None
                    #                     )
                    #
                    # # add edge from new node to target
                    # self.graph.add_edge(new_name, target,
                    #                     source_var=edge_ir.output_var, source_idx=[coupling_vec_idx],
                    #                     target_var=target_var, target_idx=[target_idx],
                    #                     weight=weight, delay=delay, spread=spread
                    #                     )

                    # in case we have additional sources that are not linked to the source node, add edges for them also
                    if extra_sources is not None:
                        for edge_var, source in extra_sources.items():
                            *node, source_op, source_var = source.split("/")
                            source, source_idx = self.label_map["/".join(node)]
                            input_var = edge_ir.inputs[edge_var][0]

                            self.graph.add_edge(source, new_name,
                                                source_var="/".join((source_op, source_var)), source_idx=[source_idx],
                                                target_var=input_var, target_idx=[coupling_vec_idx],
                                                weight=1, delay=None, spread=None
                                                )

            # remove old edge
            self.graph.remove_edge(*specifier)

    def _vectorize_edges(self, source: str, target: str) -> None:
        """Combines edges in list and adds a new edge to the new net config.

        Parameters
        ----------
        source
            Name of the source node
        target
            Name of the target node

        Returns
        -------
        None

        """

        # extract edges between source and target
        edges = [(s, t, e, d) for s, t, e, d in self.edges(source, keys=True, data=True) if t == target]

        # extract edges that connect the same variables on source and target
        ####################################################################

        idx = 0
        while edges and idx < len(edges):

            source_tmp, target_tmp, edge_tmp, edge_data_tmp = edges.pop(idx)

            # get source and target variable
            source_var = edge_data_tmp['source_var']
            target_var = edge_data_tmp['target_var']

            # get edges with equal source and target variables between source and target node
            edges_tmp = [(source_tmp, target_tmp, edge_tmp, edge_data_tmp)]
            i, n_edges = 0, len(edges)
            for _ in range(n_edges):
                if edges[i][3]['source_var'] == source_var and edges[i][3]['target_var'] == target_var:
                    edges_tmp.append(edges.pop(i))
                else:
                    i += 1

            # vectorize those edges
            #######################

            n_edges = len(edges_tmp)

            if n_edges > 0:

                # go through edges and extract weight and delay
                weight_col = []
                delay_col = []
                spread_col = []
                old_svar_idx = []
                old_tvar_idx = []

                for *_, edge_data in edges_tmp:

                    weight = edge_data['weight']
                    delay = edge_data['delay']
                    spread = edge_data['spread']

                    # add weight, delay and variable indices to collector lists
                    weight_col.append(1. if weight is None else weight)
                    delay_col.append(0. if delay is None else delay)
                    spread_col.append(0. if spread is None else spread)
                    idx_tmp = edge_data['source_idx']
                    if idx_tmp:
                        old_svar_idx += idx_tmp
                    idx_tmp = edge_data['target_idx']
                    if idx_tmp:
                        old_tvar_idx += idx_tmp

                # create new, vectorized edge
                #############################

                weights = np.squeeze(weight_col)

                if np.any(weights):

                    # extract edge
                    new_edge = self.edges[edges_tmp.pop(0)[:3]]

                    # change delay and weight attributes
                    weight_col = weights.tolist()
                    delay_col = np.squeeze(delay_col).tolist()
                    spread_col = np.squeeze(spread_col).tolist()
                    new_edge['delay'] = delay_col
                    new_edge['weight'] = weight_col
                    new_edge['spread'] = spread_col
                    new_edge['source_idx'] = old_svar_idx
                    new_edge['target_idx'] = old_tvar_idx

                # delete vectorized edges from list
                self.graph.remove_edges_from(edges_tmp)

            else:

                # advance in edge list
                print(f'WARNING: Vectorization of edges between {source}/{source_var} and {target}/{target_var} '
                      f'failed.')
                idx += 1

    def set_node_var(self, key: str, val):
        """

        Parameters
        ----------
        key
        val

        Returns
        -------

        """

        var_dict = self.get_node_var(key, apply_idx=False)
        if 'set_value' in var_dict:
            var_dict['set_value'][var_dict['var']] = val
        else:
            var_dict = var_dict.popitem()[1]
            var, idx = var_dict.pop('var'), var_dict.pop('idx')
            var[idx] = val

    def get_node_var(self, key: str, apply_idx: bool = True) -> dict:
        """This function extracts and returns variables from nodes of the network graph.

        Parameters
        ----------
        key
            Contains the node name, operator name and variable name, separated via slash notation: 'node/op/var'. The
            node name can consist of multiple slash-separated names referring to different levels of organization of the
            node hierarchy in the graph (e.g. 'circuit1/subcircuit2/node3'). At each hierarchical level, either a
            specific node name or a reference to all nodes can be passed (e.g. 'circuit1/subcircuit2/all' for all nodes
            of subcircuit2 of circuit1). Keys can refer to vectorized nodes as well as to the orginial node names.
        apply_idx
            If true, indexing will be applied to variables that need to be extracted from their vectorized versions.
            If false, the vectorized variable and the respective index will be returned.

        Returns
        -------
        dict
            Key-value pairs for each backend variable that was found to match the passed key.

        """

        # extract node, op and var name
        *node, op, var = key.split('/')

        # if node refers to vectorized network version, return variable from vectorized network
        try:
            var_dict = self[key]
            if not self._compiled:
                var_dict['value'] = self["/".join(node)].values[op][var]
                var_dict['var'] = var
                var_dict['set_value'] = self["/".join(node)].values[op]
            return var_dict
        except KeyError:

            # get mapping from original network nodes to vectorized network nodes
            #####################################################################

            # split original node keys
            node_keys = [key.split('/') for key in self.label_map]

            # remove all nodes from original node keys that are not referred to
            for i, node_lvl in enumerate(node):
                n_popped = 0
                if node_lvl != 'all':
                    for j, net_node in enumerate(node_keys.copy()):
                        if net_node[i] != node_lvl:
                            node_keys.pop(j - n_popped)
                            n_popped += 1

            # collect variable indices for the remaining nodes
            vnode_indices = {}
            for node in node_keys:
                node_name_orig = "/".join(node)
                vnode_key, vnode_idx = self.label_map[node_name_orig]
                if vnode_key not in vnode_indices:
                    vnode_indices[vnode_key] = {'var': [vnode_idx], 'nodes': [node_name_orig]}
                else:
                    vnode_indices[vnode_key]['var'].append(vnode_idx)
                    vnode_indices[vnode_key]['nodes'].append(node_name_orig)

            # apply the indices to the vectorized node variables
            for vnode_key in vnode_indices:
                var_value = self[f"{vnode_key}/{op}/{var}"]['value']
                if var_value.name == 'pyrates_index':
                    idx_start = var_value.value.index(self._backend.idx_l)
                    idx = var_value.value[idx_start + 1:-1]
                    idx = [int(i) for i in idx.split(':')]
                    idx_tmp = vnode_indices[vnode_key]['var']
                    idx = [int(i) + idx[0] for i in idx_tmp]
                    var_value = var_value.numpy()
                else:
                    idx = vnode_indices[vnode_key]['var']
                if apply_idx:
                    idx = f"{idx[0]}:{idx[-1] + 1}" if all(np.diff(idx) == 1) else [idx]
                    vnode_indices[vnode_key]['var'] = self._backend.apply_idx(var_value, idx)
                else:
                    vnode_indices[vnode_key]['var'] = var_value
                    vnode_indices[vnode_key]['idx'] = idx

            return vnode_indices

    def run(self,
            simulation_time: Optional[float] = None,
            step_size: Optional[float] = None,
            inputs: Optional[dict] = None,
            outputs: Optional[dict] = None,
            sampling_step_size: Optional[float] = None,
            solver: str = 'euler',
            out_dir: Optional[str] = None,
            verbose: bool = True,
            profile: bool = False,
            **kwargs
            ) -> Union[DataFrame, Tuple[DataFrame, float]]:
        """Simulate the backend behavior over time via a tensorflow session.

        Parameters
        ----------
        simulation_time
            Simulation time in seconds.
        step_size
            Simulation step size in seconds.
        inputs
            Inputs for placeholder variables. Each key is a string that specifies a node variable in the graph
            via the following format: 'node_name/op_name/var_nam'. Thereby, the node name can consist of multiple node
            levels for hierarchical networks and either refer to a specific node name ('../node_lvl_name/..') or to
            all nodes ('../all/..') at each level. Each value is an array that defines the input for the input variable
            over time (first dimension).
        outputs
            Output variables that will be returned. Each key is the desired name of an output variable and each value is
            a string that specifies a variable in the graph in the same format as used for the input definition:
            'node_name/op_name/var_name'.
        sampling_step_size
            Time in seconds between sampling points of the output variables.
        solver
            Numerical solving scheme to use for differential equations. Currently supported ODE solving schemes:
            - 'euler' for the explicit Euler method
            - 'scipy' for integration via the `scipy.integrate.solve_ivp` method.
        out_dir
            Directory in which to store outputs.
        verbose
            If true, status updates will be printed to the console.
        profile
            If true, the total graph execution time will be printed and returned.
        kwargs
            Keyword arguments that are passed on to the chosen solver.

        Returns
        -------
        Union[DataFrame, Tuple[DataFrame, float]]
            First entry of the tuple contains the output variables in a pandas dataframe, the second contains the
            simulation time in seconds. If profiling was not chosen during call of the function, only the dataframe
            will be returned.

        """

        filterwarnings("ignore", category=FutureWarning)

        if verbose:
            print("Simulation Progress")
            print("-------------------")

        # prepare simulation
        ####################

        if verbose:
            print("Preparing the simulation:")

        if not self._first_run:
            self._backend.remove_layer(0)
            self._backend.remove_layer(self._backend.top_layer())
        else:
            self._first_run = False

        # basic simulation parameters initialization
        if self.solver is not None:
            solver = self.solver
        if self.step_size is not None:
            step_size = self.step_size
        if step_size is None:
            raise ValueError('Step-size not provided. Please pass the desired initial simulation step-size to `run()`.')
        if not simulation_time:
            simulation_time = step_size
        sim_steps = int(np.round(simulation_time / step_size, decimals=0))

        # collect backend output variables
        ##################################

        outputs_col = {}

        if outputs:

            # go through passed output names
            for key, val in outputs.items():
                # extract respective output variables from the network and store their information
                outputs_col[key] = [[var_info['idx'], var_info['nodes']]
                                    for var_info in self.get_node_var(val, apply_idx=False).values()]

            if verbose:
                print("    ...user-defined output variables are logged.")

        # collect backend input variables
        #################################

        inputs_col = []

        if inputs:

            # go through passed inputs
            for key, val in inputs.items():

                in_shape = val.shape[1] if len(val.shape) > 1 else 1

                # extract respective input variable from the network
                for var_key, var_info in self.get_node_var(key, apply_idx=False).items():
                    var_shape = int(np.max(var_info['var'].shape)) if tuple(var_info['var'].shape) else 1
                    var_idx = var_info['idx'] if var_shape > 1 else None
                    var_idx_shape = len(var_idx) if var_idx else 1
                    if var_idx_shape == in_shape:
                        inputs_col.append((val, var_info['var'], var_idx))
                    elif (var_idx_shape % in_shape) == 0:
                        inputs_col.append((np.tile(val, (1, var_idx_shape)), var_info['var'], var_idx))
                    else:
                        inputs_col.append((np.reshape(val, (sim_steps, var_idx_shape)), var_info['var'], var_idx))

        # run simulation
        ################

        output_col, times, *time = self._backend.run(T=simulation_time, dt=step_size, dts=sampling_step_size,
                                                     out_dir=out_dir, outputs=outputs_col, inputs=inputs_col,
                                                     solver=solver, profile=profile, verbose=verbose, **kwargs)

        if verbose and profile:
            if simulation_time:
                print(f"{simulation_time}s of backend behavior were simulated in {time[0]} s given a "
                      f"simulation resolution of {step_size} s.")
            else:
                print(f"ComputeGraph computations finished after {time[0]} seconds.")

        # store output variables in data frame
        ######################################

        # ungroup grouped output variables
        outputs = {}
        for outkey, (out_val, node_keys) in output_col.items():
            for i, node_key in enumerate(node_keys):
                out_val_tmp = np.squeeze(out_val[:, i]) if len(out_val.shape) > 1 else out_val
                if len(out_val_tmp.shape) < 2:
                    outputs[(outkey,) + tuple(node_key.split('/'))] = out_val_tmp
                else:
                    for k in range(out_val_tmp.shape[1]):
                        outputs[(outkey, node_key, str(k))] = np.squeeze(out_val_tmp[:, k])

        # create data frame
        if sampling_step_size and not all(np.diff(times, 1) - sampling_step_size < step_size * 0.01):
            n = int(np.round(simulation_time / sampling_step_size, decimals=0))
            new_times = np.linspace(step_size, simulation_time, n + 1)
            for key, val in outputs.items():
                outputs[key] = np.interp(new_times, times, val)
            times = new_times
        out_vars = DataFrame(outputs, index=times)

        # return results
        ################

        if profile:
            return out_vars, time[0]
        return out_vars

    def compile(self,
                vectorization: bool = True,
                backend: str = 'numpy',
                float_precision: str = 'float32',
                matrix_sparseness: float = 0.5,
                step_size: Optional[float] = None,
                solver: Optional[str] = None,
                dde_approximation_order: int = 0,
                verbose: bool = True,
                in_place: bool = False,
                **kwargs
                ) -> AbstractBaseIR:
        """Parses IR into the backend. Returns an instance of the CircuitIR that allows for numerical simulations via
        the `CircuitIR.run` method.

        Parameters
        ----------
        vectorization
            Defines the mode of automatic parallelization optimization that should be used. Can be True for lumping all
            nodes together in a vector or False for no vectorization.
        backend
            Name of the backend in which to load the compute graph. Currently supported backends:
            - 'numpy'
            - 'tensorflow'
        float_precision
            Default precision of float variables. This is only used for variables for which no precision was given.
        matrix_sparseness
            Only relevant if `vectorization` is True. All edges that are vectorized and do not contain discrete delays
            can be realized internally via inner products between an edge weight matrix and the source variables.
            The matrix sparseness indicated how sparse edge weight matrices are allowed to be. If the sparseness of an
            edge weight matrix for a given projection would be higher, no edge weight matrix will be built/used.
        step_size
            Step-size with which the network should be simulated later on. Only needs to be passed here, if the edges of
            the network contain delays. Will be used to discretize the delays.
        solver
        dde_approximation_order
            Only relevant for delayed systems. If larger than zero, all discrete delays in the system will be
            automatically approximated by a system of (n+1) coupled ODEs that represent a convolution with a
            gamma distribution centered around the original delay (n is the approximation order).
        verbose
            If true, updates about compilation process will be displayed in the terminal.
        in_place
            If true, all variable and equation attributes on operators in the graph will be overwritten, by their
            compiled, backend-compatible versions. If false, a deep copy of the graph will be made first.
        kwargs
            Additional keyword arguments that will be passed on to the backend instance. For a full list of viable
            keyword arguments, see the documentation of the respective backend class (`numpy_backend.NumpyBackend` or
            tensorflow_backend.TensorflowBackend).

        """

        filterwarnings("ignore", category=FutureWarning)

        if verbose:
            print("Compilation Progress")
            print("--------------------")

        # set basic attributes
        ######################

        G = self if in_place else deepcopy(self)
        G.solver = solver
        G.step_size = step_size

        # instantiate the backend and set the backend default_device
        if backend == 'tensorflow':
            from pyrates.backend.tensorflow_backend import TensorflowBackend
            backend = TensorflowBackend
            kwargs['squeeze'] = True
        elif backend == 'numpy':
            from pyrates.backend.numpy_backend import NumpyBackend
            backend = NumpyBackend
        elif backend == 'fortran':
            from pyrates.backend.fortran_backend import FortranBackend
            backend = FortranBackend
            kwargs['squeeze'] = True
        else:
            raise ValueError(f'Invalid backend type: {backend}. See documentation for supported backends.')
        squeeze_vars = kwargs.pop('squeeze', False)
        kwargs['name'] = G.label
        kwargs['float_default_type'] = float_precision
        G._backend = backend(**kwargs)

        # run graph optimization and vectorization
        G._first_run = True
        G.optimize_graph_in_place(vectorize=vectorization, dde_approx=dde_approximation_order, verbose=verbose)

        # move edge operations to nodes
        ###############################

        if verbose:
            print('Loading the network model into the backend:')

        # create equations and variables for each edge
        for source_node, target_node, edge_idx, data in G.edges(data=True, keys=True):

            # extract edge information
            weight = data['weight']
            sidx = data['source_idx']
            tidx = data['target_idx']
            svar = data['source_var']
            sop, svar = svar.split("/")
            sval = G[f"{source_node}/{sop}/{svar}"]
            tvar = data['target_var']
            top, tvar = tvar.split("/")
            tval = G[f"{target_node}/{top}/{tvar}"]
            target_node_ir = G[target_node]

            # check whether edge projection can be solved by a simple inner product between a weight matrix and the
            # source variables
            dot_edge = False
            if len(tval['shape']) < 2 and len(sval['shape']) < 2 and len(sidx) > 1:

                n, m = tval['shape'][0], sval['shape'][0]

                # check whether the weight matrix is dense enough for this edge realization to be efficient
                if 1 - len(weight) / (n * m) < matrix_sparseness and n > 1 and m > 1:

                    weight_mat = np.zeros((n, m), dtype=np.float32)
                    if not tidx:
                        tidx = [0 for _ in range(len(sidx))]
                    for row, col, w in zip(tidx, sidx, weight):
                        weight_mat[row, col] = w

                    # set up weights and edge projection equation
                    eq = f"{tvar} = weight @ {svar}"
                    weight = weight_mat
                    dot_edge = True

            # set up edge projection equation and edge indices for edges that cannot be realized via a matrix product
            args = {}
            if len(tidx) > 1 and sum(tval['shape']) > 1:
                d = np.zeros((tval['shape'][0], len(tidx)))
                for i, t in enumerate(tidx):
                    d[t, i] = 1
            elif len(tidx) and sum(tval['shape']) > 1:
                d = tidx
            else:
                d = []
            idx = "[source_idx]" if sidx and sum(sval['shape']) > 1 else ""
            if not dot_edge:
                if len(d) > 1:
                    eq = f"{tvar} = target_idx @ ({svar}{idx} * weight)"
                elif len(d):
                    eq = f"{tvar}[target_idx] = {svar}{idx} * weight"
                else:
                    eq = f"{tvar} = {svar}{idx} * weight"

            # add edge variables to dict
            dtype = sval["dtype"]
            args['weight'] = {'vtype': 'constant', 'dtype': dtype, 'value': weight}
            args[tvar] = tval
            if len(d):
                args['target_idx'] = {'vtype': 'constant',
                                      'value': np.array(d, dtype=G._backend._float_def if len(d) > 1 else np.int32)}
            if idx:
                args['source_idx'] = {'vtype': 'constant', 'dtype': 'int32',
                                      'value': np.array(sidx, dtype=np.int32)}

            # add edge operator to target node
            op_name = f'edge_from_{source_node}_{edge_idx}'
            target_node_ir.add_op(op_name,
                                  inputs={svar: {'sources': [sop],
                                                 'reduce_dim': True,
                                                 'node': source_node,
                                                 'var': svar}},
                                  output=tvar,
                                  equations=[eq],
                                  variables=args)

            # connect edge operator to target operator
            target_node_ir.add_op_edge(op_name, top)

            # add input information to target operator
            inputs = G[target_node][top]['inputs']
            if tvar in inputs.keys():
                inputs[tvar]['sources'].add(op_name)
            else:
                inputs[tvar] = {'sources': [op_name],
                                'reduce_dim': True}

        if verbose:
            print("    ...all edge operations have been translated to backend-compatible equations.")

        # collect node and edge operators
        #################################

        variables = {}

        # edge operators
        edge_equations, variables_tmp = G._collect_op_layers(layers=[0], exclude=False, op_identifier="edge_from_")
        variables.update(variables_tmp)
        if any(edge_equations):
            G._backend._input_layer_added = True

        # node operators
        node_equations, variables_tmp = G._collect_op_layers(layers=[], exclude=True, op_identifier="edge_from_")
        variables.update(variables_tmp)

        # bring equations into correct order
        equations = sort_equations(edge_eqs=edge_equations, node_eqs=node_equations)

        if verbose:
            print("    ...all model equations have been collected from the network.")

        # parse all equations and variables into the backend
        ####################################################

        G._backend.bottom_layer()

        if verbose:
            print("Parsing the model equations into a compute graph.")

        # parse mapping
        variables = parse_equations(equations=equations, equation_args=variables, backend=G._backend,
                                    squeeze=squeeze_vars)
        if verbose:
            print("Compilation finished!\n")

        # save parsed variables in net config
        for key, val in variables.items():
            if key != 'y' and key != 'y_delta':
                node, op, var = key.split('/')
                if "inputs" not in var:
                    try:
                        G[f"{node}/{op}/{var}"]['value'] = val
                    except KeyError as e:
                        pass

        return G

    def generate_auto_def(self, dir: str) -> str:
        """Creates fortran files needed by auto (and pyauto) to run parameter continuaitons. The `run` method should be
        called at least once before calling this method to start parameter continuations from a well-defined
        initial state (i.e. a fixed point).

        Parameters
        ----------
        dir
            Build directory. If the `CircuitIR.run` method has been called previously, this should take the same value as the
            `build_dir` argument of `run`.

        Returns
        -------
        str
            Full path to the generated auto file.
        """

        if hasattr(self._backend, 'generate_auto_def'):
            return self._backend.generate_auto_def(dir)
        else:
            raise NotImplementedError(f'Method not implemented for the chosen backend: {self._backend.name}. Please'
                                      f'choose another backend (e.g. `fortran`) to generate an auto file of the system.'
                                      )

    def to_pyauto(self, *args, **kwargs):
        """

        Parameters
        ----------
        dir

        Returns
        -------

        """
        if hasattr(self._backend, 'to_pyauto'):
            return self._backend.to_pyauto(*args, **kwargs)
        else:
            raise NotImplementedError(f'Method not implemented for the chosen backend: {self._backend.name}. Please'
                                      f'choose another backend (e.g. `fortran`) to generate a pyauto instance of the '
                                      f'system.'
                                      )

    def _collect_op_layers(self, layers: list, exclude: bool = False, op_identifier: Optional[str] = None
                           ) -> tuple:
        """

        Parameters
        ----------
        layers
        exclude
        op_identifier

        Returns
        -------

        """

        equations = []
        variables = {}

        for node_name, node in self.nodes.items():

            op_graph = node['node'].op_graph
            graph = op_graph.copy()  # type: DiGraph

            # go through all operators on node and pre-process + extract equations and variables
            i = 0
            while graph.nodes:

                # get all operators that have no dependencies on other operators
                # noinspection PyTypeChecker
                ops = [op for op, in_degree in graph.in_degree if in_degree == 0]

                if (i in layers and not exclude) or (i not in layers and exclude):

                    if op_identifier:
                        ops_tmp = [op for op in ops if op_identifier not in op] if exclude else \
                            [op for op in ops if op_identifier in op]
                    else:
                        ops_tmp = ops
                    op_eqs, op_vars = self._collect_ops(ops_tmp, node_name=node_name)

                    # collect primary operator equations and variables
                    if i == len(equations):
                        equations.append(op_eqs)
                    else:
                        equations[i] += op_eqs
                    for key, var in op_vars.items():
                        if key not in variables:
                            variables[key] = var

                # remove parsed operators from graph
                graph.remove_nodes_from(ops)
                i += 1

        return equations, variables

    def _collect_ops(self, ops: List[str], node_name: str) -> tuple:
        """Adds a number of operations to the backend graph.

        Parameters
        ----------
        ops
            Names of the operators that should be parsed into the graph.
        node_name
            Name of the node that the operators belong to.

        Returns
        -------
        tuple
            Collected and updated operator equations and variables

        """

        # set up update operation collector variable
        equations = []
        variables = {}

        # add operations of same hierarchical lvl to compute graph
        ############################################################

        for op_name in ops:

            # retrieve operator and operator args
            op_info = self[f"{node_name}/{op_name}"]
            op_args = op_info['variables']
            op_args['inputs'] = {}

            if getattr(op_info, 'collected', False):
                break

            # handle operator inputs
            in_ops = {}
            for var_name, inp in op_info['inputs'].items():

                # go through inputs to variable
                if inp['sources']:

                    in_ops_col = {}
                    reduce_inputs = inp['reduce_dim'] if type(inp['reduce_dim']) is bool else False
                    in_node = inp['node'] if 'node' in inp else node_name
                    in_var_tmp = inp.pop('var', None)

                    for i, in_op in enumerate(inp['sources']):

                        # collect single input to op
                        in_var = in_var_tmp if in_var_tmp else self[f"{in_node}/{in_op}"]['output']
                        try:
                            in_val = self[f"{in_node}/{in_op}/{in_var}"]
                        except KeyError:
                            in_val = None
                        in_ops_col[f"{in_node}/{in_op}/{in_var}"] = in_val

                    if len(in_ops_col) > 1:
                        in_ops[var_name] = self._map_multiple_inputs(in_ops_col, reduce_inputs)
                    else:
                        key, _ = in_ops_col.popitem()
                        in_node, in_op, in_var = key.split("/")
                        in_ops[var_name] = (in_var, {in_var: key})

            # replace input variables with input in operator equations
            for var, inp in in_ops.items():
                for i, eq in enumerate(op_info['equations']):
                    op_info['equations'][i] = replace(eq, var, inp[0], rhs_only=True)
                op_args['inputs'].update(inp[1])

            # collect operator variables and equations
            scope = f"{node_name}/{op_name}"
            variables[f"{scope}/inputs"] = {}
            equations += [(eq, scope) for eq in op_info['equations']]
            for key, var in op_args.items():
                full_key = f"{scope}/{key}"
                if key == 'inputs':
                    variables[f"{scope}/inputs"].update(var)
                elif full_key not in variables:
                    variables[full_key] = var
            try:
                setattr(op_info, 'collected', True)
            except AttributeError:
                op_info['collected'] = True

        return equations, variables

    def _collect_delays_from_edges(self, edges):
        means, stds, nodes = [], [], []
        for s, t, e in edges:
            d = self.edges[s, t, e]['delay'].copy() if type(self.edges[s, t, e]['delay']) is list else \
                self.edges[s, t, e]['delay']
            v = self.edges[s, t, e].pop('spread', [0])
            if v is None or np.sum(v) == 0:
                v = [0] * len(self.edges[s, t, e]['target_idx'])
                discretize = True
            else:
                discretize = False
                v = self._process_delays(v, discretize=discretize)
            if d is None or np.sum(d) == 0:
                d = [1] * len(self.edges[s, t, e]['target_idx'])
            else:
                d = self._process_delays(d, discretize=discretize)
            means += d
            stds += v

        max_delay = np.max(means)
        add_delay = ("int" in str(type(max_delay)) and max_delay > 1) or \
                    ("float" in str(type(max_delay)) and max_delay > self.step_size)
        if sum(stds) == 0:
            stds = None

        for s, t, e in edges:
            if add_delay:
                nodes.append(self.edges[s, t, e].pop('source_idx'))
                self.edges[s, t, e]['source_idx'] = []
            self.edges[s, t, e]['delay'] = None

        return means, stds, nodes, add_delay

    def _process_delays(self, d, discretize=True):
        if self.step_size is None and self.solver == 'scipy':
            raise ValueError('Step-size not passed for setting up edge delays. If delays are added to any '
                             'network edge, please pass the simulation `step-size` to the `compile` '
                             'method.')
        if type(d) is list:
            d = np.asarray(d).squeeze()
            d = [self._preprocess_delay(d_tmp, discretize=discretize) for d_tmp in d] if d.shape else \
                [self._preprocess_delay(d, discretize=discretize)]
        else:
            d = [self._preprocess_delay(d, discretize=discretize)]
        return d

    def _preprocess_delay(self, delay, discretize=True):
        if discretize:
            discretize = self.step_size is None or self.solver != 'scipy'
        return int(np.round(delay / self.step_size, decimals=0)) if discretize else delay

    @staticmethod
    def _map_multiple_inputs(inputs: dict, reduce_dim: bool) -> tuple:
        """Creates mapping between multiple input variables and a single output variable.

        Parameters
        ----------
        inputs
            Input variables.
        reduce_dim
            If true, input variables will be summed up, if false, they will be concatenated.

        Returns
        -------
        tuple
            Summed up or concatenated input variables and the mapping to the respective input variables

        """

        inputs_unique = []
        input_mapping = {}
        for key, var in inputs.items():
            node, in_op, in_var = key.split('/')
            i = 0
            inp = in_var
            while inp in inputs_unique:
                i += 1
                if inp[-2:] == f"_{i - 1}":
                    inp = inp[:-2] + f"_{i}"
                else:
                    inp = f"{inp}_{i}"
            inputs_unique.append(inp)
            input_mapping[inp] = key

        if reduce_dim:
            inputs_unique = f"sum(({','.join(inputs_unique)}), 0)"
        else:
            idx = 0
            var = inputs[input_mapping[inputs_unique[idx]]]
            while not hasattr(var, 'shape'):
                idx += 1
                var = inputs[input_mapping[inputs_unique[idx]]]
            shape = var['shape']
            if len(shape) > 0:
                inputs_unique = f"reshape(({','.join(inputs_unique)}), ({len(inputs_unique) * shape[0],}))"
            else:
                inputs_unique = f"stack({','.join(inputs_unique)})"
        return inputs_unique, input_mapping

    def _sort_edges(self, edges: List[tuple], attr: str) -> dict:
        """Sorts edges according to the given edge attribute.

        Parameters
        ----------
        edges
            Collection of edges of interest.
        attr
            Name of the edge attribute.

        Returns
        -------
        dict
            Key-value pairs of the different values the attribute can take on (keys) and the list of edges for which
            the attribute takes on that value (value).

        """

        edges_new = {}
        for edge in edges:
            if len(edge) == 3:
                source, target, edge = edge
            else:
                raise ValueError("Missing edge index. This error message should not occur.")
            value = self.edges[source, target, edge][attr]

            if value not in edges_new.keys():
                edges_new[value] = [(source, target, edge)]
            else:
                edges_new[value].append((source, target, edge))

        return edges_new

    def _add_edge_buffer(self, node: str, op: str, var: str, edges: list, delays: list, nodes: list,
                         spreads: Optional[list] = None, dde_approx: float = 0.0) -> None:
        """Adds a buffer variable to an edge.

        Parameters
        ----------
        node
            Name of the source node of the edge.
        op
            Name of the source operator of the edge.
        var
            Name of the source variable of the edge.
        edges
            List with edge identifier tuples (source_name, target_name, edge_idx).
        delays
            edge delays.
        nodes
            Node indices for each edge delay.
        spreads
            Standard deviations of delay distributions around means given by `delays`.
        dde_approx
            Only relevant for delayed systems. If larger than zero, all discrete delays in the system will be
            automatically approximated by a system of (n+1) coupled ODEs that represent a convolution with a
            gamma distribution centered around the original delay (n is the approximation order).

        Returns
        -------
        None

        """

        max_delay = np.max(delays)

        # extract target shape and node
        node_var = self.get_node_var(f"{node}/{op}/{var}")
        target_shape = node_var['shape']
        node_ir = self[node]
        nodes_tmp = []
        for n in nodes:
            nodes_tmp += n
        source_idx = np.asarray(nodes_tmp, dtype=np.int32).flatten()

        # ODE approximation to DDE
        ##########################

        if dde_approx or spreads:

            # calculate orders and rates of ODE-system approximations to delayed connections
            if spreads:
                orders, rates = [], []
                for m, v in zip(delays, spreads):
                    order = np.round((m / v) ** 2, decimals=0) if v > 0 else 0
                    orders.append(int(order) if m and order > dde_approx else dde_approx)
                    rates.append(orders[-1] / m if m else 0)
            else:
                orders, rates = [], []
                for m in delays:
                    orders.append(dde_approx if m else 0)
                    rates.append(dde_approx / m if m else 0)

            n_edges = len(orders) / target_shape[0]
            order_idx = np.argsort(orders, kind='stable')
            orders_sorted = np.asarray(orders, dtype=np.int32)[order_idx]
            orders_tmp = np.asarray(orders, dtype=np.int32)[order_idx]
            rates_tmp = np.asarray(rates)[order_idx]

            # create ODE system equations
            buffer_eqs, var_dict, final_idx = [], {}, []
            max_order = max(orders)
            target_check = sum(target_shape) > 1 or len(orders) > 1
            var_shape = target_shape
            source_var = f'{var}_reshaped_for_edge' if n_edges > 1 else var
            for i in range(max_order + 1):
                idx, idx_str, idx_var = self._bool_to_idx(orders_tmp > i)
                idx_f1, idx_f1_str, idx_f1_var = self._bool_to_idx(orders_tmp == i)
                idx_f2, idx_f2_str, idx_f2_var = self._bool_to_idx(orders_sorted == i)
                var_dict.update(idx_var)
                var_dict.update(idx_f1_var)
                var_dict.update(idx_f2_var)
                if idx or idx == 0 or not target_check:
                    var_next = f"{var}_d{i + 1}"
                    var_prev = f"{var}_d{i}" if i > 0 else source_var
                    rate = f"k_d{i + 1}"
                    idx_apply = (idx == 0) or idx
                    val = rates_tmp[idx] if idx_apply else rates_tmp
                    if not target_check or not var_shape or (type(idx) is list and len(idx) == len(orders_tmp)):
                        idx_str, idx_f1_str = "", ""
                    var_shape = (len(val),) if val.shape else ()
                    buffer_eqs.append(f"d/dt * {var_next} = {rate} * ({var_prev}{idx_str} - {var_next})")
                    var_dict[var_next] = {'vtype': 'state_var',
                                          'dtype': self._backend._float_def,
                                          'shape': var_shape,
                                          'value': 0.}
                    var_dict[rate] = {'vtype': 'state_var',
                                      'dtype': self._backend._float_def,
                                      'value': rates_tmp[idx] if idx_apply else rates_tmp}
                    if idx_apply:
                        orders_tmp = orders_tmp[idx]
                        rates_tmp = rates_tmp[idx]
                    if not orders_tmp.shape:
                        orders_tmp = np.asarray([orders_tmp], dtype=np.int32)
                        rates_tmp = np.asarray([rates_tmp])
                if idx_f1 or idx_f1 == 0:
                    n = len(idx) if type(idx) is list and idx else 1
                    if len(delays) > 1:
                        idx1 = idx_f2_str
                        n1 = len(idx_f2) if type(idx_f2) is list and idx_f2 else 1
                    else:
                        idx1 = ""
                        n1 = target_shape[0]
                    final_idx.append((i, idx1, idx_str if n1 == n else idx_f1_str))

            # remove unnecessary ODEs
            for _ in range(len(buffer_eqs) - final_idx[-1][0]):
                i = len(buffer_eqs)
                var_dict.pop(f"{var}_d{i}")
                var_dict.pop(f"k_d{i}")
                buffer_eqs.pop(-1)

            # extend source variable to match shape of edge transmission variables
            if n_edges > 1:
                if n_edges % 1 == 0:
                    n_edges = int(n_edges)
                    for i in range(n_edges):
                        sidx1, sidx2 = str(nodes_tmp[i*target_shape[0]]), str(nodes_tmp[(i+1)*target_shape[0]-1])
                        sidx = f"{sidx1}:{int(sidx2)+1}" if sidx1 != sidx2 else sidx1
                        buffer_eqs.append(f"{source_var}[{i*target_shape[0]}:{(i+1)*target_shape[0]}] = {var}[{sidx}]")
                else:
                    for i, idx in enumerate(order_idx):
                        buffer_eqs.append(f"{source_var}[{i}] = {var}[{nodes_tmp[idx]}]")
                var_dict[source_var] = {'vtype': 'state_var',
                                        'dtype': self._backend._float_def,
                                        'shape': (len(orders),),
                                        'value': 0.0}

            # create buffered variable
            for i, idx1, idx2 in final_idx:
                idx1 = idx1 if len(idx1) > 2 else ''
                idx2 = idx2 if len(idx2) > 2 else ''
                if i != 0:
                    buffer_eqs.append(f"{var}_buffered{idx1} = {var}_d{i}{idx2}")
                else:
                    buffer_eqs.append(f"{var}_buffered{idx1} = {source_var}{idx2}")
            var_dict[f"{var}_buffered"] = {'vtype': 'state_var',
                                           'dtype': self._backend._float_def,
                                           'shape': (len(delays),),
                                           'value': 0.0}

            # re-order buffered variable if necessary
            if any(np.diff(order_idx) != 1):
                buffer_eqs.append(f"{var}_buffered = {var}_buffered[{var}_buffered_idx]")
                var_dict[f"{var}_buffered_idx"] = {'vtype': 'constant',
                                                   'dtype': 'int32',
                                                   'shape': (len(order_idx),),
                                                   'value': np.argsort(order_idx, kind='stable')}

        # discretized edge buffers
        ##########################

        elif self.step_size is None or self.solver != 'scipy':

            # create buffer variable shapes
            if len(target_shape) < 1 or (len(target_shape) == 1 and target_shape[0] == 1):
                buffer_shape = (max_delay + 1,)
            else:
                buffer_shape = (target_shape[0], max_delay + 1)

            # create buffer variable definitions
            var_dict = {f'{var}_buffer': {'vtype': 'state_var',
                                          'dtype': self._backend._float_def,
                                          'shape': buffer_shape,
                                          'value': 0.},
                        f'{var}_buffered': {'vtype': 'state_var',
                                            'dtype': self._backend._float_def,
                                            'shape': (len(delays),),
                                            'value': 0.},
                        f'{var}_delays': {'vtype': 'constant',
                                          'dtype': 'int32',
                                          'value': delays},
                        f'source_idx': {'vtype': 'constant',
                                        'dtype': 'int32',
                                        'value': source_idx}}

            # create buffer equations
            if len(target_shape) < 1 or (len(target_shape) == 1 and target_shape[0] == 1):
                buffer_eqs = [f"{var}_buffer[:] = roll({var}_buffer, 1, 0)",
                              f"{var}_buffer[0] = {var}",
                              f"{var}_buffered = {var}_buffer[{var}_delays]"]
            else:
                buffer_eqs = [f"{var}_buffer[:] = roll({var}_buffer, 1, 1)",
                              f"{var}_buffer[:, 0] = {var}",
                              f"{var}_buffered = {var}_buffer[source_idx, {var}_delays]"]

        # continuous delay buffers
        ##########################

        else:

            # create buffer variables
            max_delay_int = int(np.round(max_delay / self.step_size, decimals=0)) + 2
            times = [0. - i * self.step_size for i in range(max_delay_int)]
            if len(target_shape) < 1 or (len(target_shape) == 1 and target_shape[0] == 1):
                buffer_shape = (len(times),)
            else:
                buffer_shape = (target_shape[0], len(times))

            # create buffer variable definitions
            var_dict = {f'{var}_buffer': {'vtype': 'state_var',
                                          'dtype': self._backend._float_def,
                                          'shape': buffer_shape,
                                          'value': 0.
                                          },
                        'times': {'vtype': 'state_var',
                                  'dtype': self._backend._float_def,
                                  'shape': (len(times),),
                                  'value': np.asarray(times)
                                  },
                        't': {'vtype': 'state_var',
                              'dtype': self._backend._float_def,
                              'shape': (),
                              'value': 0.0
                              },
                        f'{var}_buffered': {'vtype': 'state_var',
                                            'dtype': self._backend._float_def,
                                            'shape': (len(delays),),
                                            'value': 0.},
                        f'{var}_delays': {'vtype': 'constant',
                                          'dtype': self._backend._float_def,
                                          'value': delays},
                        f'source_idx': {'vtype': 'constant',
                                        'dtype': 'int32',
                                        'value': source_idx},
                        f'{var}_maxdelay': {'vtype': 'constant',
                                            'dtype': self._backend._float_def,
                                            'value': (max_delay_int + 1) * self.step_size},
                        f'{var}_idx': {'vtype': 'state_var',
                                       'dtype': 'bool',
                                       'value': True}}

            # create buffer equations
            if len(target_shape) < 1 or (len(target_shape) == 1 and target_shape[0] == 1):
                buffer_eqs = [f"{var}_idx = times >= (t - {var}_maxdelay)",
                              f"{var}_buffer = {var}_buffer[{var}_idx]",
                              f"times = times[{var}_idx]",
                              f"times = append(t, times)",
                              f"{var}_buffer = append({var}, {var}_buffer)",
                              f"{var}_buffered = interpolate_1d(times, {var}_buffer, t - {var}_delays)"
                              ]
            else:
                buffer_eqs = [f"{var}_idx = times >= (t - {var}_maxdelay)",
                              f"{var}_buffer = {var}_buffer[:, {var}_idx]",
                              f"times = times[{var}_idx]",
                              f"times = append(t, times)",
                              f"{var}_buffer = append({var}, {var}_buffer, 1)",
                              f"{var}_buffered = interpolate_nd(times, {var}_buffer, {var}_delays, source_idx, t)"
                              ]
            if self._buffered:
                buffer_eqs.pop(2)
                buffer_eqs.pop(0)
            self._buffered = True

        # add buffer equations to node operator
        op_info = node_ir[op]
        op_info['equations'] += buffer_eqs
        op_info['variables'].update(var_dict)
        op_info['output'] = f"{var}_buffered"

        # update input information of node operators connected to this operator
        for succ in node_ir.op_graph.succ[op]:
            inputs = self.get_node_var(f"{node}/{succ}")['inputs']
            if var not in inputs.keys():
                inputs[var] = {'sources': {op},
                               'reduce_dim': True}

        # update edge information
        idx_l = 0
        for i, edge in enumerate(edges):
            s, t, e = edge
            self.edges[s, t, e]['source_var'] = f"{op}/{var}_buffered"
            if len(edges) > 1:
                idx_h = idx_l + len(nodes[i])
                self.edges[s, t, e]['source_idx'] = list(range(idx_l, idx_h))
                idx_l = idx_h

    def _bool_to_idx(self, v):
        v_idx = np.argwhere(v).squeeze()
        v_dict = {}
        if v_idx.shape and v_idx.shape[0] > 1 and all(np.diff(v_idx) == 1):
            v_idx_str = f"[{v_idx[0]}:{v_idx[-1] + 1}]"
        elif v_idx.shape and v_idx.shape[0] > 1:
            var_name = f"delay_idx_{self._edge_idx_counter}"
            v_idx_str = f"[{var_name}]"
            v_dict[var_name] = {'value': v_idx, 'vtype': 'constant'}
            self._edge_idx_counter += 1
        else:
            try:
                v_idx_str = f"[{v_idx.max()}]"
            except ValueError:
                v_idx_str = ""
        return v_idx.tolist(), v_idx_str, v_dict

    def _add_edge_input_collector(self, node: str, op: str, var: str, idx: int, edge: tuple) -> None:
        """Adds an input collector variable to an edge.

        Parameters
        ----------
        node
            Name of the target node of the edge.
        op
            Name of the target operator of the edge.
        var
            Name of the target variable of the edge.
        idx
            Index of the input collector variable on that edge.
        edge
            Edge identifier (source_name, target_name, edge_idx).

        Returns
        -------
        None

        """

        target_shape = self.get_node_var(f"{node}/{op}/{var}")['shape']
        node_ir = self[node]

        # create collector equation
        eqs = [f"{var} = {var}_col_{idx}"]

        # create collector variable definition
        val_dict = {'vtype': 'state_var',
                    'dtype': self._backend._float_def if self._backend is not None else float,
                    'shape': target_shape,
                    'value': 0.
                    }
        var_dict = {f'{var}_col_{idx}': val_dict,
                    var: val_dict}
        # added the actual output variable as well.

        # add collector operator to operator graph
        node_ir.add_op(f'{op}_{var}_col_{idx}',
                       inputs={},
                       output=var,
                       equations=eqs,
                       variables=var_dict)
        node_ir.add_op_edge(f'{op}_{var}_col_{idx}', op)

        # add input information to target operator
        op_inputs = self.get_node_var(f"{node}/{op}")['inputs']
        if var in op_inputs.keys():
            op_inputs[var]['sources'].add(f'{op}_{var}_col_{idx}')
        else:
            op_inputs[var] = {'sources': {f'{op}_{var}_col_{idx}'},
                              'reduce_dim': True}

        # update edge target information
        s, t, e = edge
        self.edges[s, t, e]['target_var'] = f'{op}_{var}_col_{idx}/{var}_col_{idx}'

    def clear(self):
        """Clears the backend graph from all operations and variables.
        """
        self._backend.clear()


class SubCircuitView(AbstractBaseIR):
    """View on a subgraph of a circuit. In order to keep memory footprint and computational cost low, the original (top
    lvl) circuit is referenced locally as 'top_level_circuit' and all subgraph-related information is computed only
    when needed."""

    def __init__(self, top_level_circuit: CircuitIR, subgraph_key: str):

        super().__init__()
        self.top_level_circuit = top_level_circuit
        self.subgraph_key = subgraph_key

    def getitem_from_iterator(self, key: str, key_iter: Iterator[str]):

        key = f"{self.subgraph_key}/{key}"

        if key in self.top_level_circuit.sub_circuits:
            return SubCircuitView(self.top_level_circuit, key)
        else:
            return self.top_level_circuit.nodes[key]["node"]

    @property
    def induced_graph(self):
        """Return the subgraph specified by `subgraph_key`."""

        nodes = (node for node in self.top_level_circuit.nodes if node.startswith(self.subgraph_key))
        return subgraph(self.top_level_circuit.graph, nodes)

    def __str__(self):

        return f"{self.__class__.__name__} on '{self.subgraph_key}' in {self.top_level_circuit}"


def sort_equations(edge_eqs: list, node_eqs: list) -> list:
    """

    Parameters
    ----------
    edge_eqs
    node_eqs

    Returns
    -------

    """

    # clean up equations
    for i, layer in enumerate(edge_eqs.copy()):
        if not layer:
            edge_eqs.pop(i)
    for i, layer in enumerate(node_eqs.copy()):
        if not layer:
            node_eqs.pop(i)

    # re-order node equations
    eqs_new = []
    n_popped = 0
    for i, node_layer in enumerate(node_eqs.copy()):

        # collect non-differential equations from node layer
        layer_eqs = []
        for eq, scope in node_layer.copy():
            if not is_diff_eq(eq):
                layer_eqs.append((eq, scope))
                node_layer.pop(node_layer.index((eq, scope)))

        # add non-differential equations to new equations
        if layer_eqs:
            eqs_new.append(layer_eqs)

        # clean-up already added equations from node equations
        if node_layer:
            node_eqs[i - n_popped] = node_layer
        else:
            node_eqs.pop(i - n_popped)
            n_popped += 1

    eqs_new += edge_eqs
    eqs_new += node_eqs

    return eqs_new
