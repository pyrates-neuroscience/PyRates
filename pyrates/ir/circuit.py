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
from typing import Union, Dict, Iterator, Optional, List, Tuple
from warnings import filterwarnings

from networkx import MultiDiGraph, subgraph, DiGraph
from pandas import DataFrame
import numpy as np

from pyrates import PyRatesException
from pyrates.ir.node import NodeIR, VectorizedNodeIR
from pyrates.ir.edge import EdgeIR
from pyrates.ir.abc import AbstractBaseIR
from pyrates.backend.parser import parse_dict, parse_equations, is_diff_eq, replace

__author__ = "Daniel Rose, Richard Gast"
__status__ = "Development"


class CircuitIR(AbstractBaseIR):
    """Custom graph data structure that represents a backend of nodes and edges with associated equations
    and variables."""

    # _node_label_grammar = Word(alphanums+"_") + Suppress(".") + Word(nums)
    __slots__ = ["label", "label_map", "graph", "sub_circuits", "_reference_map", "_buffered",
                 "_first_run", "_vectorized", "_compile_info", "_backend", "step_size", "solver"]

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
        self._compile_info = {}
        self._backend = None
        self._buffered = False
        self.solver = None
        self.step_size = None

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
            # get weight
            weight = edge_dict.get("weight", 1.)
            # get delay
            delay = edge_dict.get("delay", None)

            # get edge_ir or (if not included) default to an empty edge
            edge_ir = edge_dict.get("edge_ir", None)

            if "target_var" in edge_dict:
                target_var = edge_dict["target_var"]
                target = f"{target}/{target_var}"

            if "source_var" in edge_dict:
                source_var = edge_dict["source_var"]
                source = f"{source}/{source_var}"

            # test, if variables at source and target exist and reference them properly
            source, target = self._validate_separate_key_path(source, target)

            edge_list.append((source[0], target[0],  # edge_unique_key,
                              {"edge_ir": edge_ir,
                               "weight": weight,
                               "delay": delay,
                               "source_var": "/".join(source[-2:]),
                               "target_var": "/".join(target[-2:])
                               }))

            # collect references to op_graph in edge ir
            self._collect_references(edge_ir)

        self.graph.add_edges_from(edge_list, **attr)

    def add_edges_from_matrix(self, source_var: str, target_var: str, nodes: list, weight=None, delay=None, **attr
                              ) -> None:
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
                 identify_relations=True,
                 **data):
        """
        Parameters
        ----------
        source
        target
        edge_ir
        weight
        delay
        data
            If no template is given, `data` is assumed to conform to the format that is needed to add an edge. I.e.,
            `data` needs to contain fields for `weight`, `delay`, `edge_ir`, `source_var`, `target_var`.
        identify_relations

        Returns
        -------

        """

        source_var = ""
        target_var = ""
        if identify_relations:
            # test, if variables at source and target exist and reference them properly
            source, target = self._validate_separate_key_path(source, target)
        else:
            # assume that source and target strings are already consistent. This should be the case,
            # if the given strings were coming from existing circuits (instances of `CircuitIR`)
            # or in general, if operators are not renamed.
            for path in (source, target):
                if path not in self:
                    raise PyRatesException(f"Failed to add edge, because referenced node `{path}` does not exist in "
                                           f"network graph. Edges can only be added to existing nodes.")

            source_var = data.pop("source_var", "")
            target_var = data.pop("target_var", "")
            source = source.split("/")
            target = target.split("/")

        # temporary workaround to make sure source/target variable/operator and nodes are defined properly
        if source_var:
            source_node = "/".join(source)
        else:
            source_node = "/".join(source[:-2])
            source_var = "/".join(source[-2:])

        if target_var:
            target_node = "/".join(target)
        else:
            target_node = "/".join(target[:-2])
            target_var = "/".join(target[-2:])

        attr_dict = dict(edge_ir=edge_ir,
                         weight=weight,
                         delay=delay,
                         source_var=source_var,
                         target_var=target_var,
                         **data)

        self.graph.add_edge(source_node, target_node, **attr_dict)

        # collect references to op_graph in edge ir
        self._collect_references(edge_ir)

    def _validate_separate_key_path(self, *paths: str):

        for key in paths:
            # (circuits), node, operator and variable specifiers

            *node, op, var = key.split("/")

            node = "/".join(node)

            # TODO: check, whether checking the node label against the label map ist still necessary
            # re-reference node labels, if necessary
            # this syntax yields `node` back as default if it is not in label_map
            node = self.label_map.get(node, node)
            # ignore circuits for now
            path = "/".join((node, op, var))
            # check if path is valid
            if path not in self:
                raise PyRatesException(f"Could not find object with key path `{path}`.")

            separated = (node, op, var)
            yield separated

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
            self.add_edge(f"{label}/{source}", f"{label}/{target}", identify_relations=False, **data)

    @staticmethod
    def from_yaml(path):
        from pyrates.frontend import circuit_from_yaml
        return circuit_from_yaml(path)

    def optimize_graph_in_place(self, max_node_idx: int = 100000, vectorize: bool = True, dt: Optional[float] = None):
        """Restructures network graph to collapse nodes and edges that share the same operator graphs. Variable values
        get an additional vector dimension. References to the respective index is saved in the internal `label_map`."""

        # node vectorization
        old_nodes = self._vectorize_nodes_in_place(max_node_idx)
        self._vectorize_edges_in_place(max_node_idx)
        nodes = (node for node, data in old_nodes)
        self.graph.remove_nodes_from(nodes)

        # edge vectorization
        if vectorize:
            for source in self.nodes:
                for target in self.nodes:
                    self._vectorize_edges(source, target)

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
                delays, nodes, add_delay = self._collect_delays_from_edges(edges)

                # add synaptic buffer to output variables with delay
                if add_delay:
                    self._add_edge_buffer(node_name, op_name, var_name, edges=edges, delays=delays,
                                          nodes=nodes if len(delays) > 1 else None)

            # loop over input variables of node
            for i, (in_var, edges) in enumerate(node_inputs.items()):

                # extract delay info from input variable connections
                n_inputs = len(edges)
                op_name, var_name = in_var.split('/')

                # add synaptic input collector to the input variables
                for j in range(n_inputs):
                    if n_inputs > 1:
                        self._add_edge_input_collector(node_name, op_name, var_name, idx=j, edge=edges[j])

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
                self.label_map[node_key] = (new_name, len(collapsed_node)-1)

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
            edge_ir = data["edge_ir"]
            source_var = data["source_var"]
            target_var = data["target_var"]
            if edge_ir is None:
                # if the edge is empty, just add one with remapped names
                source, source_idx = self.label_map[source]
                target, target_idx = self.label_map[target]

                # add edge from source to the new node
                self.graph.add_edge(source, target,
                                    source_var=source_var, source_idx=[source_idx],
                                    target_var=target_var, target_idx=[target_idx],
                                    weight=weight, delay=delay
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
                            "Too many nodes with generic name 'node{counter}' exist. Aborting vectorization."
                            "Consider not using this naming scheme for your own nodes as it is used for "
                            "vectorization. This problem will also occur, when more unique operator graphs "
                            "exist than the maximum number of iterations allows (default: 100k). You can "
                            "increase this number by setting `max_node_idx` to a larger number.")

                    # add new node directly to node graph, bypassing external interface
                    # this is the "in_place" way to do this. Otherwise we would create an entirely new CircuitIR instance
                    self.graph.add_node(new_name, node=collapsed_node)
                    node_op_graph_map[op_graph] = (new_name, collapsed_node)

                    # set current index to 0
                    coupling_vec_idx = 0
                    # and set size of this node to 1
                    node_sizes[op_graph] = 1

                # TODO: decide, whether reference collecting for operator_graphs in `_reference_map` is actually necessary
                #   and if we thus need to remove these reference again after vectorization.

                # refer node key to new node and respective list index of its values
                # format: "nodeX[Z]" with X = node index and Z = list index for values
                self.label_map[specifier] = f"{new_name}[{coupling_vec_idx}]"

                # get new reference for source/target nodes
                # new references should have format "vector_node{node_idx}[{vector_idx}]"
                # the following raises an error, if the format is wrong for some reason
                source, source_idx = self.label_map[source]
                target, target_idx = self.label_map[target]

                # add edge from source to the new node
                self.graph.add_edge(source, new_name,
                                    source_var=source_var, source_idx=[source_idx],
                                    target_var=edge_ir.input_var, target_idx=[coupling_vec_idx],
                                    weight=1, delay=None
                                    )

                # add edge from new node to target
                self.graph.add_edge(new_name, target,
                                    source_var=edge_ir.output_var, source_idx=[coupling_vec_idx],
                                    target_var=target_var, target_idx=[target_idx],
                                    weight=weight, delay=delay
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
                old_svar_idx = []
                old_tvar_idx = []

                for *_, edge_data in edges_tmp:

                    weight = edge_data['weight']
                    delay = edge_data['delay']

                    # add weight, delay and variable indices to collector lists
                    weight_col.append(1. if weight is None else weight)
                    delay_col.append(0. if delay is None else delay)
                    idx_tmp = edge_data['source_idx']
                    if idx_tmp:
                        old_svar_idx += idx_tmp
                    idx_tmp = edge_data['target_idx']
                    if idx_tmp:
                        old_tvar_idx += idx_tmp

                # create new, vectorized edge
                #############################

                # extract edge

                new_edge = self.edges[edges_tmp.pop(0)[:3]]

                # change delay and weight attributes
                weight_col = np.squeeze(weight_col).tolist()
                delay_col = np.squeeze(delay_col).tolist()
                new_edge['delay'] = delay_col
                new_edge['weight'] = weight_col
                new_edge['source_idx'] = old_svar_idx
                new_edge['target_idx'] = old_tvar_idx

                # delete vectorized edges from list
                self.graph.remove_edges_from(edges_tmp)

            else:

                # advance in edge list
                print(f'WARNING: Vectorization of edges between {source}/{source_var} and {target}/{target_var} '
                      f'failed.')
                idx += 1

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
            return self[key]
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
                            node_keys.pop(j-n_popped)
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
                    idx_start = var_value.value.index('[')
                    idx = var_value.value[idx_start + 1:-1]
                    idx = [int(i) for i in idx.split(':')]
                    idx_tmp = vnode_indices[vnode_key]['var']
                    idx = [int(i)+idx[0] for i in idx_tmp]
                    var_value = var_value.eval()
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

        # prepare simulation
        ####################

        if verbose:
            print("Preparing the simulation...")

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
        sim_steps = int(np.round(simulation_time/step_size, decimals=0))

        # collect backend output variables
        ##################################

        outputs_col = {}

        if outputs:

            # go through passed output names
            for key, val in outputs.items():

                # extract respective output variables from the network and store their information
                outputs_col[key] = [[var_info['idx'], var_info['nodes']]
                                    for var_info in self.get_node_var(val, apply_idx=False).values()]

        # collect backend input variables
        #################################

        inputs_col = []

        if inputs:

            # go through passed inputs
            for key, val in inputs.items():

                in_shape = val.shape[1] if len(val.shape) > 1 else 1

                # extract respective input variable from the network
                for var_key, var_info in self.get_node_var(key, apply_idx=False).items():
                    var_shape = len(var_info['idx'])
                    var_idx = var_info['idx'] if np.sum(var_shape) > 1 else None
                    if var_shape == in_shape:
                        inputs_col.append((val, var_info['var'], var_idx))
                    elif (var_shape % in_shape) == 0:
                        inputs_col.append((np.tile(val, (1, var_shape)), var_info['var'], var_idx))
                    else:
                        inputs_col.append((np.reshape(val, (sim_steps, var_shape)), var_info['var'], var_idx))

        # run simulation
        ################

        if verbose:
            print("Running the simulation...")

        output_col, times, *time = self._backend.run(T=simulation_time, dt=step_size, dts=sampling_step_size,
                                                     out_dir=out_dir, outputs=outputs_col, inputs=inputs_col,
                                                     solver=solver, profile=profile, **kwargs)

        if verbose and profile:
            if simulation_time:
                print(f"{simulation_time}s of backend behavior were simulated in {time[0]} s given a "
                      f"simulation resolution of {step_size} s.")
            else:
                print(f"ComputeGraph computations finished after {time[0]} seconds.")
        elif verbose:
            print('finished!')

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
        if sampling_step_size and not all(np.diff(times, 1) == step_size):
            n = int(np.round(simulation_time/sampling_step_size, decimals=0))
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

        kwargs
            Additional keyword arguments that will be passed on to the backend instance. For a full list of viable
            keyword arguments, see the documentation of the respective backend class (`numpy_backend.NumpyBackend` or
            tensorflow_backend.TensorflowBackend).

        """

        filterwarnings("ignore", category=FutureWarning)

        # set basic attributes
        ######################

        self.solver = solver
        self.step_size = step_size

        # instantiate the backend and set the backend default_device
        if backend == 'tensorflow':
            from pyrates.backend.tensorflow_backend import TensorflowBackend
            backend = TensorflowBackend
        elif backend == 'numpy':
            from pyrates.backend.numpy_backend import NumpyBackend
            backend = NumpyBackend
        else:
            raise ValueError(f'Invalid backend type: {backend}. See documentation for supported backends.')
        kwargs['name'] = self.label
        kwargs['float_default_type'] = float_precision
        self._backend = backend(**kwargs)

        # run graph optimization and vectorization
        self._first_run = True
        self.optimize_graph_in_place(vectorize=vectorization, dt=step_size)

        # move edge operations to nodes
        ###############################

        print('building the compute graph...')

        # create equations and variables for each edge
        for source_node, target_node, edge_idx, data in self.edges(data=True, keys=True):

            # extract edge information
            weight = data['weight']
            sidx = data['source_idx']
            tidx = data['target_idx']
            svar = data['source_var']
            sop, svar = svar.split("/")
            sval = self[f"{source_node}/{sop}/{svar}"]
            tvar = data['target_var']
            top, tvar = tvar.split("/")
            tval = self[f"{target_node}/{top}/{tvar}"]
            target_node_ir = self[target_node]

            # create mapping equation and its arguments
            args = {}
            if tidx and sum(tval['shape']) > 1:
                d = np.zeros((tval['shape'][0], len(tidx)))
                for i, t in enumerate(tidx):
                    d[t, i] = 1
            else:
                d = []
            idx = "[source_idx]" if sidx and sum(sval['shape']) > 1 else ""

            # check whether edge projection can be solved by a simple inner product between a weight matrix and the
            # source variables
            dot_edge = False
            if len(tval['shape']) < 2 and len(sval['shape']) < 2 and len(sidx) > 1:

                n, m = tval['shape'][0], sval['shape'][0]

                # check whether the weight matrix is dense enough for this edge realization to be efficient
                if 1-len(weight)/(n*m) < matrix_sparseness and n > 1 and m > 1:

                    weight_mat = np.zeros((n, m), dtype=np.float32)
                    if not tidx:
                        tidx = [0 for _ in range(len(sidx))]
                    for row, col, w in zip(tidx, sidx, weight):
                        weight_mat[row, col] = w

                    # set up weights and edge projection equation
                    eq = f"{tvar} = target_idx @ (weight @ {svar}{idx})" if len(d) \
                        else f"{tvar} = weight @ {svar}{idx}"
                    weight = weight_mat
                    dot_edge = True

            # set up edge projection equation and edge indices for edges that cannot be realized via a matrix product
            if not dot_edge:
                eq = f"{tvar} = target_idx @ ({svar}{idx} * weight)" if len(d) \
                    else f"{tvar} = {svar}{idx} * weight"
                if len(d):
                    args['target_idx'] = {'vtype': 'constant', 'dtype': self._backend._float_def,
                                          'value': np.array(d, dtype=np.float32)}
                if idx:
                    args['source_idx'] = {'vtype': 'constant', 'dtype': 'int32',
                                          'value': np.array(sidx, dtype=np.int32)}

            # add edge variables to dict
            dtype = sval["dtype"]
            args['weight'] = {'vtype': 'constant', 'dtype': dtype, 'value': weight}
            args[tvar] = tval

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
            inputs = self[target_node][top]['inputs']
            if tvar in inputs.keys():
                inputs[tvar]['sources'].add(op_name)
            else:
                inputs[tvar] = {'sources': [op_name],
                                'reduce_dim': True}

        # collect node and edge operators
        #################################

        variables = {}

        # edge operators
        edge_equations, variables_tmp = self._collect_op_layers(layers=[0], exclude=False, op_identifier="edge_from_")
        variables.update(variables_tmp)
        if any(edge_equations):
            self._backend._input_layer_added = True

        # node operators
        node_equations, variables_tmp = self._collect_op_layers(layers=[], exclude=True, op_identifier="edge_from_")
        variables.update(variables_tmp)

        # bring equations into correct order
        equations = sort_equations(edge_eqs=edge_equations, node_eqs=node_equations)

        # parse all equations and variables into the backend
        ####################################################

        self._backend.bottom_layer()

        # parse mapping
        variables = parse_equations(equations=equations, equation_args=variables, backend=self._backend)

        # save parsed variables in net config
        for key, val in variables.items():
            if key != 'y' and key != 'y_delta':
                node, op, var = key.split('/')
                if "inputs" not in var:
                    try:
                        self[f"{node}/{op}/{var}"]['value'] = val
                    except KeyError as e:
                        pass

        return self

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
        delays, nodes = [], []
        for s, t, e in edges:
            d = self.edges[s, t, e]['delay'].copy() if type(self.edges[s, t, e]['delay']) is list else \
                self.edges[s, t, e]['delay']
            if d is None or np.sum(d) == 0:
                d = [1] * len(self.edges[s, t, e]['target_idx'])
            else:
                if self.step_size is None and self.solver == 'scipy':
                    raise ValueError('Step-size not passed for setting up edge delays. If delays are added to any '
                                     'network edge, please pass the simulation `step-size` to the `compile` '
                                     'method.')
                if type(d) is list:
                    d = np.asarray(d).squeeze()
                    d = [self._preprocess_delay(d_tmp) for d_tmp in d] if d.shape else \
                        [self._preprocess_delay(d)]
                else:
                    d = [self._preprocess_delay(d)]
            delays += d

        max_delay = np.max(delays)
        add_delay = ("int" in str(type(max_delay)) and max_delay > 1) or \
                    ("float" in str(type(max_delay)) and max_delay > self.step_size)

        for s, t, e in edges:
            if add_delay:
                nodes.append(self.edges[s, t, e].pop('source_idx'))
                self.edges[s, t, e]['source_idx'] = []
            self.edges[s, t, e]['delay'] = None

        return delays, nodes, add_delay

    def _preprocess_delay(self, delay):
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

    def _add_edge_buffer(self, node: str, op: str, var: str, edges: list, delays: list, nodes: list) -> None:
        """Adds a buffer variable to an edge.

        Parameters
        ----------
        node
            Name of the target node of the edge.
        op
            Name of the target operator of the edge.
        var
            Name of the target variable of the edge.
        edges
            List with edge identifier tuples (source_name, target_name, edge_idx).
        delays
            edge delays.
        nodes
            Node indices for each edge delay.

        Returns
        -------
        None

        """

        max_delay = np.max(delays)

        # extract target shape and node
        node_var = self.get_node_var(f"{node}/{op}/{var}")
        target_shape = node_var['shape']
        node_ir = self[node]
        source_idx = np.asarray(nodes, dtype=np.int32).flatten()

        # discretized edge buffers
        ##########################

        if self.step_size is None or self.solver != 'scipy':

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
            max_delay_int = int(np.round(max_delay / self.step_size, decimals=0))
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
                                  'value': times
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
                                        'value': source_idx}}

            # create buffer equations
            if len(target_shape) < 1 or (len(target_shape) == 1 and target_shape[0] == 1):
                buffer_eqs = [f"times[:] = roll(times, 1)",
                              f"{var}_buffer[:] = roll({var}_buffer, 1)",
                              f"times[0] = t",
                              f"{var}_buffer[0] = {var}",
                              f"{var}_buffered = interpolate_1d(times, {var}_buffer, t + {var}_delays)"
                              ]
            else:
                buffer_eqs = [f"times[:] = roll(times, 1)",
                              f"{var}_buffer[:] = roll({var}_buffer, 1, 1)",
                              f"times[0] = t",
                              f"{var}_buffer[:, 0] = {var}",
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
            node_eqs[i-n_popped] = node_layer
        else:
            node_eqs.pop(i-n_popped)
            n_popped += 1

    eqs_new += edge_eqs
    eqs_new += node_eqs

    return eqs_new
