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
from copy import deepcopy
from typing import Union, Dict, Iterator

from pyparsing import Word, ParseException, nums, Literal
from networkx import MultiDiGraph, subgraph, find_cycle, NetworkXNoCycle, DiGraph
from pandas import DataFrame

from pyrates import PyRatesException
from pyrates.ir.node import NodeIR, VectorizedNodeIR
from pyrates.ir.edge import EdgeIR
from pyrates.ir.abc import AbstractBaseIR

__author__ = "Daniel Rose"
__status__ = "Development"


class CircuitIR(AbstractBaseIR):
    """Custom graph data structure that represents a backend of nodes and edges with associated equations
    and variables."""

    # _node_label_grammar = Word(alphanums+"_") + Suppress(".") + Word(nums)
    __slots__ = ["label", "label_map", "graph", "sub_circuits", "_reference_map"]

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

    def _collect_references(self, edge_or_node):
        """Collect all references of nodes or edges to unique operator_graph instances in local `_reference_map`.
        References are collected as a list, because nodes and edges are (currently) not hashable."""

        op_graph = edge_or_node.op_graph
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

    def add_edges_from(self, edges, **attr):
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
    def from_circuits(cls, label: str, circuits: dict, connectivity: Union[list, tuple, DataFrame] = None):
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
        connectivity
            Optional `list`, `tuple` or `pandas.DataFrame' with connectivity information to create edges between the
            given circuits. If `list` or `tuple`, then each item must be formatted the same way as `edges` in
            `add_edges_from`: ('circuit/source_node/op/var', 'circuit/target_node/op/var', edge_template, variables).
            If given as a `DataFrame`, keys (indices and column names) must refer to sources and targets, respectively,
            as column name/index (string of form 'circuit/node/op/var') and items may then be edge templates and
            associated variables.
            Empty cells in the DataFrame should be filled with something 'falsy' (as in evaluates to `False` in Python).

        Returns
        -------
        circuit
            instance of `CircuitIR`
        """
        # ToDo: Rewrite doc to account for assumption, that only CircuitIR instances are allowed

        circuit = cls(label, nodes={}, edges=[])
        for name, circ in circuits.items():
            circuit.add_circuit(name, circ)

        if connectivity is not None:
            if isinstance(connectivity, list) or isinstance(connectivity, tuple):
                circuit.add_edges_from(connectivity)
            else:
                try:
                    if isinstance(connectivity, dict):
                        key, conn_info = connectivity.popitem()
                        for target, row in conn_info.iterrows():
                            for source, content in row.iteritems():
                                snode, tnode = source.split('/')[:-2], target.split('/')[:-2]
                                svar, tvar = source.split('/')[-2:], target.split('/')[-2:]
                                snode, tnode = "/".join(snode), "/".join(tnode)
                                svar, tvar = "/".join(svar), "/".join(tvar)
                                content = {key: content} if content else {}
                                for key_tmp, conn_info_tmp in connectivity.items():
                                    content_tmp = conn_info_tmp.loc[target, source]
                                    if content_tmp:
                                        content.update({key_tmp: content_tmp})
                                content.update({'source_var': svar, 'target_var': tvar})
                                if 'weight' in content and content['weight']:
                                    circuit.add_edge(snode, tnode, edge_ir=None, identify_relations=False,
                                                     **content)
                    else:
                        for target, row in connectivity.iterrows():
                            for source, content in row.iteritems():
                                if content:  # assumes, empty entries evaluate to `False`
                                    snode, tnode = source.split('/')[:-2], target.split('/')[:-2]
                                    svar, tvar = source.split('/')[-2:], target.split('/')[-2:]
                                    snode, tnode = "/".join(snode), "/".join(tnode)
                                    svar, tvar = "/".join(svar), "/".join(tvar)
                                    if "float" in str(type(content)):
                                        content = {'weight': content, 'delay': None}
                                    content.update({'source_var': svar, 'target_var': tvar})
                                    circuit.add_edge(snode, tnode, edge_ir=None, identify_relations=False, **content)
                except AttributeError:
                    raise TypeError(f"Invalid data type of variable `connectivity` (type: {type(connectivity)}).")

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

    def optimize_graph_in_place(self, max_node_idx: int = 100000):
        """Restructures network graph to collapse nodes and edges that share the same operator graphs. Variable values
        get an additional vector dimension. References to the respective index is saved in the internal `label_map`."""

        old_nodes = self._vectorize_nodes_in_place(max_node_idx)

        self._vectorize_edges_in_place(max_node_idx)

        nodes = (node for node, data in old_nodes)
        self.graph.remove_nodes_from(nodes)

        return self

    def _vectorize_nodes_in_place(self, max_node_idx):

        # 1: collapse all nodes that use the same operator graph into one node
        ######################################################################

        node_op_graph_map = {}  # maps each unique op_graph to a collapsed node
        # node_counter = 1  # counts different unique types of nodes
        node_size = {}  # counts current size of vectorized nodes
        name_idx = 0  # this is a safeguard to prevent overlap of newly created node names with previous nodes

        # collect all node data, because networkx' node views update when the graph is changed.

        old_nodes = [(node_key, data["node"]) for node_key, data in self.nodes(data=True)]

        for node_key, node in old_nodes:
            op_graph = node.op_graph
            try:
                # get reference to a previously created node
                new_name, collapsed_node = node_op_graph_map[op_graph]
                # add values to respective lists in collapsed node
                for op_key, value_dict in node.values.items():
                    for var_key, value in value_dict.items():
                        collapsed_node.values[op_key][var_key].append(value)

                # refer node key to new node and respective list index of its values
                # format: "nodeX[Z]" with X = node index and Z = list index for values
                self.label_map[node_key] = f"{new_name}[{node_size[op_graph]}]"
                # increment op_graph size counter
                node_size[op_graph] += 1

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
                self.label_map[node_key] = f"{new_name}[0]"
                # and set size of this node to 1
                node_size[op_graph] = 1

            # TODO: decide, whether reference collecting for operator_graphs in `_reference_map` is actually necessary
            #   and if why thus need to remove these reference again after vectorization.

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
                op_graph = None
            else:
                op_graph = edge_ir.op_graph

            try:
                # get reference to a previously created node
                new_name, collapsed_node = node_op_graph_map[op_graph]
                # add values to respective lists in collapsed node
                for op_key, value_dict in edge_ir.values.items():
                    for var_key, value in value_dict.items():
                        collapsed_node.values[op_key][var_key].append(value)

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
            # the follow raises an error, if the format is wrong for some reason
            source = self.label_map[source]
            source, source_idx = source.split("[")
            source_idx = int(source_idx[:-1])
            target = self.label_map[target]
            target, target_idx = target.split("[")
            target_idx = int(target_idx[:-1])

            # add edge from source to the new node
            self.graph.add_edge(source, new_name,
                                source_var=source_var, source_idx=source_idx,
                                target_var=edge_ir.input_var, target_idx=coupling_vec_idx,
                                weight=1, delay=0
                                )

            # add edge from new node to target
            self.graph.add_edge(new_name, target,
                                source_var=edge_ir.output_var, source_idx=coupling_vec_idx,
                                target_var=target_var, target_idx=target_idx,
                                weight=weight, delay=delay
                                )

            # remove old edge
            self.graph.remove_edge(*specifier)

    # def _vectorize_common_in_place(self, max_node_idx, old_name, node_op_graph_map, node_sizes, name_idx):


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
