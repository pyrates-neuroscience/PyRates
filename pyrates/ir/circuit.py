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
from typing import Union, Dict, Iterator, Optional, List, Tuple, Any
from warnings import filterwarnings

from pyparsing import Word, ParseException, nums, Literal
from networkx import MultiDiGraph, subgraph, find_cycle, NetworkXNoCycle, DiGraph
from pandas import DataFrame, MultiIndex
import numpy as np

from pyrates import PyRatesException
# from pyrates.backend import parse_dict
from pyrates.ir.node import NodeIR, VectorizedNodeIR
from pyrates.ir.edge import EdgeIR
from pyrates.ir.abc import AbstractBaseIR
from pyrates.backend.parser import parse_dict, parse_equation_system, is_diff_eq, replace

__author__ = "Daniel Rose"
__status__ = "Development"


class CircuitIR(AbstractBaseIR):
    """Custom graph data structure that represents a backend of nodes and edges with associated equations
    and variables."""

    # _node_label_grammar = Word(alphanums+"_") + Suppress(".") + Word(nums)
    __slots__ = ["label", "label_map", "graph", "sub_circuits", "_reference_map",
                 "_first_run", "_vectorized", "_compile_info", "_backend", "_dt"]

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
        self._dt = 0.0

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

    @staticmethod
    def from_yaml(path):
        from pyrates.frontend import circuit_from_yaml
        return circuit_from_yaml(path)

    def optimize_graph_in_place(self, max_node_idx: int = 100000, vectorize: bool = True):
        """Restructures network graph to collapse nodes and edges that share the same operator graphs. Variable values
        get an additional vector dimension. References to the respective index is saved in the internal `label_map`."""

        old_nodes = self._vectorize_nodes_in_place(max_node_idx)

        self._vectorize_edges_in_place(max_node_idx)

        nodes = (node for node, data in old_nodes)
        self.graph.remove_nodes_from(nodes)

        if vectorize:

            # go through new nodes
            for source in self.nodes:
                for target in self.nodes:
                    self._vectorize_edges(source, target)

        # go through nodes and create mapping for their inputs
        for node_name in self.nodes:

            node_inputs = self.graph.in_edges(node_name, keys=True)
            node_inputs = self._sort_edges(node_inputs, 'target_var')

            # loop over input variables of node
            for i, (in_var, edges) in enumerate(node_inputs.items()):

                # extract delay info from input variable connections
                n_inputs = len(edges)
                op_name, var_name = in_var.split('/')
                delays = []
                for s, t, e in edges:
                    d = self.edges[s, t, e]['delay']
                    if d is None or np.sum(d) == 0:
                        d = 0
                    else:
                        if type(d) is list:
                            d_tmp = np.asarray(d)
                            d = np.asarray((d_tmp/self._dt) + 1, dtype=np.int32).tolist()
                        else:
                            d = int(d/self._dt)+1
                    self.edges[s, t, e]['delay'] = d
                    delays.append(d)

                max_delay = np.max(delays)

                # set delays to None of max_delay is 0
                if max_delay == 0:
                    for s, t, e in edges:
                        self.edges[s, t, e]['delay'] = None

                # loop over different input sources
                for j in range(n_inputs):

                    if max_delay > 0:

                        # add synaptic buffer to the input variable
                        self._add_edge_buffer(node_name, op_name, var_name, idx=j,
                                              buffer_length=max_delay, edge=edges[j])

                    elif n_inputs > 1:

                        # add synaptic input collector to the input variable
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

    # def _vectorize_common_in_place(self, max_node_idx, old_name, node_op_graph_map, node_sizes, name_idx):

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
        edges = [(s, t, e) for s, t, e, _ in self.edges(source, target, keys=True)]

        # extract edges that connect the same variables on source and target
        ####################################################################

        while edges:

            source_tmp, target_tmp, edge_tmp = edges[0]

            # get source and target variable
            source_var = self.edges[source_tmp, target_tmp, edge_tmp]['source_var']
            target_var = self.edges[source_tmp, target_tmp, edge_tmp]['target_var']

            # get edges with equal source and target variables between source and target node
            edges_tmp = []
            for n, (source_tmp, target_tmp, edge_tmp) in enumerate(edges):
                if self.edges[source_tmp, target_tmp, edge_tmp]['source_var'] == source_var and \
                        self.edges[source_tmp, target_tmp, edge_tmp]['target_var'] == target_var:
                    edges_tmp.append(edges[n])

            # vectorize those edges
            #######################

            n_edges = len(edges_tmp)

            if n_edges > 0:

                # go through edges and extract weight and delay
                weight_col = []
                delay_col = []
                old_svar_idx = []
                old_tvar_idx = []

                for source, target, idx in edges_tmp:

                    weight = self.edges[source, target, idx]['weight']
                    delay = self.edges[source, target, idx]['delay']

                    # add weight, delay and variable indices to collector lists
                    if weight is not None:
                        weight_col.append(weight)
                        delay_col.append(0. if delay is None else delay)
                    idx_tmp = self.edges[source, target, idx]['source_idx']
                    idx_tmp = [idx_tmp] if type(idx_tmp) is int else list(idx_tmp)
                    if idx_tmp:
                        old_svar_idx += idx_tmp
                    idx_tmp = self.edges[source, target, idx]['target_idx']
                    idx_tmp = [idx_tmp] if type(idx_tmp) is int else list(idx_tmp)
                    if idx_tmp:
                        old_tvar_idx += idx_tmp

                # create new, vectorized edge
                #############################

                # extract edge
                edge_ref = edges_tmp[0]
                new_edge = self.edges[edge_ref]

                # change delay and weight attributes
                if weight_col:
                    weight_col = np.squeeze(weight_col).tolist()
                    delay_col = np.squeeze(delay_col).tolist()
                else:
                    weight_col = None
                    delay_col = None
                new_edge['delay'] = delay_col
                new_edge['weight'] = weight_col
                new_edge['source_idx'] = old_svar_idx
                new_edge['target_idx'] = old_tvar_idx

                # add new edge to new net config
                self.graph.add_edge(source, target, **new_edge)

            # delete vectorized edges from list
            self.graph.remove_edges_from(edges_tmp)
            for edge in edges_tmp:
                edges.pop(edges.index(edge))

    def get_node_var(self, key: str, apply_idx: bool = True) -> dict:
        """

        Parameters
        ----------
        key
        apply_idx

        Returns
        -------

        """

        # extract node, op and var name
        key_split = key.split('/')
        node, op, var = key_split[:-2], key_split[-2], key_split[-1]

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
                idx = vnode_indices[vnode_key]['var']
                if apply_idx:
                    idx = f"{idx[0]}:{idx[-1] + 1}" if all(np.diff(idx) == 1) else [idx]
                    vnode_indices[vnode_key]['var'] = self._backend.apply_idx(self[f"{vnode_key}/{op}/{var}"]['value'],
                                                                              idx)
                else:
                    vnode_indices[vnode_key]['var'] = self[f"{vnode_key}/{op}/{var}"]['value']
                    vnode_indices[vnode_key]['idx'] = idx

            return vnode_indices

    def _get_node_attr(self, node: str, attr: str, op: Optional[str] = None, **kwargs) -> Any:
        """Extract attribute from node of the network.

        Parameters
        ----------
        node
            Name of the node.
        attr
            Name of the attribute on the node.
        op
            Name of the operator. Only needs to be provided for operator variables.

        Returns
        -------
        Any
            Node attribute.

        """

        if op:
            return self._get_op_attr(node, op, attr, **kwargs)
        try:
            return self[node][attr]
        except KeyError:
            vals = []
            for op in self[node]:
                vals.append(self._get_op_attr(node, op, attr, **kwargs))
            return vals

    def _get_op_attr(self, node: str, op: str, attr: str, retrieve: bool = True, **kwargs) -> Any:
        """Extracts attribute of an operator.

        Parameters
        ----------
        node
            Name of the node.
        op
            Name of the operator on the node.
        attr
            Name of the attribute of the operator on the node.
        retrieve
            If attribute is output, this can be set to True to receive the handle to the output variable, or to false
            to receive the name of the output variable.

        Returns
        -------
        Any
            Operator attribute.

        """

        if node in self.label_map:
            node, attr_idx = self.label_map[node]
            idx = f"{list(attr_idx)}" if type(attr_idx) is tuple else attr_idx
            return self._backend.apply_idx(self._get_op_attr(node, op, attr)['value'], idx) if retrieve else attr_idx
        elif node in self.nodes:
            op = self[node]['op_graph'].nodes[op]
        else:
            raise ValueError(f'Node with name {node} is not part of this network.')

        if attr == 'output' and retrieve:
            attr = op['output']
        if attr in op['variables']:
            attr_val = op['variables'][attr]
        else:
            try:
                attr_val = op[attr]
            except KeyError as e:
                try:
                    attr_val = getattr(op, attr)
                except AttributeError:
                    raise e

        return attr_val

    def run(self,
            simulation_time: Optional[float] = None,
            inputs: Optional[dict] = None,
            outputs: Optional[dict] = None,
            sampling_step_size: Optional[float] = None,
            out_dir: Optional[str] = None,
            verbose: bool = True,
            profile: Optional[str] = None,
            **kwargs
            ) -> Union[DataFrame, Tuple[DataFrame, float, float]]:
        """Simulate the backend behavior over time via a tensorflow session.

        Parameters
        ----------
        simulation_time
            Simulation time in seconds.
        inputs
            Inputs for placeholder variables. Each key is a tuple that specifies a placeholder variable in the graph
            in the following format: (node_name, op_name, var_name). Each value is an array that defines the input for
            the placeholder variable over time (first dimension).
        outputs
            Output variables that will be returned. Each key is the desired name of an output variable and each value is
            a tuple that specifies a variable in the graph in the following format: (node_name, op_name, var_name).
        sampling_step_size
            Time in seconds between sampling points of the output variables.
        out_dir
            Directory in which to store outputs.
        verbose
            If true, status updates will be printed to the console.
        profile
            Can be used to extract information about graph execution time and memory load. Can be:
            - `t` for returning the total graph execution time.
            - `m` for returning the peak memory consumption during graph excecution.
            - `mt` or `tm` for both

        Returns
        -------
        Union[DataFrame, Tuple[DataFrame, float, float]]
            First entry of the tuple contains the output variables in a pandas dataframe, the second contains the
            simulation time in seconds and the third the peak memory consumption. If profiling was not chosen during
            call of the function, only the dataframe will be returned.

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
        if not simulation_time:
            simulation_time = self._backend.dt
        sim_steps = int(simulation_time / self._dt)

        if not sampling_step_size:
            sampling_step_size = self._dt
        sampling_steps = int(sampling_step_size / self._dt)

        # add output variables to the backend
        #####################################

        output_col = {}
        output_cols = []
        output_keys = []
        output_nodes = []
        output_shapes = []

        if outputs:

            # go through passed output names
            for key, val in outputs.items():

                # extract respective output variables from the network and store their information
                for var_key, var_info in self.get_node_var(val).items():
                    var_shape = tuple(var_info['var'].shape)
                    if var_shape in output_shapes:
                        idx = output_shapes.index(var_shape)
                        output_cols[idx].append(var_info['var'])
                        output_keys[idx].append(key)
                        output_nodes[idx].append(var_info['nodes'])
                    else:
                        output_cols.append([var_info['var']])
                        output_keys.append([key])
                        output_nodes.append([var_info['nodes']])
                        output_shapes.append(var_shape)

                # parse output storage operation into backend
                output_col.update(self._backend.add_output_layer(outputs=output_cols,
                                                                 sampling_steps=int(sim_steps / sampling_steps),
                                                                 out_shapes=output_shapes))

        # add input variables to the backend
        ####################################

        if inputs:

            input_col = dict()

            # go through passed inputs
            for key, val in inputs.items():

                in_shape = val.shape[1]
                key_split = key.split('/')
                op, var = key_split[-2], key_split[-1]

                # extract respective input variable from the network
                # TODO: this needs to be adjusted such that the backend knows which parts of the backend variables to
                # TODO: project to
                for var_key, var_info in self.get_node_var(key, apply_idx=False).items():
                    var_shape = len(var_info['idx'])
                    var_idx = var_info['idx'] if np.sum(var_shape) > 1 else None
                    if var_shape == in_shape:
                        input_col[var_info['var']] = (val, var_idx)
                    elif (var_shape % in_shape) == 0:
                        input_col[var_info['var']] = (np.tile(val, (1, var_shape)), var_idx)
                    else:
                        input_col[var_info['var']] = (np.reshape(val, (sim_steps,) + var_shape), var_idx)

            self._backend.add_input_layer(inputs=input_col)

        # run simulation
        ################

        if verbose:
            print("Running the simulation...")

        if profile is None:
            output_col = self._backend.run(steps=sim_steps, outputs=output_col, sampling_steps=sampling_steps,
                                           out_dir=out_dir, profile=profile, **kwargs)
        else:
            output_col, time, memory = self._backend.run(steps=sim_steps, outputs=output_col, out_dir=out_dir,
                                                         profile=profile, sampling_steps=sampling_steps, **kwargs)

        if verbose and profile:
            if simulation_time:
                print(f"{simulation_time}s of backend behavior were simulated in {time} s given a "
                      f"simulation resolution of {self._dt} s.")
            else:
                print(f"ComputeGraph computations finished after {time} seconds.")
        elif verbose:
            print('finished!')

        # store output variables in data frame
        ######################################

        # ungroup grouped output variables
        outputs = {}
        levels = 0
        for names, group_key, nodes in zip(output_keys, output_col, output_nodes):
            out_group = output_col[group_key]
            for i, (key, node_keys) in enumerate(zip(names, nodes)):
                out_val = np.squeeze(out_group[:, i])
                if len(out_val.shape) == 1:
                    if levels > 0:
                        nulls = tuple([None for _ in range(levels)])
                        outputs[(key,) + nulls] = out_val
                    else:
                        outputs[key] = out_val
                else:

                    for j, node_key in enumerate(node_keys):
                        out_val_tmp = np.squeeze(out_val[:, j])
                        if len(out_val_tmp.shape) == 1:
                            if levels > 1:
                                outputs[(key, node_key, None)] = out_val
                            else:
                                outputs[(key, node_key)] = out_val_tmp
                                levels = 1
                        else:
                            for k in range(out_val_tmp.shape[1]):
                                outputs[(key, node_key, str(k))] = np.squeeze(out_val_tmp[:, k])
                                levels = 2

        # create data frame
        out_vars = DataFrame(outputs)

        # return results
        ################

        if profile:
            return out_vars, time, memory
        return out_vars

    def compile(self,
                dt: float = 1e-3,
                vectorization: bool = True,
                build_in_place: bool = True,
                backend: str = 'numpy',
                solver: str = 'euler',
                float_precision: str = 'float32',
                **kwargs
                ) -> AbstractBaseIR:
        """Parses IR into the backend.
        """

        filterwarnings("ignore", category=FutureWarning)

        # set basic attributes
        ######################

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

        # set time constant of the network
        self._dt = dt
        dt = parse_dict({'dt': {'vtype': 'constant', 'dtype': float_precision, 'shape': (), 'value': self._dt}},
                        backend=self._backend)['dt']

        # run graph optimization and vectorization
        self._first_run = True
        self.optimize_graph_in_place(vectorize=vectorization)

        # move edge operations to nodes
        ###############################

        print('building the compute graph...')

        # create equations and variables for each edge
        for source_node, target_node, edge_idx, data in self.edges(data=True, keys=True):

            # extract edge information
            weight = data['weight']
            delay = data['delay']
            sidx = data['source_idx']
            tidx = data['target_idx']
            svar = data['source_var']
            sop, svar = svar.split("/")
            sval = self[f"{source_node}/{sop}/{svar}"]

            tvar = data['target_var']
            top, tvar = tvar.split("/")
            # get variable properties
            # tval --> variable properties
            # fetch both values and variable definitions of target variable
            tval = self[f"{target_node}/{top}/{tvar}"]

            add_project = data.get('add_project', False)  # get a False, in case it is not defined
            target_node_ir = self[target_node]

            # define target index
            if delay is not None and tidx:
                tidx_tmp = []
                for idx, d in zip(tidx, delay):
                    if type(idx) is list:
                        tidx_tmp.append(idx + [d])
                    else:
                        tidx_tmp.append([idx, d])
                tidx = tidx_tmp
            elif delay is not None:
                tidx = list(delay)

            # create mapping equation and its arguments
            d = "[target_idx]" if tidx else ""
            idx = "[source_idx]" if sidx else ""
            assign = '+=' if add_project else '='
            eq = f"{tvar}{d} {assign} {svar}{idx} * weight"
            args = {}
            dtype = sval["dtype"]
            args['weight'] = {'vtype': 'constant', 'dtype': dtype, 'value': weight}
            if tidx:
                args['target_idx'] = {'vtype': 'constant', 'dtype': 'int32',
                                      'value': np.array(tidx, dtype=np.int32)}
            if sidx:
                args['source_idx'] = {'vtype': 'constant', 'dtype': 'int32',
                                      'value': np.array(sidx, dtype=np.int32)}
            args[tvar] = tval

            # add edge operator to target node
            op_name = f'edge_from_{source_node}_{edge_idx}'
            target_node_ir.add_op(op_name,
                                  inputs={svar: {'sources': [sop],
                                                 'reduce_dim': True,
                                                 'node': source_node}},
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

        variables = {'all/all/dt': dt}

        # edge operators
        equations, variables_tmp = self._collect_op_layers(layers=[0], exclude=False, op_identifier="edge_from_")
        variables.update(variables_tmp)
        if equations:
            self._backend._input_layer_added = True

        # node operators
        equations_tmp, variables_tmp = self._collect_op_layers(layers=[], exclude=True, op_identifier="edge_from_")
        variables.update(variables_tmp)

        # bring equations into correct order
        equations = sort_equations(edge_eqs=equations, node_eqs=equations_tmp)

        # parse all equations and variables into the backend
        ####################################################

        self._backend.bottom_layer()

        # parse mapping
        variables = parse_equation_system(equations=equations, equation_args=variables, backend=self._backend,
                                          solver=solver)

        # save parsed variables in net config
        for key, val in variables.items():
            node, op, var = key.split('/')
            if "inputs" not in var and var != "dt":
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

                    for i, in_op in enumerate(inp['sources']):

                        # collect single input to op
                        in_var = self[f"{in_node}/{in_op}"]['output']
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

    def _add_edge_buffer(self, node: str, op: str, var: str, idx: int, buffer_length: int, edge: tuple) -> None:
        """Adds a buffer variable to an edge.

        Parameters
        ----------
        node
            Name of the target node of the edge.
        op
            Name of the target operator of the edge.
        var
            Name of the target variable of the edge.
        idx
            Index of the buffer variable for that specific edge.
        buffer_length
            Length of the time-buffer that should be added to realize edge delays.
        edge
            Edge identifier (source_name, target_name, edge_idx).

        Returns
        -------
        None

        """

        target_shape = self._get_op_attr(node, op, var)['shape']
        node_ir = self[node]

        # create buffer variable definitions
        if len(target_shape) < 1 or (len(target_shape) == 1 and target_shape[0] == 1):
            buffer_shape = (buffer_length + 1,)
            buffer_shape_reset = (1,)
        else:
            buffer_shape = (target_shape[0], buffer_length + 1)
            buffer_shape_reset = (target_shape[0], 1)
        var_dict = {f'{var}_buffer_{idx}': {'vtype': 'state_var',
                                            'dtype': self._backend._float_def,
                                            'shape': buffer_shape,
                                            'value': 0.
                                            },
                    f'{var}_buffer_{idx}_reset': {'vtype': 'constant',
                                                  'dtype': self._backend._float_def,
                                                  'shape': buffer_shape_reset,
                                                  'value': 0.
                                                  }
                    }

        # create buffer equations
        if len(target_shape) < 1 or (len(target_shape) == 1 and target_shape[0] == 1):
            eqs_op_read = [f"{var} = {var}_buffer_{idx}[0]"]
            eqs_op_rotate = [f"{var}_buffer_{idx} = concat(({var}_buffer_{idx}[1:], {var}_buffer_{idx}_reset), 0)"]
        else:
            eqs_op_read = [f"{var} = {var}_buffer_{idx}[:, 0]"]
            eqs_op_rotate = [f"{var}_buffer_{idx} = concat(({var}_buffer_{idx}[:, 1:], {var}_buffer_{idx}_reset), 1)"]

        # add buffer operators to operator graph
        node_ir.add_op(f'{op}_{var}_buffer_rotate_{idx}',
                       inputs={},
                       output=f'{var}_buffer_{idx}',
                       equations=eqs_op_rotate,
                       variables=var_dict)
        node_ir.add_op(f'{op}_{var}_buffer_read_{idx}',
                       inputs={f'{var}_buffer_{idx}': {'sources': [f'{op}_{var}_buffer_rotate_{idx}'],
                                                       'reduce_dim': False}},
                       output=var,
                       equations=eqs_op_read,
                       variables={})

        # connect operators to rest of the graph
        node_ir.add_op_edge(f'{op}_{var}_buffer_rotate_{idx}', f'{op}_{var}_buffer_read_{idx}')
        node_ir.add_op_edge(f'{op}_{var}_buffer_read_{idx}', op)

        # add input information to target operator
        inputs = self._get_op_attr(node, op, 'inputs')
        if var in inputs.keys():
            inputs[var]['sources'].add(f'{op}_{var}_buffer_read_{idx}')
        else:
            inputs[var] = {'sources': {f'{op}_{var}_buffer_read_{idx}'},
                           'reduce_dim': True}

        # update edge information
        s, t, e = edge
        self.edges[s, t, e]['target_var'] = f'{op}_{var}_buffer_rotate_{idx}/{var}_buffer_{idx}'
        self.edges[s, t, e]['add_project'] = True

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

        target_shape = self._get_op_attr(node, op, var)['shape']
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
        op_inputs = self._get_op_attr(node, op, 'inputs')
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
    for node_layer in node_eqs.copy():
        if not any([is_diff_eq(eq) for eq, _ in node_layer]):
            eqs_new.append(node_layer)
            node_eqs.pop(node_eqs.index(node_layer))

    eqs_new += edge_eqs
    eqs_new += node_eqs

    return eqs_new
