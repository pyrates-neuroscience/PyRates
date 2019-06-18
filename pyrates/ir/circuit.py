
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
import re
from copy import deepcopy
from typing import Union, Dict, Iterator

from pyparsing import Word, ParseException, nums, Literal, LineEnd, Suppress, alphanums
from networkx import MultiDiGraph, subgraph, find_cycle, NetworkXNoCycle, DiGraph
from pandas import DataFrame

from pyrates import PyRatesException
from pyrates.ir.node import NodeIR
from pyrates.ir.edge import EdgeIR
from pyrates.ir.abc import AbstractBaseIR

__author__ = "Daniel Rose"
__status__ = "Development"


class CircuitIR(AbstractBaseIR):
    """Custom graph data structure that represents a backend of nodes and edges with associated equations
    and variables."""

    # _node_label_grammar = Word(alphanums+"_") + Suppress(".") + Word(nums)

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
        self.label_counter = {}
        self.label_map = {}

        self.graph = MultiDiGraph()
        self.sub_circuits = set()

        if circuits:
            for key, temp in circuits.items():
                self.add_circuit(key, temp)

        if nodes:
            self.add_nodes_from(nodes)

        if edges:
            self.add_edges_from(edges)

    def add_nodes_from(self, nodes: Dict[str, NodeIR], **attr):
        """ Add multiple nodes to circuit. Allows networkx-style adding if from_templates is set to False.

        Parameters
        ----------
        nodes
            Dictionary with node label as key. The item is a NodeIR instance. Note that the item type is not tested
            here, but passing anything that does not behave like a `NodeIR` may cause problems later.
        attr
            additional keyword attributes that can be added to the node data. (default `networkx` syntax.)
        """

        # get unique labels for nodes
        for label in nodes:
            self.label_map[label] = self._get_unique_label(label)

        # rename node keys
        # assign NodeIR instances as "node" keys in a separate dictionary, because networkx saves node attributes into
        # a dictionary
        # reformat dictionary to tuple/generator, since networkx does not parse dictionary correctly in add_nodes_from
        nodes = ((self.label_map[key], {"node": node}) for key, node in nodes.items())
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

        new_label = self._get_unique_label(label)

        self.graph.add_node(new_label, node=node, **attr)

        self.label_map[label] = new_label

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
            edge_ir = edge_dict.get("edge_ir", EdgeIR())

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
        self.graph.add_edges_from(edge_list, **attr)

    def add_edge(self, source: str, target: str, edge_ir: EdgeIR, weight: float = 1., delay: float = None,
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

    def _get_unique_label(self, label: str) -> str:
        """Tests, if a given node `label` already exists in the circuit and renames it uniquely, if necessary.
        Uniqueness is generally ensure by appending a counter of the form ".0" . If the given label already has a
        counter, it will be detected ond potentially altered if uniqueness requires it. The resulting (new) label is
        returned.

        Parameters
        ----------
        label

        Returns
        -------
        unique_label
        """
        # test, if label already has a counter and separate it, if necessary
        # could use pyparsing instead of regex, but actually seems more robust and simple enough in this case
        match = re.match("(.+)[.]([\d]+$)", label)
        if match:
            # see if label already exists, just continue if it doesn't
            if label in self:
                # create new label
                # separate base label and counter
                label, counter = match.groups()
                counter = int(counter)
                if label not in self.label_counter:
                    self.label_counter[label] = counter + 1
                elif counter > self.label_counter[label]:
                    self.label_counter[label] = counter + 1
                else:
                    self.label_counter[label] += 1
                # set label
                unique_label = f"{label}.{self.label_counter[label]}"
            else:
                # pass original label
                unique_label = label

        else:
            # define counter
            if label in self.label_counter:
                self.label_counter[label] += 1
            else:
                self.label_counter[label] = 0

            # set label
            unique_label = f"{label}.{self.label_counter[label]}"

        return unique_label

    def _validate_separate_key_path(self, *paths: str):

        for key in paths:

            # (circuits), node, operator and variable specifiers

            *node, op, var = key.split("/")

            node = "/".join(node)

            # re-reference node labels, if necessary
            # this syntax yields "node" back as default if it is not in label_map
            node = self.label_map.get(node, node)
            # re_reference operator labels, if necessary
            op = self._validate_rename_op_label(self[node], op)
            # ignore circuits for now
            # note: current implementation assumes, that this method is only called, if an edge is added
            path = "/".join((node, op, var))
            # check if path is valid
            if path not in self:
                raise PyRatesException(f"Could not find object with key path `{path}`.")

            separated = (node, op, var)
            yield separated

    @staticmethod
    def _validate_rename_op_label(node: NodeIR, op_label: str) -> str:
        """
        References to operators in source/target references may mismatch actual operator labels, due to internal renaming
        that can not be accounted for in the YAML templates. This method looks for the first instance of an operator in
        a specific node, assuming it exists at all. Additionally, this assumes, that only one variation of an operator
        template (of same name) can exist in any single node.
        Note: The latter assumption becomes invalid, if edge operators are moved to nodes, because in that case multiple
        variations of the same operator can exist in one node.

        Parameters
        ----------
        node
        op_label

        Returns
        -------
        new_op_label

        """
        # variant 1: label already has a counter --> might actually already exist.

        if "." in op_label:
            try:
                _ = node[op_label]
            except KeyError:
                op_label, *_ = op_label.split(".")
            else:
                return op_label

        # build grammar for pyparsing
        grammar = Literal(op_label) + "." + Word(nums)
        found = 0  # count how many matching operators were found
        for op_key in node:
            try:
                grammar.parseString(op_key, parseAll=True)
            except ParseException:
                continue
            else:
                op_label = op_key
                found += 1

        if found == 1:
            return op_label
        elif not found:
            raise PyRatesException(f"Could not identify operator with base name '{op_label}' "
                                   f"in node `{node}`.")
        else:
            raise PyRatesException(f"Unable to uniquely identify operator key. "
                                   f"Found multiple occurrences for operator with base name '{op_label}'.")

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

    def move_edge_operators_to_nodes(self, copy_data=True):
        """Returns a new CircuitIR instance, where all operators that were previously in edges are moved to their
        respective target nodes."""
        # if copy_data:
        #    nodes = {key: deepcopy(data) for key, data in self.nodes(data=True)}
        # else:
        #    nodes = {key: data for key, data in self.nodes(data=True)}

        # this does not preserve additional node attributes
        # node_attrs = {}
        # for key, data in nodes.items():
        #    nodes[key] = data["node"]
        # node_attrs.update(**data)

        # node_attrs.pop("node")

        op_label_counter = {}

        # edges = []
        for source, target, data in self.edges(data=True):

            if "edge_ir" in data and data["edge_ir"] and data["edge_ir"].op_graph:
                source_var = data["source_var"]
                target_var = data["target_var"]
                weight = data["weight"]
                delay = data["delay"]

                edge_ir = data["edge_ir"]  # type: EdgeIR
                op_graph = edge_ir.op_graph
                if copy_data:
                    op_graph = deepcopy(op_graph)
                input_var = edge_ir.input
                output_var = edge_ir.output
                if len(op_graph) > 0:
                    target_var = self._move_ops_to_target(target, input_var, output_var, target_var, op_graph,
                                                          op_label_counter, self.nodes)
                    # side effect: changes op_label_counter and nodes dictionary

                data.update({'source_var': source_var, 'target_var': target_var, 'edge_ir': EdgeIR(), 'weight': weight,
                             'delay': delay})
            # edges.append((source, target, data))

        # circuit = CircuitIR()
        # circuit.sub_circuits = self.sub_circuits
        # circuit.add_nodes_from(nodes)
        # circuit.add_edges_from(edges)
        # return circuit

        return self

    @staticmethod
    def _move_ops_to_target(target, input_var, output_var, target_var, op_graph: DiGraph, op_label_counter, nodes):
        # check, if cycles are present in operator graph (which would be problematic
        try:
            find_cycle(op_graph)
        except NetworkXNoCycle:
            pass
        else:
            raise PyRatesException("Found cyclic operator graph. Cycles are not allowed for operators within one node.")

        target_node = nodes[target]["node"]
        if target not in op_label_counter:
            op_label_counter[target] = {}

        key_map = {}
        # first pass: get all labels right
        for key in op_graph.nodes:
            # rename operator key by appending another counter
            if key in op_label_counter[target]:
                # already renamed it previously --> increase counter
                op_label_counter[target][key] += 1
                key_map[key] = f"{key}.{op_label_counter[target][key]}"

            else:
                # not renamed previously --> initialize counter to 0
                op_label_counter[target][key] = 0
                key_map[key] = f"{key}.0"

        # second pass: add operators to target op graph and rename sources according to key_map
        for key, data in op_graph.nodes(data=True):
            op = data["operator"]
            variables = data["variables"]
            # go through all input variables and rename source operators
            for var, var_data in op.inputs.items():
                sources = []
                for source_op in var_data["sources"]:
                    sources.append(key_map[source_op])
                op.inputs[var]["sources"] = sources

            # add operator to target_node's operator graph
            # TODO: implement add_node method on OperatorGraph class
            target_node.op_graph.add_node(key_map[key], operator=op, variables=variables)

        # add edges that previously existed
        for source_op, target_op in op_graph.edges:
            # rename edge keys accordingly
            source_op = key_map[source_op]
            target_op = key_map[target_op]

            target.op_graph.add_edge(source_op, target_op)

        # add new edges based on output and target of edge
        target_op, target_var = target_var.split("/")
        output_op, output_var = output_var.split("/")
        output_op = key_map[output_op]

        target_node[target_op].inputs[target_var]["sources"].append(output_op)

        # make sure the changes are saved (might not be necessary)
        # nodes[target] = target_node

        # reassign target variable of edge
        target_op, target_var = input_var.split("/")
        return f"{key_map[target_op]}/{target_var}"


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
