"""
"""
import re
from copy import deepcopy
from typing import Union, Dict

from networkx import MultiDiGraph, subgraph
from pandas import DataFrame

from pyrates import PyRatesException
from pyrates.ir.node import NodeIR
from pyrates.ir.edge import EdgeIR
from pyrates.ir.operator import OperatorIR
from pyrates.ir.abc import AbstractBaseIR

__author__ = "Daniel Rose"
__status__ = "Development"


class CircuitIR(AbstractBaseIR):
    """Custom graph data structure that represents a backend of nodes and edges with associated equations
    and variables."""

    def __init__(self, label: str = "circuit", circuits: dict = None, nodes: Dict[str, NodeIR] = None,
                 edges: list = None, template: str = None):
        """
        Parameters:
        -----------
        label
            string label, used if circuit is part of other circuits
            ToDo: check, if this is actually used in practise
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

        self.label = label
        self.label_counter = {}
        self.label_map = {}
        self.template = template

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
            weight = edge_dict.pop("weight", 1.)
            # get delay
            delay = edge_dict.pop("delay", None)

            edge_ir = edge_dict["edge_ir"]
            # ToDo: Implement default for empty edges without operators --> no need to provide edge IR

            # test, if variables at source and target exist and reference them properly
            source, target = self._identify_sources_targets(source, target)

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

        # ToDo: streamline code by removing duplications for source and target
        source_var = ""
        target_var = ""
        if identify_relations:
            # test, if variables at source and target exist and reference them properly
            source, target = self._identify_sources_targets(source, target)
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
        TODO: cache assigned labels directly in label_map instead of solely returning it (may still want to return it
         though)

        Parameters
        ----------
        label

        Returns
        -------
        unique_label
        """
        # test, if label already has a counter and separate it, if necessary
        match = re.match("(.+)[.]([\d]+$)", label)
        # ToDo: use pyparsing instead of regex for readability
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

    def _identify_sources_targets(self, source: str, target: str):

        # separate source and target specifiers
        # TODO: streamline code by looping over source and target instead of duplicating the code.
        #  possibly even separate both into separate function calls to the same (more generic) function
        *source_node, source_op, source_var = source.split("/")
        source_node = "/".join(source_node)
        *target_node, target_op, target_var = target.split("/")
        target_node = "/".join(target_node)

        # re-reference node labels, if necessary
        # TODO: test first, if label actually exists in label map (which has always been the case so far).
        source_node = self.label_map[source_node]
        target_node = self.label_map[target_node]
        # re_reference operator labels, if necessary
        source_op = self._rename_operator(source_node, source_op)
        target_op = self._rename_operator(target_node, target_op)

        # ignore circuits for now
        # note: current implementation assumes, that this method is only called, if an edge is added
        source_path = "/".join((source_node, source_op, source_var))
        target_path = "/".join((target_node, target_op, target_var))

        # check if path is valid
        for path in (source_path, target_path):
            if path not in self:
                raise PyRatesException(f"Could not find object with key path `{path}`.")

        source = (source_node, source_op, source_var)
        target = (target_node, target_op, target_var)
        return source, target

    def _rename_operator(self, node_label: str, op_label: str) -> str:
        """
        References to operators in source/target references may mismatch actual operator labels, due to internal renaming
        that can not be accounted for in the YAML templates. This method looks for the first instance of an operator in
        a specific node, assuming it exists at all. Additionally, this assumes, that only one variation of an operator
        template (of same name) can exist in any single node.
        Note: The latter assumption becomes invalid, if edge operators are moved to nodes, because in that case multiple
        variations of the same operator can exist in one node.

        Parameters
        ----------
        node_label
        op_label

        Returns
        -------
        new_op_label

        """
        # ToDo: reformat OperatorIR --> Operator label remains true to template and version number is stored internally
        if "." in op_label:
            try:
                _ = self[node_label][op_label]
            except KeyError:
                op_label, *_ = op_label.split(".")
            else:
                return op_label

        key_counter = OperatorIR.key_counter
        if op_label in key_counter:
            for i in range(key_counter[op_label] + 1):
                new_op_label = f"{op_label}.{i}"
                try:
                    _ = self[node_label][new_op_label]
                    break
                except KeyError:
                    continue
            else:
                raise PyRatesException(f"Could not identify operator with template name {op_label} "
                                       f"in node {node_label}.")
        else:
            new_op_label = f"{op_label}.0"

        return new_op_label

    def _getter(self, key):

        if key in self.sub_circuits:
            return SubCircuitView(self, key)
        else:
            return self.graph.nodes[key]["node"]

    @property
    def nodes(self):
        """Shortcut to self.graph.nodes. See documentation of `networkx.MultiDiGraph.nodes`."""
        return self.graph.nodes

    @property
    def edges(self):
        """Shortcut to self.graph.edges. See documentation of `networkx.MultiDiGraph.edges`."""
        return self.graph.edges

    @classmethod
    def from_template(cls, template, *args, **kwargs):
        return template.apply(*args, **kwargs)

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

        if connectivity:
            if isinstance(connectivity, list) or isinstance(connectivity, tuple):
                circuit.add_edges_from(connectivity)
            else:
                try:
                    for source, row in connectivity.iterrows():
                        for target, content in row.iteritems():
                            if content:  # assumes, empty entries evaluate to `False`
                                circuit.add_edge(source, target, identify_relations=False, **content)
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

        # add sub circuit label map items to local label map
        for old, new in circuit.label_map.items():
            self.label_map[f"{label}/{old}"] = f"{label}/{new}"

        # add edges
        for source, target, data in circuit.edges(data=True):
            # source_var = data.pop("source_var")
            # target_var = data.pop("target_var")
            self.add_edge(f"{label}/{source}", f"{label}/{target}", identify_relations=False, **data)

    def network_def(self, revert_node_names=True):
        from pyrates.frontend.circuit.utility import BackendIRFormatter
        return BackendIRFormatter.network_def(self, revert_node_names=revert_node_names)

    def to_dict(self):
        """Reformat graph structure into a dictionary that can be saved as YAML template."""

        node_dict = {}
        for node_key, node_data in self.nodes(data=True):
            node = node_data["node"]
            if node.template:
                node_dict[node_key] = node.template.path
            else:
                # if no template is given, build and search deeper for node templates
                pass

        edge_list = []
        for source, target, edge_data in self.edges(data=True):
            edge = edge_data.pop("edge_ir")
            source = f"{source}/{edge_data['source_var']}"
            target = f"{target}/{edge_data['target_var']}"
            edge_list.append((source, target, edge.template.path, dict(weight=edge_data["weight"],
                                                                       delay=edge_data["delay"])))

        # use Python template as base, since inheritance from YAML templates is ambiguous for circuits
        base = "CircuitTemplate"

        return dict(nodes=node_dict, edges=edge_list, base=base)


class SubCircuitView(AbstractBaseIR):
    """View on a subgraph of a circuit. In order to keep memory footprint and computational cost low, the original (top
    level) circuit is referenced locally as 'top_level_circuit' and all subgraph-related information is computed only
    when needed."""

    def __init__(self, top_level_circuit: CircuitIR, subgraph_key: str):

        self.top_level_circuit = top_level_circuit
        self.subgraph_key = subgraph_key

    def _getter(self, key: str):

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

    @classmethod
    def from_template(cls, template, **kwargs):
        raise NotImplementedError(f"{cls} does not implement from_template.")

    def __str__(self):

        return f"{self.__class__.__name__} on '{self.subgraph_key}' in {self.top_level_circuit}"
