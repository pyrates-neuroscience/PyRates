"""Basic neural mass model class plus derivations of it.

This module includes the base circuit class that manages the set-up and simulation of population networks. Additionally,
it features various sub-classes that act as circuit constructors and allow to build circuits based on different input
arguments. For more detailed descriptions, see the respective docstrings.

"""

# external packages
import re
from typing import List, Dict, Union
from networkx import MultiDiGraph, subgraph
from copy import deepcopy
from pandas import DataFrame

# pyrates internal imports
from pyrates.frontend.abc import AbstractBaseTemplate, AbstractBaseIR
from pyrates.frontend.edge import EdgeTemplate, EdgeIR
from pyrates.frontend.node import NodeTemplate, NodeIR
from pyrates import PyRatesException
from pyrates.frontend.operator import OperatorIR
from pyrates.frontend.yaml_parser import TemplateLoader

# from pyrates.frontend.operator import OperatorTemplate

# meta infos
__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


class CircuitIR(AbstractBaseIR):
    """Custom graph data structure that represents a backend of nodes and edges with associated equations
    and variables."""

    def __init__(self, label: str = "circuit", circuits: dict = None, nodes: dict = None,
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
        self.label_map = {}
        self.template = template

        self.graph = MultiDiGraph()
        self.sub_circuits = set()

        if circuits:
            for key, temp in circuits.items():
                self.add_circuit(key, temp)

        if nodes:
            self.add_nodes_from(nodes, from_templates=True)

        if edges:
            self.add_edges_from(edges)

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
        # update circuit-wide label map, assuming all labels are now unique
        self.label_map.update(label_map)

    def add_node(self, label: str, template: NodeTemplate = None,
                 node: NodeIR = None, **attr) -> str:
        """Add single node"""

        new_label = self._get_unique_label(label)

        if template:
            node = template.apply()
        self.graph.add_node(new_label, node=node, **attr)

        self.label_map[label] = new_label

    def add_edges_from(self, edges, **attr):
        """ Add multiple edges. This method explicitly assumes, that edges are given in edge_templates instead of
        existing instances of `EdgeIR`.

        Parameters
        ----------
        edges
            list of edges, each of shape [source/op/var, target/op/var, edge_template, variables]
        attr

        Returns
        -------
        """

        edge_list = []
        for (source, target, template, values) in edges:
            # get edge template and instantiate it
            values = deepcopy(values)
            weight = values.pop("weight", 1)
            # get delay
            delay = values.pop("delay", 0)

            edge_ir = template.apply(values=values)  # type: EdgeIR # edge spec
            # get weight

            # ToDo: Implement source/op/var syntax
            # take apart source and target strings
            # for circuit from circuit:
            # for ... in circuits
            # for ... in nodes...

            # test, if variables at source and target exist and reference them properly
            source, target = self._identify_sources_targets(source, target, edge_ir)

            edge_list.append((source[0], target[0],  # edge_unique_key,
                              {"edge_ir": edge_ir,
                               # "target_operator": target_operator,
                               "weight": weight,
                               "delay": delay,
                               "source_var": "/".join(source[-2:]),
                               "target_var": "/".join(target[-2:])
                               }))
        self.graph.add_edges_from(edge_list, **attr)

    def add_edge(self, source: str, target: str, template: EdgeTemplate=None,
                 values: dict=None, identify_relations=True,
                 **data):
        """

        Parameters
        ----------
        source
        target
        template
            An instance of `EdgeTemplate` that will be applied using `values` to get an `EdgeIR` instance. If a template
            is given, any 'edge_ir' that is additionally passed, will be ignored.
        values
        data
            If no template is given, `data` is assumed to conform to the format that is needed to add an edge. I.e.,
            `data` needs to contain fields for `weight`, `delay`, `edge_ir`, `source_var`, `target_var`.
        identify_relations

        Returns
        -------

        """

        if template:
            weight = values.pop("weight", 1.)
            delay = values.pop("delay", 0.)
            values_to_update = deepcopy(values)
            edge_ir = template.apply(values_to_update)

            if identify_relations:
                # test, if variables at source and target exist and reference them properly
                source, target = self._identify_sources_targets(source, target, edge_ir)

            else:
                # assume that source and target strings are already consistent. This should be the case,
                # if the given strings were coming from existing circuits (instances of `CircuitIR`)
                # or in general, if operators are not renamed.
                source = source.split("/")
                target = target.split("/")

            source_node = "/".join(source[:-2])
            target_node = "/".join(target[:-2])
            source_var = "/".join(source[-2:])
            target_var = "/".join(target[-2:])

            self.graph.add_edge(source_node, target_node,
                                {"edge_ir": edge_ir,
                                 "weight": weight,
                                 "delay": delay,
                                 "source_var": source_var,
                                 "target_var": target_var})

        else:
            for path in (source, target):
                if path not in self:
                    raise PyRatesException(f"Failed to add edge, because referenced node `{path}` does not exist in "
                                           f"network graph. Edges can only be added to existing nodes.")
            self.graph.add_edge(source, target, **data)

    def _get_unique_label(self, label: str) -> str:
        """

        Parameters
        ----------
        label

        Returns
        -------
        unique_label
        """
        # test, if label already has a counter and separate it, if necessary
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

    # noinspection PyUnresolvedReferences
    def _identify_sources_targets(self, source: str, target: str,
                                  edge_ir: EdgeIR):

        input_var = edge_ir.input
        output_var = edge_ir.output
        # separate source and target specifiers
        *source_node, source_op, source_var = source.split("/")
        source_node = "/".join(source_node)
        *target_node, target_op, target_var = target.split("/")
        target_node = "/".join(target_node)

        # re-reference node labels, if necessary
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
        Parameters
        ----------
        node_label
        op_label

        Returns
        -------
        new_op_label

        """
        # ToDo: reformat OperatorIR --> Operator label remains true to template and version number is stored internally
        key_counter = OperatorIR.key_counter
        if op_label in key_counter:
            for i in range(key_counter[op_label]):
                new_op_label = f"{op_label}.{i+1}"
                try:
                    self[node_label][new_op_label]
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
        return cls(template.label, template.nodes, template.edges, template.path)

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
            as (string of form 'circuit/node/op/var') and items may then be edge templates and associated variables.
            Empty cells in the DataFrame should be filled with something 'falsy' (as in evaluates to `False` in Python).

        Returns
        -------
        circuit
            instance of `CircuitIR`
        """

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
            self.add_edge(f"{label}/{source}", f"{label}/{target}", **data)

    def network_def(self, revert_node_names=False):
        from pyrates.frontend.circuit.utility import BackendIRFormatter
        return BackendIRFormatter.network_def(self, revert_node_names=revert_node_names)


class CircuitTemplate(AbstractBaseTemplate):

    def __init__(self, name: str, path: str, description: str, label: str = "circuit",
                 circuits: dict =None, nodes: dict = None, edges: List[tuple] = None):

        super().__init__(name, path, description)

        self.nodes = {}
        if nodes:
            for key, path in nodes.items():
                if isinstance(path, str):
                    path = self._format_path(path)
                    self.nodes[key] = NodeTemplate.from_yaml(path)

        self.circuits = {}
        if circuits:
            for key, path in circuits.items():
                if isinstance(path, str):
                    path = self._format_path(path)
                    self.circuits[key] = CircuitTemplate.from_yaml(path)

        if edges:
            self.edges = self._get_edge_templates(edges)
        else:
            self.edges = []

        self.label = label

    def apply(self, label: str = None):
        """Create a Circuit graph instance based on the template"""

        if not label:
            label = self.label

        return CircuitIR(label, self.circuits, self.nodes, self.edges, self.path)

    def _get_edge_templates(self, edges: List[Union[tuple, dict]]):
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
            if isinstance(edge, dict):
                edge = (edge["source"], edge["target"], edge["template"], edge["variables"])
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
                        label: str = None, circuits: dict = None, nodes: dict = None,
                        edges: List[tuple] = None):
        """Update all entries of the circuit template in their respective ways."""

        if not description:
            description = base.__doc__

        if not label:
            label = base.label

        if nodes:
            nodes = cls.update_dict(base.nodes, nodes)
        else:
            nodes = base.nodes

        if circuits:
            circuits = cls.update_dict(base.circuits, circuits)
        else:
            circuits = base.circuits

        if edges:
            edges = cls.update_edges(base.edges, edges)
        else:
            edges = base.edges

        return CircuitTemplate(name=name, path=path, description=description,
                               label=label, circuits=circuits, nodes=nodes,
                               edges=edges)

    @staticmethod
    def update_dict(base_dict: dict, updates: dict):

        updated = deepcopy(base_dict)

        updated.update(updates)

        return updated

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
