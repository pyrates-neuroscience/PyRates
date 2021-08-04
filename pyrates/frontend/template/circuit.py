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
"""Basic neural mass backend class plus derivations of it.

This module includes the base circuit class that manages the set-up and simulation of population networks. Additionally,
it features various sub-classes that act as circuit constructors and allow to build circuits based on different input
arguments. For more detailed descriptions, see the respective docstrings.

"""

# external packages
from typing import List, Union, Dict, Optional
from copy import deepcopy

# pyrates internal imports
import numpy as np

from pyrates import PyRatesException
from pyrates.frontend.template._io import _complete_template_path
from pyrates.frontend.template.abc import AbstractBaseTemplate
from pyrates.frontend.template.edge import EdgeTemplate
from pyrates.frontend.template.node import NodeTemplate
from pyrates.frontend.template.operator import OperatorTemplate

# meta infos
from pyrates.ir.circuit import CircuitIR
from pyrates.ir.edge import EdgeIR

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


class CircuitTemplate(AbstractBaseTemplate):
    target_ir = CircuitIR

    def __init__(self, name: str, path: str, description: str = "A circuit template.", label: str = "circuit",
                 circuits: dict = None, nodes: dict = None, edges: List[tuple] = None):

        super().__init__(name, path, description)

        if nodes and circuits:
            raise ValueError('CircuitTemplate has been initialized with both sub-circuits and nodes. However, all '
                             'nodes in a circuit must have the same hierarchical depth. Please provide only a node or '
                             'only a circuit dictionary. Note that you can add redundant hierarchy levels to each '
                             'sub-circuit or node by constructing a CircuitTemplate that takes only a single other '
                             'CircuitTemplate as input.')

        self.nodes = {}  # type: Dict[str, NodeTemplate]
        if nodes:
            for key, node in nodes.items():
                if isinstance(node, str):
                    path = _complete_template_path(node, self.path)
                    self.nodes[key] = NodeTemplate.from_yaml(path)
                else:
                    self.nodes[key] = node

        self.circuits = {}
        if circuits:
            for key, circuit in circuits.items():
                if isinstance(circuit, str):
                    path = _complete_template_path(circuit, self.path)
                    self.circuits[key] = CircuitTemplate.from_yaml(path)
                else:
                    self.circuits[key] = circuit

        if edges:
            self.edges = self._load_edge_templates(edges)
        else:
            self.edges = []

        self.label = label

    def update_template(self, name: str = None, path: str = None, description: str = None,
                        label: str = None, circuits: dict = None, nodes: dict = None,
                        edges: List[tuple] = None):
        """Update all entries of the circuit template in their respective ways."""

        if nodes and circuits:
            raise ValueError('CircuitTemplate cannot use both sub-circuits and nodes, since all '
                             'nodes in a circuit must have the same hierarchical depth. Please provide only a node or '
                             'only a circuit dictionary. Note that you can add redundant hierarchy levels to each '
                             'sub-circuit or node by constructing a CircuitTemplate that takes only a single other '
                             'CircuitTemplate as input.')

        if not name:
            name = self.name

        if not path:
            path = self.path

        if not description:
            description = self.__doc__

        if not label:
            label = self.label

        if nodes:
            nodes = update_dict(self.nodes, nodes)
        else:
            nodes = self.nodes

        if circuits:
            circuits = update_dict(self.circuits, circuits)
        else:
            circuits = self.circuits

        if edges:
            edges = update_edges(self.edges, edges)
        else:
            edges = self.edges

        return self.__class__(name=name, path=path, description=description,
                              label=label, circuits=circuits, nodes=nodes,
                              edges=edges)

    def run(self, simulation_time: float, step_size: float, inputs: Optional[dict] = None,
            outputs: Optional[dict] = None, sampling_step_size: Optional[float] = None, solver: str = 'euler',
            backend: str = 'numpy', out_dir: Optional[str] = None, verbose: bool = True, profile: bool = False,
            apply_kwargs: dict = None, **kwargs):

        # TODO 3: CircuitIR.__init__() should integrate the CircuitIR.compile() method and be automatically invoked by calling
        #  apply
        # TODO 4: call CircuitIR.run()

        # add extrinsic inputs to network
        #################################

        adaptive_steps = is_integration_adaptive(solver, **kwargs)
        for target, in_array in inputs.items():

            *node_id, op, var = target.split('/')

            # get network nodes that input should be provided to
            target_nodes = self.get_nodes(node_id)

            # create node template that generates the input
            out_key, new_node = get_input_node(var, in_array, adaptive_steps, simulation_time)

            # add input node to network and connect it to target variables
            out_var = "/".join(out_key)
            new_edges = [(out_var, f"{n}/{op}/{var}", None, {'weight': 1.0}) for n in target_nodes]
            self.update_template(nodes={out_key[0]: new_node}, edges=new_edges)

        # translate circuit template into a graph representation
        ########################################################

        if not apply_kwargs:
            apply_kwargs = {}
        ir = self.apply(adaptive_steps=adaptive_steps, verbose=verbose, backend=backend, **apply_kwargs)

    def apply(self, adaptive_steps: bool, label: str = None, node_values: dict = None, edge_values: dict = None,
              vectorize: bool = True, verbose: bool = True, **kwargs):
        """Create a Circuit graph instance based on the template


        Parameters
        ----------
        adaptive_steps
        label
            (optional) Assign a label that is saved as a sort of name to the circuit instance. This is particularly
            relevant, when adding multiple circuits to a bigger circuit. This way circuits can be identified by their
            given label.
        node_values
            (optional) Dictionary containing values (and possibly other variable properties) that overwrite defaults in
            specific nodes/operators/variables. Values must be given in the form: {'node/op/var': value}
        edge_values
            (optional) Dictionary containing source and target variable pairs as items and value dictionaries as values
            (e.g. {('source/op1/var1', 'target/op1/var2'): {'weight': 0.3, 'delay': 1.0}}). Can be used to overwrite
            default values defined in template.
        vectorize
        verbose

        Returns
        -------

        """

        if not label:
            label = self.label
        if not edge_values:
            edge_values = {}

        # vectorize network if requested
        template, node_map = vectorize_circuit(deepcopy(self), vectorize)

        # turn nodes from templates into IRs
        ####################################

        # prepare node parameter updates for IR transformation
        values = dict()
        if node_values:
            for key, value in node_values.items():
                *node_id, op, var = key.split("/")
                target_nodes = self.get_nodes(node_id)
                for n in target_nodes:
                    try:
                        n, idx = node_map[n]
                        value = {'idx': idx, 'value': value}
                    except KeyError:
                        pass
                    if n not in values:
                        values[n] = dict()
                    values[n]["/".join((op, var))] = value

        # go through node templates and transform them into intermediate representations
        node_keys = template.get_nodes(['all'])
        nodes = {}
        for node in node_keys:
            updates = values[node] if node in values else {}
            node_template = template.get_node_template(node)
            nodes[node] = node_template.apply(values=updates)

        # reformat edge templates to EdgeIR instances
        edges = []
        for (source, target, edge_template, values) in template.edges:

            # get edge template and instantiate it
            values = deepcopy(values)

            # update edge template default values with passed edge values, if source and target are simple variable keys
            if type(source) is str and type(target) is str and (source, target) in edge_values:
                values.update(edge_values[(source, target)])
            weight = values.pop("weight", 1.)

            # get delay
            delay = values.pop("delay", None)
            spread = values.pop("spread", None)

            # get source/target indices
            source_idx = values.pop("source_idx", None)
            target_idx = values.pop("target_idx", None)

            # treat additional source assignments in values dictionary
            extra_sources = {}
            keys_to_remove = []
            for key, value in values.items():
                try:
                    _, _, _ = value.split("/")
                except AttributeError:
                    # not a string
                    continue
                except ValueError as e:
                    raise ValueError(f"Wrong format of source specifier. Expected form: `node/op/var`. "
                                     f"Actual form: {value}")
                else:
                    # was actually able to split that string? Then it is an additional source specifier.
                    # Let's treat it as such
                    try:
                        op, var = key.split("/")
                    except ValueError as e:
                        if e.args[0].startswith("not enough"):
                            # No operator specified: assume that it is a known input variable. If not, we will notice
                            # later.
                            var = key
                            # Need to remove the key, but need to wait until after the iteration finishes.
                            keys_to_remove.append(key)
                        else:
                            raise e
                    else:
                        # we know which variable in which operator, so let us set its type to "input"
                        values[key] = "input"

                    # assuming, the variable has been set to "input", we can omit any operator description and only
                    # pass the actual variable name
                    extra_sources[var] = value

            for key in keys_to_remove:
                values.pop(key)

            # treat empty dummy edge templates as not existent templates
            if edge_template and len(edge_template.operators) == 0:
                edge_template = None
            if edge_template is None:
                edge_ir = None
                if values:
                    # should not happen. Putting this just in case.
                    raise PyRatesException("An empty edge IR was provided with additional values. "
                                           "No way to figure out where to apply those values.")

            else:
                edge_ir = edge_template.apply(values=values)  # type: Optional[EdgeIR] # edge spec

            # check whether multiple source variables are defined
            try:
                # if source is a dictionary, pass on its values as source_var
                source_var = list(source.values())[0]  # type: dict
                source = list(source.keys())[0]
            except AttributeError:
                # no dictionary? only singular source definition present. go on as planned.
                edge_dict = dict(edge_ir=edge_ir,
                                 weight=weight,
                                 delay=delay,
                                 spread=spread)

            else:
                # oh source was indeed a dictionary. go pass source information as separate entry
                edge_dict = dict(edge_ir=edge_ir,
                                 weight=weight,
                                 delay=delay,
                                 spread=spread,
                                 source_var=source_var)

            # now add extra sources, if there are some
            if extra_sources:
                edge_dict["extra_sources"] = extra_sources
            if source_idx:
                edge_dict['source_idx'] = source_idx
            if target_idx:
                edge_dict['target_idx'] = target_idx

            edges.append((source, target,  # edge_unique_key,
                          edge_dict
                          ))

        # instantiate an intermediate representation of the circuit template
        return CircuitIR(label, nodes=nodes, edges=edges, verbose=verbose, adaptive_steps=adaptive_steps,
                         **kwargs)

    def get_nodes(self, node_identifier: Union[str, list]) -> list:
        """Extracts nodes from the CircuitTemplate that match the provided identifier.

        Parameters
        ----------
        node_identifier
            Can be a simple string or a list of strings. If the CircuitTemplate is a hierarchical circuit (composed of
            circuits itself), different list entries should refer to the different hierarchy levels. Alternatively,
            separation via slashes can be used if a string is provided.

        Returns
        -------
        list
            List of node keys that match the provided identifier. Each entry is a string that refers to a node of the
            circuit with circuit hierarchy levels separated via slashes.

        """

        if type(node_identifier) is str:
            node_identifier = node_identifier.split('/')
        net = self.circuits if self.circuits else self.nodes

        if len(node_identifier) == 1:

            # return target nodes of circuit based on single identifier
            if node_identifier[0] in net:
                return node_identifier
            if node_identifier[0] == 'all':
                nodes = []
                for key, node in net.items():
                    nodes.extend(collect_nodes(key, node))
                return nodes
            raise ValueError(f'Node with label {node_identifier[0]} could not be found in CircuitTemplate {self.name}.')

        else:

            # collect target nodes from circuit based on hierarchical identifier
            nodes = ['']
            for i, node_lvl in enumerate(node_identifier):
                if node_lvl == 'all':
                    nodes_tmp = []
                    for key, c in net.items():
                        nodes_tmp.extend(collect_nodes(key, c))
                else:
                    nodes_tmp = collect_nodes(node_lvl, net[node_lvl])

                # join hierarchical levels via slash notation
                nodes_new = []
                for n1 in nodes:
                    for n2 in nodes_tmp:
                        nodes_new.append("/".join((n1, n2)))
                nodes = nodes_new

            return nodes

    def get_edges(self, source: Union[str, list], target: Union[str, list]) -> list:
        """Extracts nodes from the CircuitTemplate that match the provided identifier.

        Parameters
        ----------
        source, target
            Can be a simple string or a list of strings. If the CircuitTemplate is a hierarchical circuit (composed of
            circuits itself), different list entries should refer to the different hierarchy levels. Alternatively,
            separation via slashes can be used if a string is provided.

        Returns
        -------
        list
            List of edge keys that match the provided identifier. Each entry is a tuple that includes the source and
            target variables as well as the edge template. Circuit hierarchy levels are separated via slashes in the
            variable names.

        """

        # extract all existing edges from circuit
        all_edges = self.collect_edges()

        # return those edges if requested
        if source == 'all' and target == 'all':
            return all_edges

        # extract source and target variable information
        if type(source) is list:
            source = "/".join(source)
        if type(target) is list:
            target = "/".join(target)

        *s_node, s_op, s_var = source.split('/')
        *t_node, t_op, t_var = source.split('/')

        # extract requested source and target nodes from circuit
        source_nodes = self.get_nodes(s_node)
        target_nodes = self.get_nodes(t_node)

        # create source and target variable identifiers
        source_ids = [f"{s}/{s_op}/{s_var}" for s in source_nodes]
        target_ids = [f"{t}/{t_op}/{t_var}" for t in target_nodes]

        # collect edges that match source and target variable requests
        return [(s, t, template, edge_dict) for s, t, template, edge_dict in all_edges
                if s in source_ids and t in target_ids]

    def get_node_template(self, node: Union[str, list]):
        """Extract NodeTemplate from CircuitTemplate.

        Parameters
        ----------

        node
            Can be a simple string or a list of strings. If the CircuitTemplate is a hierarchical circuit (composed of
            circuits itself), different list entries should refer to the different hierarchy levels. Alternatively,
            separation via slashes can be used if a string is provided.

        Returns
        -------
        NodeTemplate

        """

        if type(node) is str:
            node = node.split('/')
        net = self.circuits if self.circuits else self.nodes
        net_node = net[node[0]]
        if isinstance(net_node, CircuitTemplate):
            return net_node.get_node_template(node[1:])
        return net_node

    def _load_edge_templates(self, edges: List[Union[tuple, dict]]):
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
                try:
                    source = edge["source"]
                    target = edge["target"]
                    template = edge["template"]
                    variables = edge["variables"]
                except KeyError as e:
                    raise TypeError(f"Wrong edge configuration. Unable to find key {e.args[0]}")

            else:
                source, target, template, variables = edge

            # "template" is EdgeTemplate, just use it
            # also just leave it as None, if so it be.
            if isinstance(template, EdgeTemplate) or template is None:
                pass

            # if not, try to load template path from yaml
            else:
                path = _complete_template_path(template, self.path)
                template = EdgeTemplate.from_yaml(path)

            edges_with_templates.append((source, target, template, variables))
        return edges_with_templates

    def collect_edges(self):
        """
        Collect all edges that exist in circuit.

        Returns
        -------
        List of edge tuples with entries 1 - source variable, 2 - target variable, 3 - edge template,
        4 - edge dictionary. Source and target variables are strings that use slash notations to resolve node
        hierarchies.

        """
        edges = self.edges
        for c_scope, c in self.circuits.items():
            edges_tmp = c.collect_edges()
            for svar, tvar, template, edge_dict in edges_tmp:
                edges.append((f"{c_scope}/{svar}", f"{c_scope}/{tvar}", template, edge_dict))
        return edges


def vectorize_circuit(circuit: CircuitTemplate, vectorize: bool) -> tuple:

    # TODO: treat updates of Operatortemplates/NodeTemplates/EdgeTemplates
    node_map = dict()

    if not vectorize:
        return circuit, node_map

    # vectorize nodes
    #################

    node_keys = circuit.get_nodes(['all'])

    # group node templates that should be vectorized
    node_col = dict()
    for node in node_keys:

        template = circuit.get_node_template(node)

        if template in node_col:

            # extend existing template information
            template_dict = node_col[template]
            template_dict['old_nodes'].append(node)
            template_dict['indices'].append(template_dict['indices'][-1]+1)
            for op in template.operators:
                for key, var in op.variables.items():
                    base_dict = template_dict[op.name][key]  # type: dict
                    template_dict[op.name][key] = extend_var_dict(base_dict, var)

        else:

            # create new template entry
            template_dict = dict()
            template_dict['old_nodes'] = [node]
            template_dict['indices'] = [0]
            for op in template.operators:
                template_dict[op.name] = op.variables.copy()
            node_col[template] = template_dict

    # create new, vectorized node templates
    nodes = {}
    for old_template, new_node in node_col.items():

        # extract information about original circuit
        old_nodes = new_node.pop('old_nodes')
        indices = new_node.pop('indices')

        # create vectorized operators
        operators = []
        for op in old_template.operators:
            operators.append(OperatorTemplate(name=op.name, path=op.path, equations=op.equations,
                                              variables=new_node[op.name], description=op.__doc__))

        # create vectorized node
        name = old_template.name
        new_node = NodeTemplate(name=name, path=old_template.path, operators=operators,
                                label=old_template.label, description=old_template.__doc__)

        # store new node information
        nodes[name] = new_node
        for n, idx in zip(old_nodes, indices):
            node_map[n] = (new_node, idx)

    # vectorize edges
    #################

    # group edges that should be vectorized
    old_edges = circuit.collect_edges()
    edge_col = {}
    for source, target, template, edge_dict in old_edges:

        # extract edge information for vectorized network
        *s_node, s_op, s_var = source.split('/')
        *t_node, t_op, t_var = target.split('/')
        s_node_new, s_idx = node_map['/'.join(s_node)]
        t_node_new, t_idx = node_map['/'.join(t_node)]
        source_new = '/'.join((s_node_new.name, s_op, s_var))
        target_new = '/'.join((t_node_new.name, t_op, t_var))

        # group edges that connect the same vectorized node variables via the same edge templates
        # TODO: vectorize edge templates as well
        if (source_new, target_new, template) in edge_col:

            # extend edge dict by edge variables
            base_dict = edge_col[(source_new, target_new, template)]
            for key, val in edge_dict.items():
                if type(val) is float or type(val) is int:
                    base_dict[key].append(val)
            base_dict['source_idx'].append(s_idx)
            base_dict['target_idx'].append(t_idx)

        else:

            # prepare edge dict for vectorization
            for key, val in edge_dict.items():
                if type(val) is float or type(val) is int:
                    edge_dict[key] = [val]
            edge_dict['source_idx'] = [s_idx]
            edge_dict['target_idx'] = [t_idx]

            # add edge dict to edge collection
            edge_col[(source_new, target_new, template)] = edge_dict

    # create final set of vectorized edges
    edges = []
    for (source, target, template), edge_dict in edge_col.items():
        edges.append((source, target, template, edge_dict))

    # finalize new, vectorized circuit
    ##################################

    #  TODO: decide whether to introduce the node hierarchy at this point or not (does that even make sense in a
    #   vectorized circuit?)

    c_new = CircuitTemplate(name=circuit.name, path=circuit.path, description=circuit.__doc__, label=circuit.label,
                            nodes=nodes, edges=edges)

    return c_new, node_map


def extend_var_dict(origin: dict, extension: dict):
    value1, vtype = extract_var_value(origin)
    value2, _ = extract_var_value(extension)
    origin['default'] = vtype
    origin['value'] = np.asarray([value1, value2]).squeeze()
    origin['shape'] = origin['value'].shape
    return origin


def extract_var_value(var: dict):
    if 'value' in var:
        return var['value'], var['default']
    if type(var['default']) is str:
        if '(' in var['default']:
            idx_start = var['default'].find('(')
            idx_end = var['default'].find(')')
            return float(var['default'][idx_start:idx_end]), var['default'][:idx_start]
        return 0.0, var['default']
    return var['default'], 'constant'


def update_edges(base_edges: List[tuple], updates: List[Union[tuple, dict]]):
    """Add edges to list of edges. Removing or altering is currently not supported."""

    updated = deepcopy(base_edges)
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


def update_dict(base_dict: dict, updates: dict):
    updated = deepcopy(base_dict)

    updated.update(updates)

    return updated


def is_integration_adaptive(solver: str, **solver_kwargs):

    return solver == 'scipy'


def get_input_node(var: str, inp: np.ndarray, continuous: bool, T: float) -> tuple:

    # create input equationd and variables
    ######################################

    if continuous:

        # interpolate input variable if time steps can be variable
        from scipy.interpolate import interp1d
        time = np.linspace(0, T, inp.shape[0])
        f = interp1d(time, inp, axis=0, copy=False, kind='linear')
        f.shape = inp.shape[1:]
        eqs = [f"{var} = {var}_input(t)"]
        vars = {
            var: {'vtype': 'output', 'value': f(0.0)},
            f"{var}_input": {'vtype': 'raw', 'value': f},
            't': {'vtype': 'input', 'value': 0.0}
        }

    else:

        raise ValueError

    # create input node
    ###################

    op_key = 'in_op'
    node_key = f'{var}_input_generator'
    in_op = OperatorTemplate(name=op_key, path='none', equations=eqs, variables=vars)
    in_node = NodeTemplate(name=node_key, path='none', operators=[in_op])
    out_var = [node_key, op_key, var]

    return out_var, in_node


def collect_nodes(key: str, val: Union[CircuitTemplate, NodeTemplate]):
    if isinstance(val, CircuitTemplate):
        return val.get_nodes(['all'])
    return [key]
