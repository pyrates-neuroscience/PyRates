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
import gc
from typing import List, Union, Dict, Optional
from copy import deepcopy

import pandas as pd
from pandas import DataFrame, MultiIndex
import numpy as np

# pyrates internal _imports
from pyrates.frontend.template._io import _complete_template_path
from pyrates.frontend.template.abc import AbstractBaseTemplate
from pyrates.frontend.template.edge import EdgeTemplate
from pyrates.frontend.template.node import NodeTemplate
from pyrates.frontend.template.operator import OperatorTemplate
from pyrates.ir.circuit import get_unique_label, CircuitIR, PyRatesException
from pyrates.ir.edge import EdgeIR
from pyrates.ir.node import clear_ir_caches

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


class CircuitTemplate(AbstractBaseTemplate):

    target_ir = CircuitIR

    def __init__(self, name: str, path: str, description: str = "A circuit template.", circuits: dict = None,
                 nodes: dict = None, edges: List[tuple] = None):

        # initialize base template clase
        super().__init__(name, path, description)

        # ensure that either only nodes or only circuits exist on template for consistency of the node hierarchy
        if nodes and circuits:
            raise ValueError('CircuitTemplate has been initialized with both sub-circuits and nodes. However, all '
                             'nodes in a circuit must have the same hierarchical depth. Please provide only a node or '
                             'only a circuit dictionary. Note that you can add redundant hierarchy levels to each '
                             'sub-circuit or node by constructing a CircuitTemplate that takes only a single other '
                             'CircuitTemplate as input.')

        # add nodes to the template
        self.nodes = {}  # type: Dict[str, NodeTemplate]
        if nodes:
            for key, node in nodes.items():
                if isinstance(node, str):
                    path = _complete_template_path(node, self.path)
                    self.nodes[key] = NodeTemplate.from_yaml(path)
                else:
                    self.nodes[key] = node

        # add circuits to the template
        self.circuits = {}
        if circuits:
            for key, circuit in circuits.items():
                if isinstance(circuit, str):
                    path = _complete_template_path(circuit, self.path)
                    self.circuits[key] = CircuitTemplate.from_yaml(path)
                else:
                    self.circuits[key] = circuit

        # add edges to the template
        self._edge_map = {}
        if edges:
            self.edges = self._load_edge_templates(edges)
        else:
            self.edges = []

        # private attributes
        self._ir = None
        self._depth = self._get_hierarchy_depth()
        self._vectorization_labels = {}
        self._vectorization_indices = {}

    def update_template(self, name: str = None, path: str = None, description: str = None, circuits: dict = None,
                        nodes: dict = None, edges: List[tuple] = None):
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

        return self.__class__(name=name, path=path, description=description, circuits=circuits, nodes=nodes, edges=edges
                              )

    def update_var(self, node_vars: dict = None, edge_vars: list = None):
        """

        Parameters
        ----------
        node_vars
        edge_vars

        Returns
        -------

        """

        if node_vars is None:
            node_vars = {}
        if edge_vars is None:
            edge_vars = []

        # updates to node variable values
        for key, val in node_vars.items():
            *node, op, var = key.split('/')
            target_nodes = self.get_nodes(node_identifier=node, var_identifier=(op, var))
            if not target_nodes:
                print(f'WARNING: Variable {var} has not been found on operator {op} of node {node[0]}.')
            for n in target_nodes:
                node_temp = self.get_node_template(n)
                node_temp.update_var(op=op, var=var, val=val)

        # updates to edge variable values
        for source, target, edge_dict in edge_vars:
            _, _, _, base_dict = self.get_edge(source, target)
            base_dict.update(edge_dict)

        return self

    def run(self, simulation_time: float, step_size: float, inputs: Optional[dict] = None,
            outputs: Optional[Union[dict, list]] = None, sampling_step_size: Optional[float] = None,
            solver: str = 'euler', backend: str = None,  vectorize: bool = True, verbose: bool = True,
            clear: bool = True, **kwargs) -> pd.DataFrame:

        # translate circuit template into a graph representation
        ########################################################

        # add extrinsic inputs to network
        adaptive_steps = is_integration_adaptive(solver, **kwargs)
        net = self
        if inputs:
            for target, in_array in inputs.items():
                net = net._add_input(target, in_array, adaptive_steps, simulation_time, vectorize)

        # validate backend settings
        self._validate_backend_args(backend, vectorize, adaptive_steps, **kwargs)

        # apply template (translate into compute graph, optional vectorization process)
        net.apply(adaptive_steps=adaptive_steps, vectorize=vectorize, verbose=verbose, backend=backend,
                  step_size=step_size, **kwargs)

        # perform simulation via the graph representation
        #################################################

        # create mapping between requested output variables and the current network variables
        if type(outputs) is dict:
            output_map, outputs_ir = net._map_output_variables(outputs)
        else:
            output_map = {}
            outputs_ir = {}
            for output in outputs:
                out_map_tmp, out_vars_tmp = net._map_output_variables(output)
                output_map.update(out_map_tmp)
                outputs_ir.update(out_vars_tmp)

        # perform simulation
        outputs = net._ir.run(simulation_time=simulation_time, step_size=step_size, solver=solver,
                              sampling_step_size=sampling_step_size, outputs=outputs_ir, **kwargs)

        # apply indices to output variables
        outputs_final = {}
        for key, out_info in output_map.items():
            if type(out_info) is dict:
                outputs_final[key] = {key2: np.squeeze(outputs.pop(key2)[:, idx]) for key2, idx in out_info.items()}
            else:
                outputs_final[key] = np.squeeze(outputs.pop(key)[:, out_info])
        time_vec = outputs.pop('time')

        # interpolate data if necessary
        if sampling_step_size and not all(np.diff(time_vec, 1) - sampling_step_size < step_size * 0.01):
            n = int(np.round(simulation_time / sampling_step_size, decimals=0))
            new_times = np.linspace(step_size, simulation_time, n + 1)
            for key, val in outputs_final.items():
                if type(val) is dict:
                    for key2, v in val.items():
                        outputs_final[key][key2] = np.interp(new_times, time_vec, v)
                else:
                    outputs_final[key] = np.interp(new_times, time_vec, val)
            time_vec = new_times

        # create multi-index dataframe
        data = []
        columns = []
        multi_index = False
        for key, out in outputs_final.items():
            if type(out) is dict:
                multi_index = True
                for key2, v in out.items():
                    columns.append((key, key2))
                    data.append(v)
            else:
                columns.append(key)
                data.append(out)
        if multi_index:
            columns = MultiIndex.from_tuples(columns)
        results = DataFrame(data=np.asarray(data).T, columns=columns, index=time_vec)

        if clear:
            net.clear()

        return results

    def get_run_func(self, func_name: str, step_size: float, inputs: Optional[dict] = None, backend: str = None,
                     vectorize: bool = True, verbose: bool = True, clear: bool = False, **kwargs) -> tuple:

        # add extrinsic inputs to network
        adaptive_steps = is_integration_adaptive(kwargs.pop('solver', 'euler'), **kwargs)
        net = self
        if inputs:
            for target, in_array in inputs.items():
                net = net._add_input(target, in_array, adaptive_steps, in_array.shape[0] * step_size, vectorize)

        # validate backend settings
        net._validate_backend_args(backend, vectorize, adaptive_steps)

        # translate circuit template into a graph representation
        net.apply(adaptive_steps=adaptive_steps, verbose=verbose, backend=backend, step_size=step_size,
                  vectorize=vectorize, **kwargs)

        # generate the run function
        func, args = net._ir.get_run_func(func_name=func_name, step_size=step_size, **kwargs)

        # clear the network temporary files
        if clear:
            net.clear()

        return func, args

    def apply(self, adaptive_steps: bool = None, label: str = None, node_values: dict = None, edge_values: dict = None,
              vectorize: bool = True, verbose: bool = True, **kwargs) -> None:
        """Create a `CircuitIR` instance based on the template.

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
            If true, nodes that are governed by the same underlying equations will be vectorized, i.e. grouped together.
        verbose
            If true, updates about the backend translation process will be given.

        Returns
        -------
        None
        """

        if not label:
            label = self.name
        if not edge_values:
            edge_values = {}
        scalar_shape = (1,) if vectorize else ()

        # turn nodes from templates into IRs
        ####################################

        # prepare node parameter updates for IR transformation
        values = dict()
        if node_values:
            for key, value in node_values.items():
                *node_id, op, var = key.split("/")
                target_nodes = self.get_nodes(node_id)
                for n in target_nodes:
                    if n not in values:
                        values[n] = dict()
                    values[n]["/".join((op, var))] = value

        # go through node templates and transform them into intermediate representations
        nodes = self._apply_nodes(node_keys=self.get_nodes(['all']), values=values, vectorize=vectorize)

        # reformat edge templates to EdgeIR instances
        #############################################

        # group edges that should be vectorized
        old_edges = self.collect_edges(delay_info=True)
        edge_col = self._group_edges(edges=old_edges)

        # create final set of vectorized edges
        edges = []
        for (source, target, template, _), values in edge_col.items():

            # get edge template and instantiate it
            values = deepcopy(values)

            # update edge template default values with passed edge values,
            if (source, target) in edge_values:
                values.update(edge_values[(source, target)])
            weight = values.pop("weight", 1.)

            # get delay
            delay = values.pop("delay", None)
            spread = values.pop("spread", None)

            # get source/target indices
            source_idx = values.pop("source_idx", None)
            target_idx = values.pop("target_idx", None)

            # treat empty dummy edge templates as not existent templates
            if template and len(template.operators) == 0:
                template = None

            # add standard edge
            if template is None:

                # should not happen. Putting this just in case.
                if values:
                    raise PyRatesException("An empty edge IR was provided with additional values. "
                                           "No way to figure out where to apply those values.")

                # add simple linear edge to edge collection
                edge_dict = self._prepare_edge_for_circuit(weight=weight, delay=delay, spread=spread,
                                                           source_idx=source_idx, target_idx=target_idx)
                edges.append((source, target, edge_dict))

            # add edge from EdgeIR
            else:

                sources = {}
                for key, v in values.copy().items():
                    if type(v) is str:
                        sources[key] = values.pop(key)

                # create edge ir
                n = len(source_idx)
                if n == 1:
                    edge_ir = self._apply_edge(template, values=values, nodes=nodes, vectorize=vectorize,
                                               label_map=self._vectorization_labels)
                else:
                    edge_ir = self._apply_edge(template, values={key: v[0] for key, v in values.items()},
                                               nodes=nodes, label_map=self._vectorization_labels, vectorize=vectorize)
                    for i in range(1, n):
                        edge_ir = self._apply_edge(template, values={key: v[i] for key, v in values.items()},
                                                   nodes=nodes, label_map=self._vectorization_labels,
                                                   vectorize=vectorize)

                # project inputs to edge ir node
                if not sources:
                    i_, in_var = edge_ir.inputs.copy().popitem()
                    sources[f"{edge_ir.label}/{in_var[0]}"] = 'source'

                # collect new edge attributes
                edge_idx = list(np.arange(edge_ir.length - n, edge_ir.length))
                edge_ir_sources = self._extract_sources_from_edge_dict(sources, orig_source=source, vectorize=vectorize)
                new_edges = {}
                for edge_source, edge_target in edge_ir_sources.items():

                    if vectorize:

                        # case I: vectorized edges
                        edge_source, idx = edge_source

                        # add edge attributes to dict
                        if (edge_source, edge_target) in new_edges:
                            new_edges[(edge_source, edge_target)]['source_idx'].append(source_idx[idx])
                            new_edges[(edge_source, edge_target)]['weight'].append(1.0)
                        else:
                            new_edges[(edge_source, edge_target)] = {'source_idx': [source_idx[idx]], 'weight': [1.0]}

                    else:

                        # case II: non-vectorized edges
                        new_edges[(edge_source, edge_target)] = {'source_idx': source_idx, 'weight': 1.0}

                # add an edge for each source node that projects to new edge node
                for (s, t), data in new_edges.items():

                    # add edge from source node to the new edge node
                    edge_dict = self._prepare_edge_for_circuit(weight=data['weight'], source_idx=data['source_idx'],
                                                               target_idx=edge_idx)
                    edges.append((s, t, edge_dict))

                # add edge from new edge node to target node
                edge_dict = self._prepare_edge_for_circuit(weight=weight, delay=delay, spread=spread,
                                                           source_idx=edge_idx, target_idx=target_idx)
                edges.append((edge_ir.output, target, edge_dict))

        # instantiate an intermediate representation of the circuit template
        self._ir = CircuitIR(label, nodes=nodes, edges=edges, verbose=verbose, step_size_adaptation=adaptive_steps,
                             scalar_shape=scalar_shape, **kwargs)

    def get_nodes(self, node_identifier: Union[str, list, tuple], var_identifier: Optional[tuple] = None) -> List[str]:
        """Extracts nodes from the CircuitTemplate that match the provided identifier.

        Parameters
        ----------
        node_identifier
            Can be a simple string or a list of strings. If the CircuitTemplate is a hierarchical circuit (composed of
            circuits itself), different list entries should refer to the different hierarchy levels. Alternatively,
            separation via slashes can be used if a string is provided.
        var_identifier
            If provided, only nodes will be returned for which this variable is defined.

        Returns
        -------
        List
            List of node keys that match the provided identifier. Each entry is a string that refers to a node of the
            circuit with circuit hierarchy levels separated via slashes.

        """

        if type(node_identifier) is str:
            node_identifier = node_identifier.split('/')
        net = self.circuits if self.circuits else self.nodes

        if len(node_identifier) == 1:

            # return target nodes of circuit based on single identifier
            if node_identifier[0] in net:
                return self._get_nodes_with_var(var_identifier, nodes=node_identifier)
            if node_identifier[0] == 'all':
                if self.circuits:
                    nodes = []
                    for n in net:
                        nodes.extend(self.get_nodes(node_identifier=f"{n}/all", var_identifier=var_identifier))
                    return nodes
                return self._get_nodes_with_var(var_identifier, nodes=list(net.keys()))
            return list()

        else:

            # collect target nodes from circuit based on hierarchical identifier
            nodes = []
            node_lvl = node_identifier[0]

            # get network node identifiers that should be added to overall node list
            if node_lvl == 'all':
                for n in list(net.keys()):
                    net_tmp = net[n]
                    if isinstance(net_tmp, CircuitTemplate):
                        for n2 in net_tmp.get_nodes(node_identifier[1:], var_identifier):
                            node_key = "/".join((n, n2))
                            if node_key not in nodes:
                                nodes.append(node_key)
                    else:
                        nodes.append(n)
            else:
                net_tmp = net[node_lvl]
                if isinstance(net_tmp, CircuitTemplate):
                    for n in net_tmp.get_nodes(node_identifier[1:], var_identifier):
                        node_key = "/".join((node_lvl, n))
                        if node_key not in nodes:
                            nodes.append(node_key)
                else:
                    nodes.append(node_lvl)

            return self._get_nodes_with_var(var_identifier, nodes=nodes)

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
        *t_node, t_op, t_var = target.split('/')

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

    def collect_edges(self, delay_info: bool = False):
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
        if delay_info:
            for i, (svar, tvar, template, edge) in enumerate(edges):
                delayed = True if 'delay' in edge and edge['delay'] else False
                edges[i] = (svar, tvar, template, edge, delayed)
        return edges

    def get_edge(self, source: str, target: str, idx: int = None):

        if idx is None:
            idx = 0
        return self._edge_map[(source, target, idx)]

    def clear(self):
        """Removes all temporary files and directories that may have been created during simulations of that circuit.
        Also deletes operator template caches, _imports and path variables from working memory."""
        self._ir.clear()
        self._ir = None
        clear_ir_caches()
        input_labels.clear()
        gc.collect()

    @property
    def intermediate_representation(self):
        return self._ir

    @property
    def compute_graph(self):
        return self._ir.graph

    def _apply_nodes(self, node_keys: list, values: dict, vectorize: bool = True) -> dict:
        nodes = {}
        indices = {}
        label_map = {}
        for node in node_keys:
            updates = values[node] if node in values else {}
            node_template = self.get_node_template(node)
            node_ir, label_map_tmp = node_template.apply(values=updates, label=node, vectorize=vectorize)
            nodes[node_ir.label] = node_ir
            indices[node] = node_ir.length - 1
            for key, val in label_map_tmp.items():
                label_map[f"{node}/{key}"] = f"{node_ir.label}/{val}"
            else:
                if node != node_ir.label:
                    label_map[node] = node_ir.label

        # save label map and indices on instance
        self._vectorization_labels = label_map
        self._vectorization_indices = indices

        return nodes

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

            idx = 0
            while (source, target, idx) in self._edge_map:
                idx += 1
            self._edge_map[(source, target, idx)] = edges_with_templates[-1]

        return edges_with_templates

    def _add_input(self, target: str, inp: np.ndarray, adaptive: bool, sim_time: float, vectorized_net: bool):

        # extract target nodes from network
        *node_id, op, var = target.split('/')
        target_nodes = self.get_nodes(node_id, var_identifier=(op, var))

        # create input node
        node_key, op_key, var_key, in_node = create_input_node(var, inp, adaptive, sim_time, vectorized_net)

        # ensure that inputs match the CircuitTemplate hierarchy
        node_key, net = self._add_input_node(node_key, in_node, self._depth)

        # connect input node to target nodes
        edges = [(f"{node_key}/{op_key}/{var_key}", f"{t}/{op}/{var}", None, {'weight': 1.0})
                 for t in target_nodes]

        return net.update_template(edges=edges)

    def _add_input_node(self, node_key: str, node: NodeTemplate, depth: int) -> tuple:

        if depth > self._depth:
            raise ValueError('Input depth does not match the hierarchical depth of the circuit.')

        path = []
        input_circuits = {}
        inp_circuit = input_circuits
        net = self
        for i in range(depth):
            circuit_key = f"input_lvl_{i}"
            if circuit_key not in net.circuits:
                c = CircuitTemplate(name=circuit_key, path='none')
                net = net.update_template(circuits={circuit_key: c})
                inp_circuit[circuit_key] = {}
            else:
                inp_circuit[circuit_key] = net.circuits[circuit_key]
            net = net.circuits[circuit_key]
            if i < depth - 1:
                inp_circuit = inp_circuit[circuit_key]
            else:
                net = net.update_template(nodes={node_key: node})
                inp_circuit[circuit_key] = net
            path.append(circuit_key)
        else:
            net = net.update_template(nodes={node_key: node})
        if depth > 0:
            net = self.update_template(circuits=input_circuits)
        return "/".join(path + [node_key]), net

    def _get_nodes_with_var(self, var: tuple, nodes: list) -> list:
        if not var:
            return nodes
        final_nodes = []
        op_key, var_key = var
        for n in nodes:
            try:
                node = self.get_node_template(n)
                op_keys = [op.name for op in node.operators]
                if op_key in op_keys:
                    op = list(node.operators)[op_keys.index(op_key)]
                    op_vars = list(op.variables)
                    if var_key in op_vars:
                        final_nodes.append(n)
            except IndexError as e:
                raise e
        return final_nodes

    def _map_output_variables(self, outputs: Union[dict, str]) -> tuple:

        out_map = {}
        out_vars = {}
        indices = self._vectorization_indices

        if type(outputs) is dict:
            for key, out in outputs.items():

                *out_nodes, out_op, out_var = out.split('/')

                # get all requested node variables
                target_nodes = self.get_nodes(out_nodes, var_identifier=(out_op, out_var))

                if len(target_nodes) == 1:

                    # extract index for single output node
                    backend_key = self._relabel_var(f"{target_nodes[0]}/{out_op}/{out_var}", self._vectorization_labels)
                    idx = indices[target_nodes[0]]
                    out_map[key] = [idx]
                    out_vars[key] = backend_key

                elif target_nodes:

                    # extract index for multiple output nodes
                    out_map[key] = {}
                    for t in target_nodes:
                        key2 = f"{t}/{out_op}/{out_var}"
                        backend_key = self._relabel_var(key2, self._vectorization_labels)
                        idx = indices[t]
                        out_map[key][key2] = [idx]
                        out_vars[key2] = backend_key

        else:

            outputs = self._relabel_var(outputs, self._vectorization_labels)
            *out_nodes, out_op, out_var = outputs.split('/')
            target_nodes = self.get_nodes(out_nodes, var_identifier=(out_op, out_var))

            # extract index for single output node
            for t in target_nodes:
                key = f"{t}/{out_op}/{out_var}"
                backend_key = self._relabel_var(key, self._vectorization_labels)
                idx = indices[t]
                out_map[key] = [idx]
                out_vars[key] = backend_key

        return out_map, out_vars

    def _get_hierarchy_depth(self):
        circuit_lvls = 0
        net = self
        while net.circuits:
            circuit_lvls += 1
            net = net.circuits[list(net.circuits)[0]]
        return circuit_lvls

    def _group_edges(self, edges: list) -> dict:

        # get label map and indices from self
        label_map = self._vectorization_labels
        indices = self._vectorization_indices

        edge_col = {}
        for source, target, template, edge_dict, delayed in edges:

            edge_dict = deepcopy(edge_dict)

            # extract variable info
            *s_node, s_op, s_var = source.split('/')
            *t_node, t_op, t_var = target.split('/')
            s_node = '/'.join(s_node)
            t_node = '/'.join(t_node)

            # relabel variables according to variable map (accounting for vectorization)
            source_new = self._relabel_var(source, label_map)
            target_new = self._relabel_var(target, label_map)

            s_idx = indices[s_node]
            t_idx = indices[t_node]

            # group edges that connect the same vectorized node variables via the same edge templates
            if (source_new, target_new, template, delayed) in edge_col:

                # extend edge dict by edge variables
                base_dict = edge_col[(source_new, target_new, template, delayed)]
                for key, val in edge_dict.items():
                    if type(val) is not str:
                        base_dict[key].append(val)
                base_dict['source_idx'].append(s_idx)
                base_dict['target_idx'].append(t_idx)

            else:

                # prepare edge dict for vectorization
                for key, val in edge_dict.items():
                    if type(val) is str:
                        edge_dict[key] = val
                    else:
                        edge_dict[key] = [val]
                edge_dict['source_idx'] = [s_idx]
                edge_dict['target_idx'] = [t_idx]

                # add edge dict to edge collection
                edge_col[(source_new, target_new, template, delayed)] = edge_dict

        return edge_col

    def _extract_sources_from_edge_dict(self, edge_dict: dict, orig_source: str, vectorize: bool = False) -> dict:

        label_map = self._vectorization_labels
        indices = self._vectorization_indices

        edge_sources = {}
        for key, value in edge_dict.copy().items():

            # handle edge input source
            try:
                _, _, _ = value.split("/")
            except AttributeError:
                continue
            except ValueError:
                value = orig_source

            # handle edge input variable
            try:
                _, _, _ = key.split("/")
            except ValueError as e:
                raise e
            else:
                edge_dict.pop(key)

            # add new mapping from edge source to edge input variable
            source_key = self._relabel_var(value, label_map)

            if vectorize:

                # case I: vectorized edges
                *snode, _, _ = value.split('/')
                idx = indices['/'.join(snode)]
                edge_sources[(source_key, idx)] = self._relabel_var(key, label_map)

            else:

                # case II: non-vectorized edges
                if source_key in edge_sources:
                    raise ValueError(f'The same source variable ({value})is used for two distinct edge variables of the '
                                     f'same edge. Either provide different sources for the two edge variables, collapse '
                                     f'them into a single edge variable, or set `vectorize` to `True`.')
                edge_sources[source_key] = self._relabel_var(key, label_map)

        return edge_sources

    @staticmethod
    def _apply_edge(edge: EdgeTemplate, values: dict, nodes: dict, label_map: dict, vectorize: bool = True):

        # apply edge template
        edge_ir, label_map_tmp = edge.apply(values=values, vectorize=vectorize)  # type: EdgeIR, dict

        # save edge ir references to dictionaries (treat it as a node)
        nodes[edge_ir.label] = edge_ir
        for key, val in label_map_tmp.items():
            label_map[f"{edge.name}/{key}"] = f"{edge_ir.label}/{val}"
        else:
            if edge.name != edge_ir.label:
                label_map[edge.name] = edge_ir.label

        return edge_ir

    @staticmethod
    def _prepare_edge_for_circuit(weight: Union[float, list], delay: list = None, spread: list = None,
                                  source_idx: list = None, target_idx: list = None) -> dict:

        edge_dict = dict(weight=weight,
                         delay=delay,
                         spread=spread)

        # now add extra sources, if there are some
        if source_idx:
            edge_dict['source_idx'] = source_idx
            if type(edge_dict['weight']) is float:
                edge_dict['weight'] = [edge_dict['weight']] * len(edge_dict['source_idx'])
        if target_idx:
            edge_dict['target_idx'] = target_idx
            if type(edge_dict['weight']) is float:
                edge_dict['weight'] = [edge_dict['weight']] * len(edge_dict['target_idx'])

        return edge_dict

    @staticmethod
    def _relabel_var(var: str, var_map: dict) -> str:
        var_split = var.split('/')
        var_op = '/'.join(var_split[:-1])
        var_node = '/'.join(var_split[:-2])
        if var_op in var_map:
             var = f"{var_map[var_op]}/{var_split[-1]}"
        elif var_node in var_map:
            var = f"{var_map[var_node]}/{var_split[-2]}/{var_split[-1]}"
        return var

    @staticmethod
    def _validate_backend_args(backend: str, vectorize: bool, dt_adapt: bool, **kwargs) -> None:

        if vectorize and backend in ['fortran', 'julia']:
            raise PyRatesException(f'Vectorization is not enabled for your choice of backend: {backend}. Please either '
                                   f'choose another backend or set `vectorize` to `False`.')
        if backend == 'julia' and 'julia_path' not in kwargs:
            raise PyRatesException('You chose the Julia backend, which compiles Julia code via PyJulia. To do this,'
                                   'please provide the path to Julia executable via `julia_path`.')


def extend_var_dict(origin: dict, extension: dict):
    value1, vtype = extract_var_value(origin)
    value2, _ = extract_var_value(extension)
    if vtype != 'raw':
        origin['default'] = vtype
        origin['value'] = np.asarray([value1, value2]).squeeze()
        origin['shape'] = origin['value'].shape
    elif value1 != value2:
        raise NotImplementedError('Raw input variables with different values cannot be vectorized currently.')
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
    return solver != 'euler'


input_labels = []  # cache for input nodes


def create_input_node(var: str, inp: np.ndarray, continuous: bool, T: float, vectorized_net: bool) -> tuple:

    # create input equation and variables
    #####################################

    # create left-hand side of input assignment
    var_name = get_unique_label(f"{var}_timed_input", input_labels)
    lhs = f"index({var_name}, 0)" if vectorized_net else var_name
    lhs_shape = (1,) if vectorized_net else ()
    input_labels.append(var_name)

    if continuous:

        # case I: interpolate input variable if time steps can vary
        inp = inp.squeeze()
        inp = inp.squeeze()
        time = np.linspace(0, T, inp.shape[0])
        y_new = np.interp(0.0, time, inp)
        eqs = [f"{lhs} = interp(t, time, {var}_input)"]
        var_dict = {
            var_name: {'default': 'output', 'value': float(y_new), 'shape': lhs_shape, 'dtype': 'float'},
            f"{var}_input": {'default': 'constant', 'value': inp, 'shape': inp.shape, 'dtype': 'float'},
            't': {'default': 'variable', 'value': 0.0, 'dtype': 'float', 'shape': ()},
            'time': {'default': 'input_variable', 'value': time, 'shape': time.shape, 'dtype': 'float'}
        }

    else:

        # case II: simply index the input variable with fixed time steps
        eqs = [f"{lhs} = index({var}_input,t)"]
        var_dict = {
            var_name: {'default': 'output', 'value': 0.0, 'shape': lhs_shape, 'dtype': 'float'},
            f"{var}_input": {'default': 'constant', 'value': inp, 'shape': inp.shape, 'dtype': 'float'},
            't': {'default': 'variable', 'value': 0, 'dtype': 'int32', 'shape': ()}
        }

    # create input operator
    #######################

    op_key = get_unique_label(f'{var}_input_op', input_labels)
    in_op = OperatorTemplate(name=op_key, path='none', equations=eqs, variables=var_dict)
    node_key = get_unique_label(f'{var}_input_node', input_labels)
    in_node = NodeTemplate(name=node_key, path='none', operators=[in_op])
    input_labels.append(node_key)
    input_labels.append(op_key)

    return node_key, op_key, var_name, in_node


def collect_nodes(key: str, val: Union[CircuitTemplate, NodeTemplate]):
    if isinstance(val, CircuitTemplate):
        return val.get_nodes(['all'])
    return [key]
