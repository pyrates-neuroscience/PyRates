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

# external _imports
import time
from typing import Union, Dict, Iterator, Optional, List, Tuple
from warnings import filterwarnings
from networkx import MultiDiGraph, DiGraph
import numpy as np
from copy import deepcopy
from warnings import warn

# pyrates-internal _imports
from pyrates.backend import PyRatesException, PyRatesWarning
from pyrates.ir.node import NodeIR
from pyrates.ir.edge import EdgeIR
from pyrates.ir.abc import AbstractBaseIR
from pyrates.backend.parser import parse_equations, get_unique_label
from pyrates.backend.computegraph import ComputeGraph, ComputeVar, ComputeGraphBackProp

__author__ = "Daniel Rose, Richard Gast"
__status__ = "Development"


in_edge_indices = {}  # cache for the number of input edges per network node
in_edge_vars = {}   # cache for the input variables that enter at each target operator


#####################
# class definitions #
#####################

# networkx-based representation of all nodes and edges in circuit
class NetworkGraph(AbstractBaseIR):
    """View on the entire network as a graph. Translates edge operations and attributes into a form that allows to
    parse the network graph into a final compute graph."""

    def __init__(self, label: str = "circuit", nodes: Dict[str, NodeIR] = None, edges: list = None,
                 template: str = None, step_size: float = 1e-3, step_size_adaptation: bool = True, verbose: bool = True,
                 **kwargs):

        super().__init__(label=label, template=template)
        self._edge_idx_counter = 0
        self.step_size = step_size
        self.step_size_adaptation = step_size_adaptation
        self.graph = MultiDiGraph()

        if verbose:
            print("Compilation Progress")
            print("--------------------")
            print('\t(1) Translating the circuit template into a networkx graph representation...')

        # add nodes to graph
        if nodes:
            nodes = ((key, {"node": node}) for key, node in nodes.items())
            self.graph.add_nodes_from(nodes)

        # add edges to graph
        if edges:
            for (source, target, edge_dict) in edges:
                self.add_edge(source, target, **edge_dict)

        if verbose:
            print('\t\t...finished.')
            print("\t(2) Preprocessing edge transmission operations...")

        # translate edge operations and attributes into graph operators
        self._preprocess_edge_operations(dde_approx=kwargs.pop('dde_approx', 0),
                                         matrix_sparseness=kwargs.pop('matrix_sparseness', 0.1),
                                         vectorized=kwargs.pop('vectorized'))

        if verbose:
            print("\t\t...finished.")

    def __getitem__(self, key: str):
        """
        Custom implementation of __getitem__ that dissolves strings of form "key1/key2/key3" into
        lookups of form self[key1][key2][key3].

        Parameters
        ----------
        key

        Returns
        -------
        item
        """

        try:
            return super().__getitem__(key)
        except KeyError:
            keys = key.split('/')
            for i in range(len(keys)):
                if "/".join(keys[:i+1]) in self.nodes:
                    break
            key_iter = iter(['/'.join(keys[:i+1])] + keys[i+1:])
            key = next(key_iter)
            item = self.getitem_from_iterator(key, key_iter)
            for key in key_iter:
                item = item.getitem_from_iterator(key, key_iter)
        return item

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
        source_vars, extra_sources = self._parse_source_vars(source_node, source_var, edge_ir,
                                                             data.pop("extra_sources", None))

        # step 3: add edges
        attr_dict = dict(edge_ir=edge_ir,
                         weight=weight,
                         delay=delay,
                         spread=spread,
                         source_var=source_vars,
                         target_var=target_var,
                         extra_sources=extra_sources,
                         **data)
        self.graph.add_edge(source_node, target_node, **attr_dict)

    def getitem_from_iterator(self, key: str, key_iter: Iterator[str]):
        return self.graph.nodes[key]["node"]

    def _preprocess_edge_operations(self, dde_approx: int = 0, vectorized: bool = True, **kwargs):
        """Restructures network graph to collapse nodes and edges that share the same operator graphs. Variable values
        get an additional vector dimension. References to the respective index is saved in the internal `label_map`."""

        # go through nodes and create buffers for delayed outputs and mappings for their inputs
        #######################################################################################

        for node_name in self.nodes:

            node_outputs = self.graph.out_edges(node_name, keys=True)
            node_outputs = self._sort_edges(node_outputs, 'source_var', data_included=False)

            # loop over ouput variables of node
            for i, (out_var, edges) in enumerate(node_outputs.items()):

                # extract delay info from variable projections
                op_name, var_name = out_var.split('/')
                delays, spreads, nodes, add_delay = self._collect_delays_from_edges(edges)

                # add synaptic buffer to output variables with delay
                if add_delay:
                    if vectorized:
                        self._add_edge_buffer(node_name, op_name, var_name, edges=edges, delays=delays,
                                              nodes=nodes, spreads=spreads, dde_approx=dde_approx)
                    else:
                        # TODO: sort edges into unique delay/spread combinations and only loop over those
                        if spreads:
                            for i, (edge, delay, spread, node) in enumerate(zip(edges, delays, spreads, nodes)):
                                self._add_edge_buffer(node_name, op_name, var_name, edges=[edge], delays=[delay],
                                                      nodes=[node], spreads=[spread], dde_approx=dde_approx,
                                                      buffer_id=f"_out{i}")
                        else:
                            for i, (edge, delay, node) in enumerate(zip(edges, delays, nodes)):
                                self._add_edge_buffer(node_name, op_name, var_name, edges=[edge], delays=[delay],
                                                      nodes=[node], dde_approx=dde_approx, buffer_id=f"_out{i}")

        # go through nodes again, and collect and process all inputs to each node variable
        ##################################################################################

        for node_name in self.nodes:

            node_inputs = self.graph.in_edges(node_name, keys=True)
            node_inputs = self._sort_edges(node_inputs, 'target_var', data_included=False)

            # loop over inputs to node variable
            for i, (in_var, edges) in enumerate(node_inputs.items()):

                # extract info from projections to input variable
                op_name, var_name = in_var.split('/')
                data = self._collect_from_edges(edges,
                                                keys=['source_var', 'weight', 'source_idx', 'target_idx'])

                # create the final equations for all edges that target the input variable
                self._generate_edge_equation(tnode=node_name, top=op_name, tvar=var_name, inputs=data, **kwargs)

    def _sort_edges(self, edges: List[tuple], attr: str, data_included: bool = False) -> dict:
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
        if data_included:
            for edge in edges:
                if len(edge) == 4:
                    source, target, edge, data = edge
                else:
                    raise ValueError("Missing edge index. This error message should not occur.")
                value = self.edges[source, target, edge][attr]

                if value not in edges_new.keys():
                    edges_new[value] = [(source, target, edge, data)]
                else:
                    edges_new[value].append((source, target, edge, data))
        else:
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

    def _collect_delays_from_edges(self, edges):
        means, stds, nodes = [], [], []
        for s, t, e in edges:

            # extract delay
            d = self.edges[s, t, e]['delay']
            if type(d) is list:
                d = [1 if d_tmp is None else d_tmp for d_tmp in d]

            # extract and process delay distribution spread
            v = self.edges[s, t, e].pop('spread', [0])
            if v is None or np.sum(v) == 0:
                v = [0] * len(self.edges[s, t, e]['target_idx'])
                discretize = True
            else:
                discretize = False
                v = self._process_delays(v, discretize=discretize)

            # finalize edge delay
            if d is None or np.sum(d) == 0:
                d = [1] * len(self.edges[s, t, e]['target_idx'])
            else:
                d = self._process_delays(d, discretize=discretize)

            # extract source var index
            source = self.edges[s, t, e]['source_idx']
            if len(d) > 1 and len(source) == 1:
                source = source * len(d)

            # collect values
            means += d
            stds += v
            nodes.append(source)

        # check whether edge delays have to be implemented or can be ignored
        max_delay = np.max(means)
        add_delay = ("int" in str(type(max_delay)) and max_delay > 1) or \
                    ("float" in str(type(max_delay)) and max_delay > self.step_size)
        if sum(stds) == 0:
            stds = None

        # if delays are going to be added from the created lists, remove the delays from the edges themselves
        if add_delay:
            for s, t, e in edges:
                self.edges[s, t, e]['source_idx'] = []
                self.edges[s, t, e]['delay'] = None

        return means, stds, nodes, add_delay

    def _collect_from_edges(self, edges: list, keys: list):
        data = dict()
        for source, target, idx in edges:
            edge = self.edges[(source, target, idx)]
            if source not in data:
                data[source] = dict()
            for key in keys:
                val = edge[key]  #deepcopy(edge[key])
                try:
                    data[source][key].extend(val)
                except AttributeError:
                    field = data[source][key]
                    if type(field) is str or field is None:
                        pass
                    else:
                        data[source][key] = [field, val]
                except KeyError:
                    data[source][key] = val
        return data

    def _add_edge_buffer(self, node: str, op: str, var: str, edges: list, delays: list, nodes: list,
                         spreads: Optional[list] = None, dde_approx: int = 0, buffer_id: str = "") -> None:
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
        node_var = self[f"{node}/{op}/{var}"]
        target_shape = node_var['shape']
        node_ir = self[node]
        nodes_tmp = list()
        for n in nodes:
            nodes_tmp += n
        source_idx = np.asarray(nodes_tmp, dtype='int').flatten()

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

            # sort all edge information in ascending ODE order
            order_idx = np.argsort(orders, kind='stable')
            orders_sorted = np.asarray(orders, dtype='int')[order_idx]
            orders_tmp = np.asarray(orders, dtype='int')[order_idx]
            rates_tmp = np.asarray(rates)[order_idx]
            source_idx_tmp = source_idx[order_idx]

            buffer_eqs, var_dict, final_idx = [], {}, []
            max_order = max(orders)
            for i in range(max_order + 1):

                # check which edges require the ODE order treated in this iteration of the loop
                k = i + 1
                idx, idx_str, idx_var = self._bool_to_idx(orders_tmp >= k)
                if type(idx) is int:
                    idx = [idx]
                var_dict.update(idx_var)

                # define new equation variable/parameter names
                var_next = f"{var}_d{k}{buffer_id}"
                var_prev = f"{var}_d{i}{buffer_id}" if i > 0 else var
                rate = f"k_d{k}{buffer_id}"

                # prepare variables for the next ODE
                idx_apply = len(idx) != len(orders_tmp)
                val = rates_tmp[idx] if idx_apply else rates_tmp
                var_shape = (len(val),) if val.shape else ()
                if i == 0 and idx != [0] and (sum(target_shape) != len(idx) or any(np.diff(order_idx) != 1)):
                    var_prev_idx = f"index({var_prev}, source_idx{buffer_id})"
                    var_dict[f"source_idx{buffer_id}"] = {'vtype': 'constant',
                                                          'dtype': 'int',
                                                          'shape': (len(source_idx_tmp[idx]),),
                                                          'value': source_idx_tmp[idx]}
                elif i != 0 and idx_apply:
                    var_prev_idx = _get_indexed_var_str(var_prev, idx_str, var_length=len(rates_tmp))
                else:
                    var_prev_idx = var_prev

                # create new ODE string and corresponding variable definitions
                buffer_eqs.append(f"d/dt * {var_next} = {rate} * ({var_prev_idx} - {var_next})")
                var_dict[var_next] = {'vtype': 'state_var',
                                      'dtype': 'float',
                                      'shape': var_shape,
                                      'value': 0.}
                var_dict[rate] = {'vtype': 'constant',
                                  'dtype': 'float',
                                  'value': val}

                # store indices that are required to fill the edge buffer variable
                if idx_apply:

                    # right-hand side index
                    if len(orders_tmp) < 2:
                        idx_rhs_str = ''
                    elif i == 0:
                        idx_rhs = np.asarray(source_idx_tmp)[orders_tmp == i]
                        n_idx = len(idx_rhs)
                        if n_idx > 1:
                            idx_rhs_str = f"source_idx2{buffer_id}"
                            var_dict[f"source_idx2{buffer_id}"] = {'vtype': 'constant',
                                                                   'dtype': 'int',
                                                                   'shape': (n_idx,),
                                                                   'value': idx_rhs}
                        else:
                            idx_rhs_str = f"{idx_rhs[0]}"
                    else:
                        _, idx_rhs_str, _ = self._bool_to_idx(orders_tmp == i)

                    # left-hand side index
                    if len(delays) > 1:
                        _, idx_lhs_str, _ = self._bool_to_idx(orders_sorted == i)
                    else:
                        idx_lhs_str = ''
                    final_idx.append((i, idx_lhs_str, idx_rhs_str))

                # reduce lists of orders and rates by the ones that are fully implemented by the current ODE set
                if idx_apply:
                    orders_tmp = orders_tmp[idx]
                    rates_tmp = rates_tmp[idx]
                if not orders_tmp.shape:
                    orders_tmp = np.asarray([orders_tmp], dtype='int')
                    rates_tmp = np.asarray([rates_tmp])

            # remove unnecessary ODEs
            for _ in range(len(buffer_eqs) - final_idx[-1][0]):
                i = len(buffer_eqs)
                var_dict.pop(f"{var}_d{i}{buffer_id}")
                var_dict.pop(f"k_d{i}{buffer_id}")
                buffer_eqs.pop(-1)

            # create edge buffer variable
            buffer_length = len(delays)
            for i, idx_l, idx_r in final_idx:
                lhs = _get_indexed_var_str(f"{var}_buffered{buffer_id}", idx_l, var_length=buffer_length)
                rhs = _get_indexed_var_str(f"{var}_d{i}{buffer_id}" if i != 0 else var, idx_r, var_length=buffer_length)
                buffer_eqs.append(f"{lhs} = {rhs}")
            var_dict[f"{var}_buffered{buffer_id}"] = {'vtype': 'variable',
                                                      'dtype': 'float',
                                                      'shape': (buffer_length,),
                                                      'value': 0.0}

            # re-order buffered variable if necessary
            if any(np.diff(order_idx) != 1):
                buffer_eqs.append(f"{var}_buffered{buffer_id} = index({var}_buffered{buffer_id}, "
                                  f"{var}_buffered_idx{buffer_id})")
                var_dict[f"{var}_buffered_idx{buffer_id}"] = {'vtype': 'constant',
                                                              'dtype': 'int',
                                                              'shape': (len(order_idx),),
                                                              'value': np.argsort(order_idx, kind='stable')}

        # discretized edge buffers
        ##########################

        elif not self.step_size_adaptation:

            # create buffer variable shapes
            if len(target_shape) < 1 or (len(target_shape) == 1 and target_shape[0] == 1):
                buffer_shape = (max_delay + 1,)
            else:
                buffer_shape = (target_shape[0], max_delay + 1)

            # create buffer variable definitions
            var_dict = {f'{var}_buffer{buffer_id}': {'vtype': 'variable',
                                                     'dtype': 'float',
                                                     'shape': buffer_shape,
                                                     'value': 0.},
                        f'{var}_buffered{buffer_id}': {'vtype': 'variable',
                                                       'dtype': 'float',
                                                       'shape': (len(delays),),
                                                       'value': 0.},
                        f'{var}_delays{buffer_id}': {'vtype': 'constant',
                                                     'dtype': 'int',
                                                     'value': delays},
                        f'source_idx{buffer_id}': {'vtype': 'constant',
                                                   'dtype': 'int',
                                                   'value': source_idx}}

            # create buffer equations
            if len(target_shape) < 1 or (len(target_shape) == 1 and target_shape[0] == 1):
                buffer_eqs = [f"index_axis({var}_buffer{buffer_id}) = roll({var}_buffer{buffer_id}, 1)",
                              f"index({var}_buffer{buffer_id}, 0) = {var}",
                              f"{var}_buffered{buffer_id} = index({var}_buffer{buffer_id}, {var}_delays{buffer_id})"]
            else:
                buffer_eqs = [f"index_axis({var}_buffer{buffer_id}) = roll({var}_buffer{buffer_id}, 1, 1)",
                              f"index_axis({var}_buffer{buffer_id}, 0, 1) = {var}",
                              f"{var}_buffered{buffer_id} = index_2d({var}_buffer{buffer_id}, source_idx{buffer_id}, "
                              f"{var}_delays{buffer_id})"]

        # Turn ODE system into DDE system
        #################################

        else:

            warn(PyRatesWarning(f'PyRates detected an edge definition that implies to represent the model as a '
                                f'delayed differential equation system.\n PyRates will thus attempt to access the '
                                f'history of the source variable {var} of operator {op} on node {node}. '
                                f'Note that this requires {var} to be a state-variable, i.e. a variable defined by '
                                f'a differential equation.'))

            # create buffer variable definitions
            var_dict = {f'{var}_buffered{buffer_id}': {'vtype': 'variable',
                                                       'dtype': 'float',
                                                       'shape': (len(delays),),
                                                       'value': 0.}
                        }

            buffer_eqs = []
            for i, (d, sidx) in enumerate(zip(delays, source_idx)):
                if len(target_shape) < 1 or (len(target_shape) == 1 and target_shape[0] == 1):
                    buffer_eqs.append(f"{var}_buffered{buffer_id} = past({var}, {d})")
                else:
                    buffer_eqs.append(f"index({var}_buffered{buffer_id}, {sidx}) = index(past({var}, {d}), {sidx})")

        # add buffer equations to node operator
        op_info = node_ir[op]
        op_info['equations'] += buffer_eqs
        op_info['variables'].update(var_dict)
        op_info['output'] = f"{var}_buffered{buffer_id}"

        # update input information of node operators connected to this operator
        for succ in node_ir.op_graph.succ[op]:
            inputs = self[f"{node}/{succ}"]['inputs']
            if var not in inputs.keys():
                inputs[var] = {'sources': {op}}

        # update edge information
        idx_l = 0
        for i, edge in enumerate(edges):
            s, t, e = edge
            self.edges[s, t, e]['source_var'] = f"{op}/{var}_buffered{buffer_id}"
            if len(edges) > 1:
                idx_h = idx_l + len(nodes[i])
                self.edges[s, t, e]['source_idx'] = list(range(idx_l, idx_h))
                idx_l = idx_h

    def _generate_edge_equation(self, tnode: str, top: str, tvar: str, inputs: dict, matrix_sparseness: float = 0.1,
                                weight_minimum: float = 1e-8):

        # step 0: check properties of the target variable and its inputs
        multiple_inputs = len(inputs) > 1
        tval = self[f"{tnode}/{top}/{tvar}"]
        if tval['shape']:
            tsize = sum(tval['shape'])
        elif type(tval['value']) is list:
            tsize = len(tval['value'])
        else:
            tsize = 0

        # step 1: collect all inputs
        weights, source_indices, target_indices, sources = [], [], [], []
        for snode, sinfo in inputs.items():
            weights.append(sinfo['weight'])
            source_indices.append(sinfo['source_idx'])
            target_indices.append(sinfo['target_idx'])
            sources.append((snode,) + tuple(sinfo['source_var'].split('/')))

        # step 2: process incoming edges
        source_vars, args = {}, {}
        eqs, in_vars = [], []
        for i, (weight, sidx, tidx, (snode, sop, svar)) in \
                enumerate(zip(weights, source_indices, target_indices, sources)):

            # get source variable size
            sval = self[f"{snode}/{sop}/{svar}"]
            if sval['shape']:
                ssize = sum(sval['shape'])
            elif type(sval['value']) is list:
                ssize = len(sval['value'])
            else:
                ssize = 1

            # check whether the edge can be realized via a matrix product
            if not tidx:
                tidx = [0 for _ in range(len(sidx))]
            n, m = len(tidx), len(sidx)
            if len(np.unique(tidx)) < len(tidx):
                if not sidx:
                    sidx = [i for i in range(len(weight))]
                dot_edge = True
            elif n*m > 1 and tsize*ssize > 1:
                dot_edge = len(weight) / (n * m) > matrix_sparseness
            else:
                dot_edge = False

            # define new input variable if necessary
            if multiple_inputs:
                in_shape = (tsize,)
                t_str = f'{tvar}_in{i}'
                w_str = f'weight_in{i}'
                s_str = f'{svar}_in{i}'
                sidx_str = f'source_idx_in{i}'
                tidx_str = f'target_idx_in{i}'
                args[t_str] = {'value': np.zeros(in_shape), 'dtype': 'float', 'vtype': 'variable',
                               'shape': in_shape}
            else:
                t_str = tvar
                w_str = 'weight'
                s_str = svar
                sidx_str = 'source_idx'
                tidx_str = 'target_idx'

            # case I: realize edge projection via a matrix product
            if dot_edge:

                # create weight matrix
                sidx_unique = np.unique(sidx)
                tidx_unique = np.unique(tidx)
                weight_mat = np.zeros((len(tidx_unique), len(sidx_unique)))
                for t, s, w in zip(tidx, sidx, weight):
                    row = np.argwhere(tidx_unique == t).squeeze()
                    col = np.argwhere(sidx_unique == s).squeeze()
                    weight_mat[row, col] = w

                # define edge projection equation
                s_str_final = _get_indexed_var_str(s_str, sidx_unique, ssize, idx_str=sidx_str, arg_dict=args)
                t_str_final = _get_indexed_var_str(t_str, tidx_unique, tsize, idx_str=tidx_str, arg_dict=args)
                eq = f"{t_str_final} = matvec({w_str}, {s_str_final})"
                args[w_str] = {'vtype': 'constant', 'value': weight_mat, 'dtype': 'float', 'shape': weight_mat.shape}

            # case II: realize edge projection via source and target indexing
            else:

                # check wether weighting of source variables is required
                if all([abs(w-1) < weight_minimum for w in weight]):
                    weighting = ""
                else:
                    weighting = f" * {w_str}"
                    args[w_str] = {'vtype': 'constant', 'dtype': 'float', 'value': weight if ssize > 1 else weight[0]}

                # get final source and target strings
                s_str_final = _get_indexed_var_str(s_str, sidx, ssize, reduce=m == 1 and tsize > 1 and n == 1,
                                                   idx_str=sidx_str, arg_dict=args)
                t_str_final = _get_indexed_var_str(t_str, tidx, tsize, reduce=tsize > 1 or ssize < tsize,
                                                   idx_str=tidx_str, arg_dict=args)

                # define edge equation
                eq = f"{t_str_final} = {s_str_final}{weighting}"

            # add equation and source information
            eqs.append(eq)
            source_vars[s_str] = {'sources': [sop], 'node': snode, 'var': svar}
            in_vars.append(t_str)

        # step 3: process multiple inputs to same variable
        if multiple_inputs:

            # finalize edge equations
            eq = f"{tvar} = {'+'.join(in_vars)}"
            eqs.append(eq)

        # step 4: define target variable as operator output
        args[tvar] = tval  #deepcopy(tval)
        args[tvar]['vtype'] = 'variable'

        # step 5: add edge operator to target node
        if tnode not in in_edge_indices:
            in_edge_indices[tnode] = 0
        op_name = f'in_edge_{in_edge_indices[tnode]}'
        in_edge_indices[tnode] += 1
        tnode_ir = self[tnode]
        tnode_ir.add_op(op_name, inputs=source_vars, output=tvar, equations=eqs, variables=args)
        tnode_ir.add_op_edge(op_name, top)

        # step 6: add input information to target operator
        inputs = self[tnode][top]['inputs']
        if tvar in inputs.keys():
            inputs[tvar]['sources'].add(op_name)
        else:
            inputs[tvar] = {'sources': [op_name]}

    def _process_delays(self, d, discretize=True):
        if type(d) is list:
            d = np.asarray(d).squeeze()
            d = [self._preprocess_delay(d_tmp, discretize=discretize) for d_tmp in d] if d.shape else \
                [self._preprocess_delay(d, discretize=discretize)]
        else:
            d = [self._preprocess_delay(d, discretize=discretize)]
        return d

    def _preprocess_delay(self, delay, discretize=True):
        return int(np.round(delay / self.step_size, decimals=0)) if discretize and not self.step_size_adaptation \
            else delay

    def _bool_to_idx(self, v):
        v_idx = np.argwhere(v).squeeze()
        v_dict = {}
        if v_idx.shape and v_idx.shape[0] > 1 and all(np.diff(v_idx) == 1):
            v_idx_str = (f"{v_idx[0]}", f"{v_idx[-1] + 1}")
        elif v_idx.shape and v_idx.shape[0] > 1:
            var_name = f"delay_idx{self._edge_idx_counter}"
            v_idx_str = f"{var_name}"
            v_dict[var_name] = {'value': v_idx, 'vtype': 'constant'}
            self._edge_idx_counter += 1
        else:
            try:
                v_idx_str = f"{v_idx.max()}"
            except ValueError:
                v_idx_str = ""
        return v_idx.tolist(), v_idx_str, v_dict

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

        if extra_sources is not None:
            for edge_var, source in extra_sources.items():
                node, op, var = source.split("/")
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

        # go trough circuit hierarchy
        path = "/".join(parts)

        # check if path is valid
        if path not in self:
            raise PyRatesException(f"Could not find object with path `{path}`.")

    @staticmethod
    def _parse_edge_specifier(specifier: str, data: dict, var_string: str) -> Tuple[str, Union[str, dict]]:
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

        return node, var

    @property
    def nodes(self):
        """Shortcut to self.graph.nodes. See documentation of `networkx.MultiDiGraph.nodes`."""
        return self.graph.nodes

    @property
    def edges(self):
        """Shortcut to self.graph.edges. See documentation of `networkx.MultiDiGraph.edges`."""
        return self.graph.edges


# intermediate representation that provides interface between frontend and backend
class CircuitIR(AbstractBaseIR):
    """Custom graph data structure that represents a backend of nodes and edges with associated equations
    and variables."""

    __slots__ = ["label", "_front_to_back", "graph", "_t", "_verbose", "_dt", "_dt_adapt", "_def_shape"]

    def __init__(self, label: str = "circuit", nodes: Dict[str, NodeIR] = None, edges: list = None,
                 template: str = None, step_size_adaptation: bool = False, step_size: float = None,
                 verbose: bool = True, backend: str = None, scalar_shape: tuple = None, **kwargs):
        """
        Parameters:
        -----------
        label
            String label, could be used as fallback when subcircuiting this circuit. Currently not used, though.
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

        # filter displayed warnings
        filterwarnings("ignore", category=FutureWarning)

        # set main attributes
        super().__init__(label=label, template=template)
        self._verbose = verbose
        self._front_to_back = dict()
        self._dt = step_size
        self._dt_adapt = step_size_adaptation
        self._def_shape = (1,) if scalar_shape is None else scalar_shape

        # translate the network into a networkx graph
        net = NetworkGraph(nodes=nodes, edges=edges, label=label, step_size=step_size,
                           step_size_adaptation=step_size_adaptation, verbose=verbose, **kwargs)

        # parse network equations into a compute graph
        if verbose:
            print("\t(3) Parsing the model equations into a compute graph...")

        self.graph = self.network_to_computegraph(graph=net, backend=backend, **kwargs)

        if verbose:
            print("\t\t...finished.")
            print("\tModel compilation was finished.")

    def get_var(self, var: str, get_key: bool = False) -> Union[str, ComputeVar]:
        """Extracts variable from the backend (i.e. the `ComputeGraph` instance).

        Parameters
        ----------
        var
            Name of the variable.
        get_key
            If true, the backend variable name will be returned

        Returns
        -------
        Union[str, ComputeVar]
            Either the backend variable or its name.
        """
        try:
            v = self[var]
        except KeyError:
            v = self._front_to_back[var]
        return v.name if get_key else v

    def get_frontend_varname(self, var: str) -> str:
        """Returns the original frontend variable name given the backend variable name `var`.

        Parameters
        ----------

        var
            Name of the backend variable.

        Returns
        -------
        str
            Name of the frontend variable
        """
        v = self.get_var(var)
        front_vars = list(self._front_to_back.keys())
        back_vars = list(self._front_to_back.values())
        idx = back_vars.index(v)
        return front_vars[idx]

    def run(self,
            simulation_time: float,
            outputs: Optional[dict] = None,
            sampling_step_size: Optional[float] = None,
            solver: str = 'euler',
            **kwargs
            ) -> dict:
        """Simulate the backend behavior over time via a tensorflow session.

        Parameters
        ----------
        simulation_time
            Simulation time in seconds.
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
        kwargs
            Keyword arguments that are passed on to the chosen solver.

        Returns
        -------
        dict
            Output variables in a dictionary.
        """

        filterwarnings("ignore", category=FutureWarning)

        # collect backend variables and functions
        #########################################

        if self._verbose:
            print("Simulation Progress")
            print("-------------------")
            print("\t (1) Generating the network run function...")

        # generate run function
        func_name = kwargs.pop('func_name', 'vector_field')
        func, func_args, _, _ = self.get_run_func(func_name, **kwargs)

        # extract backend variables that correspond to requested output variables
        if self._verbose:
            print("\t (2) Processing output variables...")

        outputs_col = {}
        if outputs:
            for key, val in outputs.items():
                outputs_col[key] = self.get_var(val, get_key=True)

        if self._verbose:
            print("\t\t...finished.")

        # perform simulation
        ####################

        if self._verbose:
            print("\t (3) Running the simulation...")
        t0 = time.perf_counter()

        # call backend run function
        results = self.graph.run(func=func, func_args=func_args, T=simulation_time, dt=self._dt, dts=sampling_step_size,
                                 outputs=outputs_col, solver=solver, **kwargs)

        if self._verbose:
            t1 = time.perf_counter()
            print(f"\t\t...finished after {t1-t0}s.")

        # return simulation results if outputs have been passed
        if outputs_col:
            return results

        # else, find the frontend variable names of the returned results and create a new results dict to return
        for key in results.copy():
            front_key = self.get_frontend_varname(key)
            results[front_key] = results.pop(key)
        return results

    def get_run_func(self, func_name: str, file_name: Optional[str] = None, **kwargs) -> tuple:

        if not file_name:
            file_name = f"pyrates_func"
        return self.graph.to_func(func_name=func_name, file_name=file_name, dt_adapt=self._dt_adapt, **kwargs)

    def network_to_computegraph(self, graph: NetworkGraph, inplace_vectorfield: bool = True, **kwargs):

        # initialize compute graph
        cg = ComputeGraph(**kwargs) if inplace_vectorfield else ComputeGraphBackProp(**kwargs)

        # add global time variable to compute graph
        cg.add_var(label="t", vtype="state_var", value=0.0 if self._dt_adapt else 0, shape=(),
                   dtype='float' if self._dt_adapt else 'int')

        # node operators
        parsing_kwargs = ['parsing_method']
        parsing_kwargs = {key: kwargs.pop(key) for key in parsing_kwargs if key in kwargs}

        self._parse_op_layers_into_computegraph(graph, cg, layers=[], exclude=True, op_identifier="edge_from_",
                                                **parsing_kwargs)

        # edge operators
        self._parse_op_layers_into_computegraph(graph, cg, layers=[0], exclude=False, op_identifier="edge_from_",
                                                **parsing_kwargs)

        return cg

    def getitem_from_iterator(self, key: str, key_iter: Iterator[str]):
        return self.graph.get_var(key)

    def clear(self):
        """Clears the backend graph from all operations and variables.
        """
        self._front_to_back.clear()
        self.graph.clear()
        in_edge_indices.clear()
        in_edge_vars.clear()

    def _parse_op_layers_into_computegraph(self, net: NetworkGraph, cg: ComputeGraph, layers: list,
                                           exclude: bool = False, op_identifier: Optional[str] = None, **kwargs
                                           ) -> None:
        """

        Parameters
        ----------
        layers
        exclude
        op_identifier
        kwargs

        Returns
        -------

        """

        for node_name, node in net.nodes.items():

            op_graph = node['node'].op_graph
            g = op_graph.copy()  # type: DiGraph

            # go through all operators on node and pre-process + extract equations and variables
            i = 0
            while g.nodes:

                # get all operators that have no dependencies on other operators
                # noinspection PyTypeChecker
                ops = [op for op, in_degree in g.in_degree if in_degree == 0]

                if (i in layers and not exclude) or (i not in layers and exclude):

                    # collect operator variables and equations from node
                    if op_identifier:
                        ops_tmp = [op for op in ops if op_identifier not in op] if exclude else \
                            [op for op in ops if op_identifier in op]
                    else:
                        ops_tmp = ops
                    op_eqs, op_vars = self._collect_ops(ops_tmp, node_name=node_name, graph=net, compute_graph=cg,
                                                        reduce=exclude)

                    # parse equations and variables into computegraph
                    variables = parse_equations(op_eqs, op_vars, cg=cg, def_shape=self._def_shape, **kwargs)

                    # remember mapping between frontend variable names and node keys in compute graph
                    for key, var in variables.items():
                        if key.split('/')[-1] != 'inputs' and isinstance(var, ComputeVar):
                            self._front_to_back[key] = var

                # remove parsed operators from graph
                g.remove_nodes_from(ops)
                i += 1

    def _collect_ops(self, ops: List[str], node_name: str, graph: NetworkGraph, compute_graph: ComputeGraph,
                     reduce: bool) -> tuple:
        """Adds a number of operations to the backend graph.

        Parameters
        ----------
        ops
            Names of the operators that should be parsed into the graph.
        node_name
            Name of the node that the operators belong to.
        graph
        compute_graph
        reduce

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
            scope = f"{node_name}/{op_name}"
            op_info = graph[f"{node_name}/{op_name}"]
            op_args = deepcopy(op_info['variables'])
            op_args['inputs'] = {}

            # handle operator inputs
            in_ops = {}
            for var_name, inp in op_info['inputs'].items():

                # go through inputs to variable
                if inp['sources']:

                    in_ops_col = {}
                    in_node = inp['node'] if 'node' in inp else node_name
                    in_var_tmp = inp.pop('var', None)

                    for i, in_op in enumerate(inp['sources']):

                        # collect single input to op
                        in_var = in_var_tmp if in_var_tmp else graph[f"{in_node}/{in_op}"]['output']
                        in_key = f"{in_node}/{in_op}/{in_var}"
                        try:
                            in_val = self._front_to_back[in_key]
                        except KeyError:
                            in_val = graph[in_key]
                        in_ops_col[in_key] = in_val

                    # if multiple inputs to variable, sum them up
                    if len(in_ops_col) > 1:
                        in_ops[var_name] = self._map_multiple_inputs(in_ops_col, scope=scope, tvar=var_name)
                    else:
                        key, _ = in_ops_col.popitem()
                        in_ops[var_name] = (None, {var_name: key})

            # replace input variables with input in operator equations
            for var, (eq, inp) in in_ops.items():
                if eq:
                    op_info['equations'] = [eq] + op_info['equations']
                op_args['inputs'].update(inp)

            # collect operator variables and equations
            variables[f"{scope}/inputs"] = {}
            equations += [(eq, scope) for eq in op_info['equations']]
            for key, var in op_args.items():

                full_key = f"{scope}/{key}"

                # case I: global time variable
                if key == "t":
                    variables[full_key] = compute_graph.get_var('t')

                # case II: input variables
                elif key == 'inputs' and var:
                    variables[f"{scope}/inputs"].update(var)
                    for in_var in var.values():
                        try:
                            variables[in_var] = self._front_to_back[in_var]
                        except KeyError:
                            variables[in_var] = graph[in_var]

                else:
                    try:
                        # case III: variables that have already been processed
                        variables[full_key] = self._front_to_back[full_key]
                    except KeyError:
                        # case IV: new variables
                        variables[full_key] = self._finalize_var_def(var, reduce)

        return equations, variables

    @staticmethod
    def _finalize_var_def(v: dict, reduce: bool):
        if not reduce:
            return v
        if not v:
            return v
        if v['vtype'] != 'constant':
            return v
        if v['dtype'] != 'float':
            return v
        if 'shape' in v and len(v['shape']) > 1:
            return v
        if len(np.unique(v['value'])) > 1:
            return v
        try:
            v['value'] = v['value'][0]
            v['shape'] = tuple()
        except (TypeError, IndexError):
            pass
        return v

    @staticmethod
    def _map_multiple_inputs(inputs: dict, scope: str, tvar: str) -> tuple:
        """Creates mapping between multiple input variables and a single output variable.

        Parameters
        ----------
        inputs
            Input variables.
        scope
            Scope of the input variables
        tvar
            Name of the input-receiving variable

        Returns
        -------
        tuple
            Equation that sums up all input variables and the mapping to the respective input variables
        """

        # preparations
        if scope not in in_edge_vars:
            in_edge_vars[scope] = []
        inputs_unique = in_edge_vars[scope]
        inputs_unique.append(tvar)
        input_mapping = {}
        new_input_vars = []

        # go through all inputs
        for key, var in inputs.items():

            # get a unique label for the input variable
            try:
                in_var = var.name
            except AttributeError:
                in_var = key.split('/')[-1]
            inp = get_unique_label(in_var, inputs_unique)

            # store input-related information
            new_input_vars.append(inp)
            inputs_unique.append(inp)
            input_mapping[inp] = key

        # collect input into single variable
        input_eq = f"{tvar} = {'+'.join(new_input_vars)}"

        return input_eq, input_mapping

    @property
    def nodes(self):
        """Shortcut to self.graph.nodes. See documentation of `networkx.MultiDiGraph.nodes`."""
        return self.graph.nodes

    @property
    def edges(self):
        """Shortcut to self.graph.edges. See documentation of `networkx.MultiDiGraph.edges`."""
        return self.graph.edges


def _get_indexed_var_str(var: str, idx: Union[tuple, str, list], var_length: int = None, reduce: bool = False,
                         idx_str: str = None, arg_dict: dict = None):
    if type(idx) is tuple:
        if var_length and int(idx[1]) - int(idx[0]) == var_length:
            return var
        elif not var_length:
            var = f"reshape({var}, 1)"
        return f"index_range({var}, {idx[0]}, {idx[1]})"
    if type(idx) is str and len(idx) > 0:
        if not var_length:
            var = f"reshape({var}, 1)"
        return f"index({var}, {idx})"
    if len(idx) > 0:
        if len(idx) == var_length:
            identical = True
            for i1, i2 in zip(idx, list(np.arange(0, var_length))):
                if i1 != i2:
                    identical = False
                    break
            if identical:
                return var
        if idx_str:
            arg_dict[idx_str] = {'vtype': 'constant', 'value': idx, 'dtype': 'int', 'shape': (len(idx),)}
            return f"index({var}, {idx_str})"
        return f"index({var}, {idx})"
    if reduce:
        return f"index({var}, {idx[0]})"
    return var


def _get_source_str():
    pass


def _get_target_str():
    pass
