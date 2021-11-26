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
import time
from typing import Union, Dict, Iterator, Optional, List, Tuple
from warnings import filterwarnings
from networkx import MultiDiGraph, DiGraph
import numpy as np
from copy import deepcopy

# pyrates-internal imports
from pyrates.backend import PyRatesException
from pyrates.ir.node import NodeIR
from pyrates.ir.edge import EdgeIR
from pyrates.ir.abc import AbstractBaseIR
from pyrates.backend.parser import parse_equations, replace, get_unique_label
from pyrates.backend.computegraph import ComputeGraph, ComputeVar

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
                 float_precision: str = "float64", **kwargs):

        # TODO: differentiate between vectorized and non-vectorized NetworkGraph

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
        self._preprocess_edge_operations(float_precision=float_precision, dde_approx=kwargs.pop('dde_approx', 0),
                                         matrix_sparseness=kwargs.pop('matrix_sparseness', 0.05))

        if verbose:
            print("\t\t...finished.")

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

    def _preprocess_edge_operations(self, float_precision: str = "float64", dde_approx: int = 0, **kwargs):
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
                    self._add_edge_buffer(node_name, op_name, var_name, edges=edges, delays=delays,
                                          nodes=nodes, spreads=spreads, dde_approx=dde_approx,
                                          float_precision=float_precision)

        # create the final equations and variables for each edge
        for source, targets in self.graph.adjacency():
            for target, edges in targets.items():
                for idx, data in edges.items():
                    self._generate_edge_equation(source_node=source, target_node=target, edge_idx=idx, data=data,
                                                 **kwargs)

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
            d = self.edges[s, t, e]['delay']
            if type(d) is list:
                d = [1 if d_tmp is None else d_tmp for d_tmp in d]
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

    def _add_edge_buffer(self, node: str, op: str, var: str, edges: list, delays: list, nodes: list,
                         spreads: Optional[list] = None, dde_approx: int = 0, float_precision: str = "float_precision"
                         ) -> None:
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
        float_precision

        Returns
        -------
        None

        """

        max_delay = np.max(delays)

        # extract target shape and node
        node_var = self[f"{node}/{op}/{var}"]
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

            # sort all edge information in ascending ODE order
            order_idx = np.argsort(orders, kind='stable')
            orders_sorted = np.asarray(orders, dtype=np.int32)[order_idx]
            orders_tmp = np.asarray(orders, dtype=np.int32)[order_idx]
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
                var_next = f"{var}_d{k}"
                var_prev = f"{var}_d{i}" if i > 0 else var
                rate = f"k_d{k}"

                # prepare variables for the next ODE
                idx_apply = len(idx) != len(orders_tmp)
                val = rates_tmp[idx] if idx_apply else rates_tmp
                var_shape = (len(val),) if val.shape else ()
                if i == 0 and idx != [0] and (sum(target_shape) != len(idx) or any(np.diff(order_idx) != 1)):
                    var_prev_idx = f"index({var_prev}, source_idx)"
                    var_dict["source_idx"] = {'vtype': 'constant',
                                              'dtype': 'int32',
                                              'shape': (len(source_idx_tmp[idx]),),
                                              'value': source_idx_tmp[idx]}
                elif i != 0 and idx_apply:
                    var_prev_idx = get_indexed_var_str(var_prev, idx_str, var_length=len(rates_tmp))
                else:
                    var_prev_idx = var_prev

                # create new ODE string and corresponding variable definitions
                buffer_eqs.append(f"d/dt * {var_next} = {rate} * ({var_prev_idx} - {var_next})")
                var_dict[var_next] = {'vtype': 'state_var',
                                      'dtype': float_precision,
                                      'shape': var_shape,
                                      'value': 0.}
                var_dict[rate] = {'vtype': 'constant',
                                  'dtype': float_precision,
                                  'value': val}

                # store indices that are required to fill the edge buffer variable
                if idx_apply:

                    # right-hand side index
                    if len(orders_tmp) < 2:
                        idx_rhs_str = ''
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
                    orders_tmp = np.asarray([orders_tmp], dtype=np.int32)
                    rates_tmp = np.asarray([rates_tmp])

            # remove unnecessary ODEs
            for _ in range(len(buffer_eqs) - final_idx[-1][0]):
                i = len(buffer_eqs)
                var_dict.pop(f"{var}_d{i}")
                var_dict.pop(f"k_d{i}")
                buffer_eqs.pop(-1)

            # create edge buffer variable
            buffer_length = len(delays)
            for i, idx_l, idx_r in final_idx:
                lhs = get_indexed_var_str(f"{var}_buffered", idx_l, var_length=buffer_length)
                rhs = get_indexed_var_str(f"{var}_d{i}" if i != 0 else var, idx_r, var_length=buffer_length)
                buffer_eqs.append(f"{lhs} = {rhs}")
            var_dict[f"{var}_buffered"] = {'vtype': 'state_var',
                                           'dtype': float_precision,
                                           'shape': (buffer_length,),
                                           'value': 0.0}

            # re-order buffered variable if necessary
            if any(np.diff(order_idx) != 1):
                buffer_eqs.append(f"{var}_buffered = index({var}_buffered, {var}_buffered_idx)")
                var_dict[f"{var}_buffered_idx"] = {'vtype': 'constant',
                                                   'dtype': 'int32',
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
            var_dict = {f'{var}_buffer': {'vtype': 'input',
                                          'dtype': float_precision,
                                          'shape': buffer_shape,
                                          'value': 0.},
                        f'{var}_buffered': {'vtype': 'state_var',
                                            'dtype': float_precision,
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
                buffer_eqs = [f"index_axis({var}_buffer) = roll({var}_buffer, 1, 0)",
                              f"index({var}_buffer, 0) = {var}",
                              f"{var}_buffered = index({var}_buffer, {var}_delays)"]
            else:
                buffer_eqs = [f"index_axis({var}_buffer) = roll({var}_buffer, 1, 1)",
                              f"index_axis({var}_buffer, 0, 1) = {var}",
                              f"{var}_buffered = index_2d({var}_buffer, source_idx, {var}_delays)"]

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
                                          'dtype': float_precision,
                                          'shape': buffer_shape,
                                          'value': 0.
                                          },
                        'times': {'vtype': 'state_var',
                                  'dtype': float_precision,
                                  'shape': (len(times),),
                                  'value': np.asarray(times)
                                  },
                        't': {'vtype': 'state_var',
                              'dtype': float_precision,
                              'shape': (),
                              'value': 0.0
                              },
                        f'{var}_buffered': {'vtype': 'state_var',
                                            'dtype': float_precision,
                                            'shape': (len(delays),),
                                            'value': 0.},
                        f'{var}_delays': {'vtype': 'constant',
                                          'dtype': float_precision,
                                          'value': delays},
                        f'source_idx': {'vtype': 'constant',
                                        'dtype': 'int32',
                                        'value': source_idx},
                        f'{var}_maxdelay': {'vtype': 'constant',
                                            'dtype': float_precision,
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

        # add buffer equations to node operator
        op_info = node_ir[op]
        op_info['equations'] += buffer_eqs
        op_info['variables'].update(var_dict)
        op_info['output'] = f"{var}_buffered"

        # update input information of node operators connected to this operator
        for succ in node_ir.op_graph.succ[op]:
            inputs = self[f"{node}/{succ}"]['inputs']
            if var not in inputs.keys():
                inputs[var] = {'sources': {op}}

        # update edge information
        idx_l = 0
        for i, edge in enumerate(edges):
            s, t, e = edge
            self.edges[s, t, e]['source_var'] = f"{op}/{var}_buffered"
            if len(edges) > 1:
                idx_h = idx_l + len(nodes[i])
                self.edges[s, t, e]['source_idx'] = list(range(idx_l, idx_h))
                idx_l = idx_h

    def _generate_edge_equation(self, source_node: str, target_node: str, edge_idx: int, data: dict,
                                matrix_sparseness: float = 0.1, float_precision: str = "float64"):

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
                eq = f"{tvar} = matmul(weight,{svar})"
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
        index_svar = sidx and sum(sval['shape']) > 1
        svar_final = f"index({svar},source_idx)" if index_svar else svar
        if not dot_edge:
            if len(d) > 1:
                eq = f"{tvar} = matmul(target_idx, {svar_final}*weight)"
            elif len(d):
                eq = f"index({tvar},target_idx) = {svar_final} * weight"
            else:
                eq = f"{tvar} = {svar_final} * weight"

        # add edge variables to dict
        dtype = sval["dtype"]
        args['weight'] = {'vtype': 'constant', 'dtype': dtype, 'value': weight}
        args[tvar] = deepcopy(tval)
        args[tvar]['vtype'] = 'variable'
        if len(d):
            args['target_idx'] = {'vtype': 'constant',
                                  'value': np.array(d, dtype=float_precision if len(
                                      d) > 1 else np.int32)}
        if index_svar:
            args['source_idx'] = {'vtype': 'constant', 'dtype': 'int32',
                                  'value': np.array(sidx, dtype=np.int32)}

        # add edge operator to target node
        if target_node not in in_edge_indices:
            in_edge_indices[target_node] = 0
        op_name = f'incoming_edge_{in_edge_indices[target_node]}'
        in_edge_indices[target_node] += 1
        target_node_ir.add_op(op_name,
                              inputs={svar: {'sources': [sop],
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
            var_name = f"delay_idx_{self._edge_idx_counter}"
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
                 verbose: bool = True, float_precision: str = 'float64', backend: str = None,
                 scalar_shape: tuple = None, **kwargs):
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
                           step_size_adaptation=step_size_adaptation, verbose=verbose, float_precision=float_precision,
                           **kwargs)

        # parse network equations into a compute graph
        if verbose:
            print("\t(3) Parsing the model equations into a compute graph...")

        self.graph = self.network_to_computegraph(graph=net, backend=backend, **kwargs)

        if verbose:
            print("\t\t...finished.")
            print("\tModel compilation was finished.")

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

        # TODO: Move method to CircuitTemplate

        raise NotImplementedError('This method does not work. Sry.')

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

    def run(self,
            simulation_time: float,
            outputs: Optional[dict] = None,
            sampling_step_size: Optional[float] = None,
            solver: str = 'euler',
            **kwargs
            ) -> Union[dict, Tuple[dict, float]]:
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
        Union[DataFrame, Tuple[DataFrame, float]]
            First entry of the tuple contains the output variables in a pandas dataframe, the second contains the
            simulation time in seconds. If profiling was not chosen during call of the function, only the dataframe
            will be returned.

        """

        filterwarnings("ignore", category=FutureWarning)

        # collect backend variables and functions
        #########################################

        if self._verbose:
            print("Simulation Progress")
            print("-------------------")
            print("\t (1) Generating the network run function...")

        # generate run function
        func_name = kwargs.pop('func_name', 'run')
        func, func_args = self.get_run_func(func_name, **kwargs)

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
            v = self.get_var(key)
            front_keys = list(self._front_to_back.keys())
            front_vs = list(self._front_to_back.values())
            idx = front_vs.index(v)
            results[front_keys[idx]] = results.pop(key)
        return results

    def get_run_func(self, func_name: str, file_name: Optional[str] = None, **kwargs) -> tuple:

        if not file_name:
            file_name = f"pyrates_func"
        return self.graph.to_func(func_name=func_name, file_name=file_name, **kwargs)

    def network_to_computegraph(self, graph: NetworkGraph, **kwargs):

        # initialize compute graph
        cg = ComputeGraph(**kwargs)

        # add global time variable to compute graph
        cg.add_var(label="t", vtype="state_var", value=0.0 if self._dt_adapt else 0)

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
                    op_eqs, op_vars = self._collect_ops(ops_tmp, node_name=node_name, graph=net, compute_graph=cg)

                    # parse equations and variables into computegraph
                    variables = parse_equations(op_eqs, op_vars, cg=cg, def_shape=self._def_shape, **kwargs)

                    # remember mapping between frontend variable names and node keys in compute graph
                    for key, var in variables.items():
                        if key.split('/')[-1] != 'inputs' and isinstance(var, ComputeVar):
                            self._front_to_back[key] = var

                # remove parsed operators from graph
                g.remove_nodes_from(ops)
                i += 1

    def _collect_ops(self, ops: List[str], node_name: str, graph: NetworkGraph, compute_graph: ComputeGraph) -> tuple:
        """Adds a number of operations to the backend graph.

        Parameters
        ----------
        ops
            Names of the operators that should be parsed into the graph.
        node_name
            Name of the node that the operators belong to.
        graph
        compute_graph

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

                    if len(in_ops_col) > 1:
                        in_ops[var_name] = self._map_multiple_inputs(in_ops_col, scope=scope)
                    else:
                        key, _ = in_ops_col.popitem()
                        *in_node, in_op, in_var = key.split("/")
                        in_ops[var_name] = (in_var, {in_var: key})

            # replace input variables with input in operator equations
            for var, inp in in_ops.items():
                for i, eq in enumerate(op_info['equations']):
                    op_info['equations'][i] = replace(eq, var, inp[0], rhs_only=True)
                op_args['inputs'].update(inp[1])

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
                        variables[full_key] = var

        return equations, variables

    @staticmethod
    def _map_multiple_inputs(inputs: dict, scope: str) -> tuple:
        """Creates mapping between multiple input variables and a single output variable.

        Parameters
        ----------
        inputs
            Input variables.
        scope
            Scope of the input variables

        Returns
        -------
        tuple
            Summed up or concatenated input variables and the mapping to the respective input variables

        """

        # preparations
        if scope not in in_edge_vars:
            in_edge_vars[scope] = []
        inputs_unique = in_edge_vars[scope]
        input_mapping = {}
        new_input_vars = []

        # go through all inputs
        for key, var in inputs.items():

            # get a unique label for the input variable
            try:
                in_var = var.name
            except AttributeError:
                in_var = var['symbol'].name
            inp = get_unique_label(in_var, inputs_unique)

            # store input-related information
            new_input_vars.append(inp)
            inputs_unique.append(inp)
            input_mapping[inp] = key

        # sum up all inputs
        input_expr = f"sum(({','.join(new_input_vars)}), 0)"

        return input_expr, input_mapping

    @property
    def nodes(self):
        """Shortcut to self.graph.nodes. See documentation of `networkx.MultiDiGraph.nodes`."""
        return self.graph.nodes

    @property
    def edges(self):
        """Shortcut to self.graph.edges. See documentation of `networkx.MultiDiGraph.edges`."""
        return self.graph.edges


def get_indexed_var_str(var: str, idx: Union[tuple, str], var_length: int = None):
    if type(idx) is tuple:
        if var_length and int(idx[1]) - int(idx[0]) == var_length:
            return var
        elif not var_length:
            var = f"reshape({var}, 1)"
        return f"index_range({var}, {idx[0]}, {idx[1]})"
    if idx:
        if not var_length:
            var = f"reshape({var}, 1)"
        return f"index({var}, {idx})"
    return var
