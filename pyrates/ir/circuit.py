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
import re as _re
from networkx import MultiDiGraph, DiGraph, topological_sort
import numpy as np
from copy import deepcopy
from warnings import warn

# pyrates-internal _imports
from pyrates.backend import PyRatesException, PyRatesWarning
from pyrates.ir.node import NodeIR
from pyrates.ir.edge import EdgeIR
from pyrates.ir.abc import AbstractBaseIR
from pyrates.backend.parser import parse_equations, get_unique_label, replace
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

                # Separate matrix-connectivity edges (2-D weight array) from scalar edges.
                # Matrix edges carry their own delay handling via _add_matrix_delay.
                matrix_edges, scalar_edges = [], []
                for s, t, e in edges:
                    w = self.edges[s, t, e]['weight']
                    if isinstance(w, np.ndarray) and w.ndim == 2:
                        matrix_edges.append((s, t, e))
                    else:
                        scalar_edges.append((s, t, e))

                for s, t, e in matrix_edges:
                    d = self.edges[s, t, e].get('delay')
                    v = self.edges[s, t, e].get('spread')
                    if d is not None and d > self.step_size:
                        self._add_matrix_delay(node_name, op_name, var_name, (s, t, e),
                                               d, v, dde_approx=dde_approx)

                if not scalar_edges:
                    continue

                delays, spreads, nodes, add_delay = self._collect_delays_from_edges(scalar_edges)

                # add synaptic buffer to output variables with delay
                if add_delay:
                    # Clear delay fields from edges so _generate_edge_equation ignores them.
                    # Kept here (not inside _collect_delays_from_edges) so that method is pure.
                    for s, t, e in scalar_edges:
                        self.edges[s, t, e]['source_idx'] = []
                        self.edges[s, t, e]['delay'] = None

                    if vectorized:
                        self._add_edge_buffer(node_name, op_name, var_name, edges=scalar_edges, delays=delays,
                                              nodes=nodes, spreads=spreads, dde_approx=dde_approx)
                    else:
                        # TODO: sort edges into unique delay/spread combinations and only loop over those
                        if spreads:
                            for i, (edge, delay, spread, node) in enumerate(zip(scalar_edges, delays, spreads, nodes)):
                                self._add_edge_buffer(node_name, op_name, var_name, edges=[edge], delays=[delay],
                                                      nodes=[node], spreads=[spread], dde_approx=dde_approx,
                                                      buffer_id=f"_out{i}")
                        else:
                            for i, (edge, delay, node) in enumerate(zip(scalar_edges, delays, nodes)):
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
                                                keys=['source_var', 'weight', 'source_idx', 'target_idx',
                                                      'edge_ir', 'edge_var_map'])

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
            n_slots = max(len(self.edges[s, t, e]['target_idx']), 1)
            if v is None or np.sum(v) == 0:
                v = [0] * n_slots
                discretize = True
            else:
                discretize = False
                v = self._process_delays(v, discretize=discretize)

            # finalize edge delay
            if d is None or np.sum(d) == 0:
                d = [1] * n_slots
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

        return means, stds, nodes, add_delay

    def _collect_from_edges(self, edges: list, keys: list):
        data = dict()
        for source, target, idx in edges:
            edge = self.edges[(source, target, idx)]
            if source not in data:
                data[source] = dict()
            for key in keys:
                raw = edge.get(key)
                val = raw if isinstance(raw, (np.ndarray, EdgeIR)) else deepcopy(raw)
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

    def _add_matrix_delay(self, node: str, op: str, var: str, edge: tuple,
                          delay: float, spread: Optional[float] = None,
                          dde_approx: int = 0, buffer_id: str = "") -> None:
        """Add delay buffer equations for a matrix-connectivity edge.

        Supports two modes:
        - **Discrete ring buffer** (default, fixed step size): ``(Ns, d+1)`` state variable,
          updated by roll/index_axis each step.  Selected when *spread* is ``None``,
          *dde_approx* is 0, and ``step_size_adaptation`` is ``False``.
        - **ODE cascade** (gamma-kernel or explicit order): a chain of *n* ODEs of shape
          ``(Ns,)`` that convolve the source signal with a gamma kernel.  Selected when
          *spread* > 0, *dde_approx* > 0, or ``step_size_adaptation`` is ``True``.
        """
        s, t, e = edge
        node_ir = self[node]
        op_info = node_ir[op]

        node_var = self[f"{node}/{op}/{var}"]
        var_shape = node_var.get('shape', ())
        Ns = int(var_shape[0]) if var_shape else 1

        var_dict: dict = {}
        buffer_eqs: list = []

        use_ring_buffer = (
            (spread is None or spread == 0)
            and dde_approx == 0
            and not self.step_size_adaptation
        )

        if use_ring_buffer:
            # --- Discrete ring buffer of shape (Ns, d_steps+1) ---
            d_steps = self._preprocess_delay(delay, discretize=True)
            buf = f'{var}_buffer{buffer_id}'
            buf_out = f'{var}_buffered{buffer_id}'
            var_dict[buf] = {'vtype': 'variable', 'dtype': 'float',
                             'shape': (Ns, d_steps + 1), 'value': 0.}
            var_dict[buf_out] = {'vtype': 'variable', 'dtype': 'float',
                                 'shape': (Ns,), 'value': 0.}
            # Inline d_steps as a literal so index_axis returns shape (Ns,) not (Ns, 1)
            buffer_eqs = [
                f"index_axis({buf}) = roll({buf}, 1, 1)",
                f"index_axis({buf}, 0, 1) = {var}",
                f"{buf_out} = index_axis({buf}, {d_steps}, 1)",
            ]
        else:
            # --- ODE cascade (gamma kernel or adaptive step size) ---
            if spread is not None and spread > 0:
                n = max(1, int(round((delay / spread) ** 2)))
            elif dde_approx > 0:
                n = dde_approx
            else:
                n = 1  # minimum ODE order for adaptive step size
            a = n / delay if delay else 0.0
            for k in range(1, n + 1):
                zk = f'{var}_d{k}{buffer_id}'
                zk_rate = f'k_d{k}{buffer_id}'
                prev = var if k == 1 else f'{var}_d{k-1}{buffer_id}'
                var_dict[zk] = {'vtype': 'state_var', 'dtype': 'float',
                                'shape': (Ns,), 'value': [0.0] * Ns}
                var_dict[zk_rate] = {'vtype': 'constant', 'dtype': 'float',
                                     'value': a, 'shape': (1,)}
                buffer_eqs.append(f"d/dt * {zk} = {zk_rate} * ({prev} - {zk})")
            buf_out = f'{var}_buffered{buffer_id}'
            var_dict[buf_out] = {'vtype': 'variable', 'dtype': 'float',
                                 'shape': (Ns,), 'value': [0.0] * Ns}
            buffer_eqs.append(f"{buf_out} = {var}_d{n}{buffer_id}")

        # Attach buffer equations and variables to the source operator
        op_info['equations'] += buffer_eqs
        op_info['variables'].update(var_dict)
        op_info['output'] = buf_out

        # Update intra-node successor inputs (mirrors _add_edge_buffer)
        for succ in node_ir.op_graph.succ[op]:
            inputs = self[f"{node}/{succ}"]['inputs']
            if var not in inputs:
                inputs[var] = {'sources': {op}}

        # Point the edge at the buffered source variable
        self.edges[s, t, e]['source_var'] = f"{op}/{buf_out}"
        self.edges[s, t, e]['delay'] = None

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

        if not delays:
            return

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

            # --- Per-edge ODE orders and rates ---
            if spreads:
                orders, rates = [], []
                for m, v in zip(delays, spreads):
                    if v > 0:
                        n_order = int(np.round((m / v) ** 2))
                        n_order = n_order if m and n_order > dde_approx else dde_approx
                    else:
                        n_order = dde_approx if m else 0
                    orders.append(n_order)
                    rates.append(n_order / m if m else 0.0)
            else:
                orders = [dde_approx if m else 0 for m in delays]
                rates = [dde_approx / m if m else 0.0 for m in delays]

            # --- Group delay slots by (order, rate) — slots in the same group share one ODE chain ---
            groups = {}
            for slot_idx, (n_order, rate, src) in enumerate(zip(orders, rates, source_idx)):
                key = (n_order, round(rate, 12))
                if key not in groups:
                    groups[key] = []
                groups[key].append((slot_idx, int(src)))

            n_src_var = sum(target_shape) if target_shape else 1
            buffer_eqs, var_dict = [], {}
            buf_var = f"{var}_buffered{buffer_id}"
            var_dict[buf_var] = {'vtype': 'variable', 'dtype': 'float',
                                 'shape': (len(delays),), 'value': 0.0}

            for chain_id, ((n_order, _), group) in enumerate(groups.items()):
                slot_indices = [g[0] for g in group]
                src_indices  = [g[1] for g in group]
                G = len(group)
                rate_val = rates[slot_indices[0]]

                # Build chain input: use source var directly when group covers all its elements
                if sorted(src_indices) == list(range(n_src_var)):
                    chain_in = var
                elif G == 1:
                    chain_in = f"index({var}, {src_indices[0]})"
                else:
                    src_name = f"{var}_src{chain_id}{buffer_id}"
                    var_dict[src_name] = {'vtype': 'constant', 'dtype': 'int',
                                          'value': np.asarray(src_indices, dtype='int'),
                                          'shape': (G,)}
                    chain_in = f"index({var}, {src_name})"

                # Build the ODE chain (n_order stages, one shared rate constant)
                chain_shape = (G,) if G > 1 else ()
                if n_order > 0:
                    rate_name = f"k_d{chain_id}{buffer_id}"
                    var_dict[rate_name] = {'vtype': 'constant', 'dtype': 'float', 'value': rate_val}
                    prev = chain_in
                    for k in range(1, n_order + 1):
                        zk = f"{var}_d{chain_id}_{k}{buffer_id}"
                        var_dict[zk] = {'vtype': 'state_var', 'dtype': 'float',
                                        'shape': chain_shape, 'value': 0.}
                        buffer_eqs.append(f"d/dt * {zk} = {rate_name} * ({prev} - {zk})")
                        prev = zk
                else:
                    prev = chain_in  # zero-order: pass-through (no ODE stages)

                # Write chain output into the correct slots of the buffer
                all_slots = (slot_indices == list(range(len(delays))))
                if all_slots:
                    buffer_eqs.append(f"{buf_var} = {prev}")
                elif G == 1:
                    buffer_eqs.append(f"index({buf_var}, {slot_indices[0]}) = {prev}")
                else:
                    slot_name = f"{var}_slots{chain_id}{buffer_id}"
                    var_dict[slot_name] = {'vtype': 'constant', 'dtype': 'int',
                                           'value': np.asarray(slot_indices, dtype='int'),
                                           'shape': (len(slot_indices),)}
                    buffer_eqs.append(f"index({buf_var}, {slot_name}) = {prev}")

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
                # For a single delay, inline the literal so buffer[int] returns a 0-d scalar
                # rather than buffer[(1,)_array] which returns a (1,) array and causes
                # numpy 2.3+ errors when assigned to a scalar dy[i] slot.
                delay_ref = str(delays[0]) if len(delays) == 1 else f"{var}_delays{buffer_id}"
                buffer_eqs = [f"index_axis({var}_buffer{buffer_id}) = roll({var}_buffer{buffer_id}, 1)",
                              f"index({var}_buffer{buffer_id}, 0) = {var}",
                              f"{var}_buffered{buffer_id} = index({var}_buffer{buffer_id}, {delay_ref})"]
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
                var_delayed = f"past({var}, {d})" if type(d) is float or d != 1 else var
                if len(target_shape) < 1 or (len(target_shape) == 1 and target_shape[0] == 1):
                    buffer_eqs.append(f"{var}_buffered{buffer_id} = {var_delayed}")
                else:
                    buffer_eqs.append(f"index({var}_buffered{buffer_id}, {sidx}) = index({var_delayed}, {sidx})")

        # add buffer equations to node operator
        op_info = node_ir[op]
        existing_vars = set(op_info.get('variables', {}).keys())
        conflicts = existing_vars & set(var_dict.keys())
        if conflicts:
            raise PyRatesException(
                f"Buffer variable name collision in operator '{op}' on node '{node}': {conflicts}. "
                f"Use a unique buffer_id to avoid this."
            )
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
        edge_irs, edge_var_maps = [], []
        for snode, sinfo in inputs.items():
            weights.append(sinfo['weight'])
            source_indices.append(sinfo['source_idx'])
            target_indices.append(sinfo['target_idx'])
            sources.append((snode,) + tuple(sinfo['source_var'].split('/')))
            edge_irs.append(sinfo.get('edge_ir'))
            edge_var_maps.append(sinfo.get('edge_var_map') or {})

        # step 2: process incoming edges
        source_vars, args = {}, {}
        eqs, in_vars = [], []
        for i, (weight, sidx, tidx, (snode, sop, svar), edge_ir, edge_var_map) in \
                enumerate(zip(weights, source_indices, target_indices, sources, edge_irs, edge_var_maps)):

            # define variable name strings (adjusted when multiple inputs share same target var)
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

            # case 0: matrix edge — weight is a 2-D numpy array supplied directly
            # (used by Connectivity; no scalar expansion needed)
            if isinstance(weight, np.ndarray) and weight.ndim == 2:
                source_vars[s_str] = {'sources': [sop], 'node': snode, 'var': svar}
                # Always register the full 2-D weight for cases 0b/0c (wsum uses it);
                # case 0a may override with a 1-D vector for the single-source path.
                args[w_str] = {'vtype': 'constant', 'value': weight, 'dtype': 'float', 'shape': weight.shape}

                if edge_ir is None:
                    # case 0a: simple matvec — no coupling function
                    # When n_source == 1 the source variable is a scalar at runtime.
                    # np.dot((n,1), scalar) returns (n,1) which causes shape errors
                    # when assigned to an (n,) target. Squeeze axis=1 to a 1-D weight
                    # vector and use broadcast multiply instead.
                    if weight.shape[1] == 1:
                        w_1d = weight.squeeze(axis=1)
                        args[w_str] = {'vtype': 'constant', 'value': w_1d, 'dtype': 'float', 'shape': w_1d.shape}
                        eqs.append(f"{t_str} = {w_str} * {s_str}")
                    else:
                        eqs.append(f"{t_str} = matvec({w_str}, {s_str})")
                else:
                    # case 0b / 0c: matrix coupling with custom edge equations

                    def _subst(text, subst_map):
                        for var, repl in subst_map.items():
                            text = _re.sub(r'\b' + _re.escape(var) + r'\b', repl, text)
                        return text

                    def _de_lhs_var(lhs):
                        """Extract bare variable name from a DE left-hand side."""
                        return _re.sub(r"d/dt\s*\*?\s*|'", "", lhs).strip()

                    Nt, Ns = weight.shape

                    # Detect which edge variables are state variables (have DEs)
                    edge_de_sv_names = set()
                    for _ok in edge_ir.op_graph.nodes:
                        for _eq in edge_ir.op_graph.nodes[_ok].get('equations', []):
                            _lhs = _eq.split('=')[0].strip()
                            if "d/dt" in _lhs or "'" in _lhs:
                                edge_de_sv_names.add(_de_lhs_var(_lhs))

                    # Build broadcast substitution map for edge input variables
                    expr_map = {}
                    for ev, info in edge_var_map.items():
                        if info['role'] == 'source':
                            expr_map[ev] = f'broadcast_pre({s_str})'
                        else:
                            post_var = info['var']
                            post_op = info['op']
                            expr_map[ev] = f'broadcast_post({post_var})'
                            source_vars[post_var] = {'sources': [post_op], 'node': tnode, 'var': post_var}

                    if edge_de_sv_names:
                        # case 0c: dynamic edge
                        # State variables are stored flat (Nt*Ns,) in the global state vector.
                        # reshape2d / flatten1d views are used in generated equations so that
                        # the ODE operates in (Nt, Ns) space while the solver sees a 1-D vector.

                        for _ok in edge_ir.op_graph.nodes:
                            for vk, vi in edge_ir.op_graph.nodes[_ok].get('variables', {}).items():
                                vi_dict = vi if isinstance(vi, dict) else {}
                                vtype = vi_dict.get('vtype', 'constant')

                                if vk in edge_de_sv_names:
                                    sv_flat = f'{vk}_edge{i}_flat'
                                    sv_init = vi_dict.get('value', 0.0)
                                    if isinstance(sv_init, list):
                                        sv_init = sv_init[0]
                                    expr_map[vk] = f'reshape2d({sv_flat}, {Nt}, {Ns})'
                                    args[sv_flat] = {
                                        'vtype': 'state_var', 'dtype': 'float',
                                        'value': [float(sv_init)] * (Nt * Ns),
                                        'shape': (Nt * Ns,),
                                    }
                                elif vtype == 'constant' and vk not in edge_var_map:
                                    const_name = f'{vk}_edge{i}'
                                    val = vi_dict.get('value', 0.0)
                                    if isinstance(val, list):
                                        val = val[0]
                                    args[const_name] = {
                                        'vtype': 'constant', 'dtype': 'float',
                                        'value': float(val), 'shape': (1,),
                                    }
                                    expr_map[vk] = const_name

                        last_out = None
                        for _ok in topological_sort(edge_ir.op_graph):
                            _od = edge_ir.op_graph.nodes[_ok]
                            for _eq in _od.get('equations', []):
                                _lhs, _rhs = (_s.strip() for _s in _eq.split('=', 1))
                                _rhs_s = _subst(_rhs, expr_map)
                                if "d/dt" in _lhs or "'" in _lhs:
                                    sv_flat = f'{_de_lhs_var(_lhs)}_edge{i}_flat'
                                    eqs.append(f"{sv_flat}' = flatten1d({_rhs_s})")
                                else:
                                    expr_map[_lhs] = _rhs_s
                            last_out = _od.get('output')

                        final_expr = expr_map.get(last_out, last_out)
                        eqs.append(f"{t_str} = wsum({w_str}, {final_expr})")

                    else:
                        # case 0b: non-dynamic (algebraic) edge — inline and reduce
                        last_out = None
                        for _ok in topological_sort(edge_ir.op_graph):
                            _od = edge_ir.op_graph.nodes[_ok]
                            for _eq in _od.get('equations', []):
                                _lhs, _rhs = (_s.strip() for _s in _eq.split('=', 1))
                                expr_map[_lhs] = _subst(_rhs, expr_map)
                            last_out = _od.get('output')

                        final_expr = expr_map.get(last_out, last_out)
                        eqs.append(f"{t_str} = wsum({w_str}, {final_expr})")

                in_vars.append(t_str)
                continue

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
                if len(sidx_unique) == 1:
                    # Single-source: the source variable is a scalar at runtime
                    # (time-series arrays are squeezed to 1D so arr[t] returns 0-d).
                    # Use broadcast multiply with a 1D weight vector so that
                    # weight_vec * scalar = (n_targets,) rather than the (n_targets,1)
                    # result that numpy's dot gives for a 2D matrix times a scalar.
                    weight_mat = weight_mat.squeeze(axis=1)
                    eq = f"{t_str_final} = {w_str} * {s_str_final}"
                else:
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
        args[tvar] = tval
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
        """Simulate the backend behavior over time.

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
        return self.graph.to_func(func_name=func_name, file_name=file_name, dt_adapt=self._dt_adapt,
                                   dt=self._dt, **kwargs)

    def get_jacobian_func(self, func_name: str, file_name: Optional[str] = None, **kwargs) -> tuple:

        if not file_name:
            file_name = "pyrates_func"
        return self.graph.get_jacobian_func(func_name=func_name, file_name=file_name,
                                             dt_adapt=self._dt_adapt, dt=self._dt, **kwargs)

    def network_to_computegraph(self, graph: NetworkGraph, inplace_vectorfield: bool = True, **kwargs):

        # initialize compute graph
        cg = ComputeGraph(**kwargs) if inplace_vectorfield else ComputeGraphBackProp(**kwargs)

        # add global time variable to compute graph
        cg.add_var(label="t", vtype="state_var", value=0.0 if self._dt_adapt else 0, shape=(),
                   dtype='float' if self._dt_adapt else 'int')

        # node operators
        parsing_kwargs = ['parsing_method', 'vectorized']
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
                                           exclude: bool = False, op_identifier: Optional[str] = None,
                                           vectorized: bool = False, **kwargs) -> None:
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
                                                        reduce=exclude, vectorized=vectorized)

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
                     reduce: bool, vectorized: bool) -> tuple:
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
        vectorized

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
                        in_ops[var_name] = self._map_multiple_inputs(in_ops_col, scope=scope)
                    else:
                        key, _ = in_ops_col.popitem()
                        in_ops[var_name] = (None, {var_name: key})

            # replace input variables with input in operator equations
            for var, (inp_term, inp) in in_ops.items():
                if inp_term:
                    op_info['equations'] = [replace(eq, var_name, inp_term) for eq in op_info['equations']]
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
                            variables[in_var] = self._finalize_var_def(graph[in_var], reduce, vectorized)

                else:
                    try:
                        # case III: variables that have already been processed
                        variables[full_key] = self._front_to_back[full_key]
                    except KeyError:
                        # case IV: new variables
                        variables[full_key] = self._finalize_var_def(var, reduce, vectorized)

        return equations, variables

    @staticmethod
    def _finalize_var_def(v: dict, reduce: bool, vectorized: bool):
        if not reduce:
            return v
        if not v:
            return v
        if v['vtype'] != "constant" and vectorized:
            return v
        if v['dtype'] != 'float':
            return v
        if 'shape' in v and len(v['shape']) > 1:
            return v
        if 'shape' in v and np.prod(v['shape']) > 1:
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
            Equation that sums up all input variables and the mapping to the respective input variables
        """

        # preparations
        if scope not in in_edge_vars:
            in_edge_vars[scope] = {}
        inputs_unique = in_edge_vars[scope]
        input_mapping = {}
        new_input_vars = []

        # go through all inputs
        for key, var in inputs.items():

            # get a unique label for the input variable
            try:
                in_var = var.name
            except AttributeError:
                in_var = key.split('/')[-1]
            inp, inputs_unique_tmp = get_unique_label(in_var, inputs_unique)
            inputs_unique.update(inputs_unique_tmp)

            # store input-related information
            new_input_vars.append(inp)
            input_mapping[inp] = key

        # collect input into single variable
        input_term = f"({'+'.join(new_input_vars)})"

        return input_term, input_mapping

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
