# -*- coding: utf-8 -*-
#
# PyRates software framework for flexible implementation of neural
# network models and simulations. See also:
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

"""PopulationTemplate and Connectivity — vector/matrix-native frontend classes.

These complement the existing scalar NodeTemplate/EdgeTemplate API with a higher-level
interface where a population of N units is treated as a single vectorized entity from
the moment of definition, and connections between populations are expressed as weight
matrices.  This avoids the O(N²) edge-creation overhead of the legacy
``add_edges_from_matrix`` path.
"""

import numpy as np
from typing import Optional, Dict, Union

from pyrates.frontend.template.node import NodeTemplate
from pyrates.frontend.template.edge import EdgeTemplate


class PopulationTemplate:
    """A population of N identical dynamical units sharing the same NodeTemplate.

    Parameters
    ----------
    name
        Label for this population, used as the node key in ``CircuitTemplate``.
    node
        ``NodeTemplate`` instance defining the single-unit dynamics.
    n
        Number of units in the population.
    params
        Optional heterogeneous parameter values, keyed by ``'op/var'`` strings.
        Each value may be:
        - a scalar → broadcast to all *n* units, or
        - an iterable of length *n* → assigned one value per unit.
    """

    def __init__(self, name: str, node: NodeTemplate, n: int,
                 params: Optional[Dict[str, Union[float, list, np.ndarray]]] = None):
        self.name = name
        self.node = node
        self.n = n
        self.params = params or {}

    def apply(self, label: str = None) -> tuple:
        """Instantiate a ``VectorizedNodeIR`` for this population of size *n*.

        Returns
        -------
        tuple
            ``(VectorizedNodeIR, label_map, var_ranges)`` — same contract as
            ``OperatorGraphTemplate.apply()``.
        """
        label = label if label is not None else self.name

        # Apply the base NodeTemplate once with no value overrides.
        # vectorize=False forces creation of a fresh VectorizedNodeIR (length=1)
        # rather than extending a cached one — we will resize it below.
        vec_node, label_map, _ = self.node.apply(values={}, label=label, vectorize=False)

        # Expand all vectorizable variables from length-1 lists to length-n lists.
        for op_key in vec_node.op_graph.operators:
            op_vars = vec_node.op_graph.nodes[op_key]['variables']
            for var_key, var_data in op_vars.items():
                if var_data['vtype'] not in ('state_var', 'constant', 'variable'):
                    # input / input_variable are managed externally — leave untouched
                    continue

                raw = var_data['value']
                base_val = raw[0] if isinstance(raw, list) and raw else raw

                param_key = f"{op_key}/{var_key}"
                if param_key in self.params:
                    pval = self.params[param_key]
                    if hasattr(pval, '__len__') and len(pval) == self.n:
                        new_val = list(pval)
                    else:
                        new_val = [pval] * self.n
                else:
                    new_val = [base_val] * self.n

                var_data['value'] = new_val
                var_data['shape'] = (len(new_val),)

        vec_node.length = self.n

        # Build var_ranges: each variable spans indices [0, n).
        var_ranges = {}
        for op_key in vec_node.op_graph.operators:
            for var_key, var_data in vec_node.op_graph.nodes[op_key]['variables'].items():
                if var_data.get('shape'):
                    var_ranges[(op_key, var_key)] = (0, var_data['shape'][0])

        return vec_node, label_map, var_ranges


class Connectivity:
    """Matrix-valued connection between two population variables.

    Encodes connectivity as a single ``(n_target, n_source)`` weight matrix,
    bypassing the legacy O(N²) edge-creation loop entirely.

    Parameters
    ----------
    source
        Source variable path in ``'population_name/op/var'`` notation.
    target
        Target variable path in ``'population_name/op/var'`` notation.
    weights
        Weight matrix of shape ``(n_target, n_source)``.  A scalar is accepted
        and will be stored as a 0-d array (must be combined with explicit
        source/target shapes at compile time — use an explicit matrix for clarity).
    edge
        Optional ``EdgeTemplate`` defining a non-dynamic coupling function.  The
        template's operators must contain **no state variables**.  Each equation is
        evaluated element-wise over the ``(n_target, n_source)`` pair space and
        the result is reduced via ``(W * coupling_matrix).sum(axis=1)``.
    edge_var_map
        Required when *edge* is given.  Maps each input variable name of the
        EdgeTemplate to either:

        * ``'source'`` — the pre-synaptic variable (from *source*), broadcast
          as ``source_var[None, :]``;
        * a ``'pop_name/op/var'`` path — a post-synaptic variable from the target
          population, broadcast as ``post_var[:, None]``.

        Example for Kuramoto coupling ``sin(theta_pre - theta_post)``::

            edge_var_map={'theta_pre': 'source', 'theta_post': 'e/phase_op/theta'}

    delays
        Optional scalar transmission delay (same time units as the simulation).
    spread
        Optional standard deviation of a gamma-kernel delay distribution centred on *delays*.
        When given together with *delays*, the delay is implemented via an ODE cascade whose
        order is ``n = round((delays/spread)**2)`` and rate ``a = n/delays``.  When ``None``
        (default), a discrete ring-buffer is used instead.
    """

    def __init__(self, source: str, target: str,
                 weights: Union[np.ndarray, float],
                 edge: Optional[EdgeTemplate] = None,
                 edge_var_map: Optional[Dict[str, str]] = None,
                 delays: Optional[float] = None,
                 spread: Optional[float] = None):
        self.source = source
        self.target = target
        self.weights = np.asarray(weights, dtype=float)
        self.edge = edge
        self.edge_var_map = edge_var_map or {}
        self.delays = delays
        self.spread = spread
