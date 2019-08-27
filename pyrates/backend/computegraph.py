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

"""This module provides the backend class that should be used to set up any backend in pyrates.
"""

# external imports
from typing import Optional, Tuple, List, Union, Any
from pandas import DataFrame, MultiIndex
import numpy as np
from networkx import find_cycle, NetworkXNoCycle, DiGraph
from copy import deepcopy
from warnings import filterwarnings

# pyrates imports
from pyrates.backend.parser import parse_equation_system, parse_dict
from pyrates import PyRatesException
#from pyrates.ir.circuit import CircuitIR
#from pyrates.frontend import CircuitTemplate
from .parser import replace

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


class ComputeGraph(object):
    """Creates a compute graph that contains all nodes in the network plus their recurrent connections.

    Parameters
    ----------
    net_config
        Intermediate representation of the network configuration. For a more detailed description, see the documentation
        of `pyrates.IR.CircuitIR`.
    dt
        Step-size with which the network should be simulated later on.
        Important for discretizing delays, differential equations, ...
    vectorization
        Defines the mode of automatic parallelization optimization that should be used. Can be `nodes` for lumping all
        nodes together in a vector, `full` for full vectorization of the network, or `None` for no vectorization.
    name
        Name of the network.
    build_in_place
        If False, a copy of the `net_config``will be made before compute graph creation. Should be used, if the
        `net_config` will be re-used for multiple compute graphs.
    backend
        Backend in which to build the compute graph.
    solver
        Numerical solver to use for differential equations.

    """

    def __new__(cls,
                net_config: Any,
                dt: float = 1e-3,
                vectorization: bool = True,
                name: Optional[str] = 'net0',
                build_in_place: bool = True,
                backend: str = 'numpy',
                solver: str = 'euler',
                float_precision: str = 'float32',
                **kwargs
                ) -> Any:
        """Instantiates operator.
        """

        return net_config.compile(dt=dt, vectorization=vectorization, backend=backend, solver=solver,
                                  float_precision=float_precision, **kwargs)

    def _net_config_consistency_check(self, net_config: Any) -> Any:
        """Checks whether the passed network configuration follows the expected intermediate representation structure.

        Parameters
        ----------
        net_config
            Intermediate representation of the network configuration that should be translated into the backend.

        Returns
        -------
        CircuitIR
            The checked (and maybe slightly altered) network configuration.

        """

        # check node attributes
        #######################

        # define which fields an operator should have
        op_fields = ['equations', 'inputs', 'output']

        # go through each node in the network  config
        for node_name, data in net_config.nodes(data=True):
            node = data["node"]

            # check whether an operation graph exists
            try:
                op_graph = node.op_graph
            except KeyError:
                raise KeyError(f'Key `node` not found on node {node_name}. Every node in the network configuration '
                               f'needs a field with the key `node` under which '
                               f'its operation graph and its template is stored.')
            except AttributeError:
                raise AttributeError(f'Attribute `op_graph` not found on node {node_name}. Every node in the network '
                                     f'configuration needs a graph object stored under its `node` key, which contains '
                                     f'all information regarding its operators, variables,'
                                     f'and input-output relationships.')

            # go through the operations on the node
            for op_name, op_info in op_graph.nodes.items():

                # check whether the variable field exists on the operator
                try:
                    variables = op_info['variables']
                except KeyError:
                    raise KeyError(f'Key `variables` not found in operator {op_name} of node {node_name}. Every '
                                   f'operator on a node needs a field `variables` under which all necessary '
                                   'variables are defined via key-value pairs.')

                # go through the variables
                for var_name, var in variables.items():

                    # check definition of each variable
                    var_def = {'state_var': {'value': np.zeros((1,), dtype=np.float32),
                                             'dtype': self._float_precision,
                                             'shape': (1,),
                                             },
                               'constant': {'value': KeyError,
                                            'shape': (1,),
                                            'dtype': self._float_precision},
                               'placeholder': {'dtype': self._float_precision,
                                               'shape': (1,),
                                               'value': None},
                               'raw': {'dtype': None,
                                       'shape': None,
                                       'value': KeyError}
                               }
                    try:
                        vtype = var['vtype']
                    except KeyError:
                        raise KeyError(f'Field `vtype` not found in variable definition of variable {var_name} of '
                                       f'operator {op_name} on node {node_name}. Each variable needs a field `vtype`'
                                       f'that indicates whether the variable is a `state_var`, `constant` or '
                                       f'`placeholder`.')
                    var_defs = var_def[vtype]

                    # check the value of the variable
                    try:
                        _ = var['value']
                    except KeyError:
                        if var_defs['value'] is KeyError:
                            raise KeyError(f'Field `value` not found in variable definition of variable {var_name} of '
                                           f'operator {op_name} on node {node_name}, but it is needed for variables of '
                                           f'type {vtype}.')
                        elif var_defs['value'] is not None:
                            var['value'] = var_defs['value']

                    # check the shape of the variable
                    try:
                        shape = var['shape']
                        if len(shape) == 0:
                            var['shape'] = (1,)
                        elif shape[0] != len(node):
                            raise ValueError(f"Mismatch between first dimension of variable {var_name} {shape} and"
                                             f"vector dimension as indicated by node {node_name} {len(node)}")
                    except KeyError:
                        var['shape'] = var_defs['shape']

                    # check the data type of the variable
                    try:
                        _ = var['dtype']
                    except KeyError:
                        var['dtype'] = var_defs['dtype']

                    val = node[f"{op_name}/{var_name}"]["value"]
                    if not hasattr(val, "shape"):
                        node[f"{op_name}/{var_name}"]["value"] = np.zeros(var['shape'], dtype=var['dtype']) + val

                # check whether the equations, inputs and output fields exist on the operator field
                if "equations" not in op_info:
                    raise KeyError(f'Field `equations` not found in operator {op_name} on '
                                   f'node {node_name}. Each operator should follow a list of equations that '
                                   f'needs to be provided at this position.')

                if "inputs" not in op_info:
                    op_info['inputs'] = {}

                if "output" not in op_info:
                    raise KeyError(f'Field `output` not found in operator {op_name} on '
                                   f'node {node_name}. Each operator should have an output, the name of which '
                                   f'needs to be provided at this position.')

        # check edge attributes
        #######################

        # go through edges
        for source, target, idx in net_config.edges:

            edge = net_config.edges[source, target, idx]

            # check whether source and target variable information were provided
            try:
                _ = edge['source_var']
            except KeyError:
                raise KeyError(f'Field `source_var` not found on edge {idx} between {source} and {target}. Every edge '
                               f'needs information about the variable on the source node it should access that needs '
                               f'to be provided at this position.')
            try:
                _ = edge['target_var']
            except KeyError:
                raise KeyError(f'Field `target_var` not found on edge {idx} between {source} and {target}. Every edge '
                               f'needs information about the variable on the target node it should access that needs '
                               f'to be provided at this position.')

            # check weight of edge
            try:
                weight = edge['weight']
                if not hasattr(weight, 'shape') or not weight.shape:
                    weight = np.ones((1,), dtype=self._float_precision) * weight
                elif 'float' not in str(weight.dtype):
                    weight = np.asarray(weight, dtype=self._float_precision)
                edge['weight'] = weight
            except KeyError:
                edge['weight'] = np.ones((1,), dtype=self._float_precision)

            # check delay of edge
            try:
                delay = edge['delay']
                if delay is not None:
                    if not hasattr(delay, 'shape') or not delay.shape:
                        delay = np.zeros((1,)) + delay
                    if 'float' in str(delay.dtype):
                        delay = np.asarray((delay / self.dt) + 1, dtype=np.int32)
                    edge['delay'] = delay
            except KeyError:
                edge['delay'] = None

        return net_config


def sort_equations(edge_eqs: list, node_eqs: list) -> list:
    """

    Parameters
    ----------
    edge_eqs
    node_eqs

    Returns
    -------

    """

    from .parser import is_diff_eq

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
