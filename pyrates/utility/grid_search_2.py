# -*- coding: utf-8 -*-
#
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
#
# CITATION:
#
# Richard Gast and Daniel Rose et. al. in preparation
"""Functions for performing parameter grid simulations with pyrates models.
"""

# external imports
import pandas as pd
import numpy as np

# pyrates internal imports
from pyrates.backend import ComputeGraph
from pyrates.frontend import CircuitTemplate
from pyrates.ir.circuit import CircuitIR

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


# TODO: Add kwargs check to use the net.run() internal timestamp functionality

def grid_search(circuit_template, param_grid, param_map, dt, simulation_time, inputs, outputs,
                sampling_step_size=None, permute_grid=False, **kwargs):
    """
    Parameters
    ----------
    circuit_template
    param_grid
    param_map
    dt
    simulation_time
    inputs
    outputs
    sampling_step_size
    permute_grid
    kwargs
    Returns
    -------
    """

    # linearize parameter grid if necessary
    if type(param_grid) is dict:
        param_grid = linearize_grid(param_grid, permute_grid)

    # assign parameter updates to each circuit and combine them to unconnected network
    circuit = CircuitIR()
    circuit_names = []
    param_info = []
    param_split = "__"
    val_split = "--"
    comb = "_"
    for n in range(param_grid.shape[0]):
        circuit_tmp = CircuitTemplate.from_yaml(circuit_template).apply()
        circuit_names.append(f'{circuit_tmp.label}_{n}')
        circuit_tmp = adapt_circuit(circuit_tmp, param_grid.iloc[n, :], param_map)
        circuit.add_circuit(circuit_names[-1], circuit_tmp)
        param_names = list(param_grid.columns.values)
        param_info_tmp = [f"{param_names[i]}{val_split}{val}" for i, val in enumerate(param_grid.iloc[n, :])]
        param_info.append(param_split.join(param_info_tmp))

    # create backend graph
    net = ComputeGraph(circuit, dt=dt, **kwargs)

    # adjust input of simulation to combined network
    for inp_key, inp in inputs.items():
        inputs[inp_key] = np.tile(inp, (1, len(circuit_names)))

    # adjust output of simulation to combined network
    nodes = list(CircuitTemplate.from_yaml(circuit_template).apply().nodes)
    out_names = list(outputs.keys())
    for out_key, out in outputs.copy().items():
        outputs.pop(out_key)
        if out[0] in nodes:
            for i, name in enumerate(param_info):
                out_tmp = list(out)
                out_tmp[0] = f'{circuit_names[i]}/{out_tmp[0]}'
                outputs[f'{name}{param_split}out_var{val_split}{out_key}'] = tuple(out_tmp)
        elif out[0] == 'all':
            out_names = []
            for node in nodes:
                for i, name in enumerate(param_info):
                    out_tmp = list(out)
                    out_tmp[0] = f'{circuit_names[i]}/{node}'
                    outputs[f'{name}{param_split}out_var{val_split}{out_key}{comb}{node}'] = tuple(out_tmp)
                    out_names.append(f'{out_key}{comb}{node}')
            out_names = list(set(out_names))
        else:
            node_found = False
            out_names = []
            for node in nodes:
                if out[0] in node:
                    node_found = True
                    for i, name in enumerate(param_info):
                        out_tmp = list(out)
                        out_tmp[0] = f'{circuit_names[i]}/{node}'
                        outputs[f'{name}{param_split}out_var{val_split}{out_key}{comb}{node}'] = tuple(out_tmp)
                        out_names.append(f'{out_key}{comb}{node}')
            out_names = list(set(out_names))
            if not node_found:
                raise ValueError(f'Invalid output identifier in output: {out_key}. '
                                 f'Node {out[0]} is not part of this network')

    # simulate the circuits behavior
    results = net.run(simulation_time=simulation_time,
                      inputs=inputs,
                      outputs=outputs,
                      sampling_step_size=sampling_step_size)

    # transform results into long-form dataframe with changed parameters as columns
    multi_idx = [param_grid[key].values for key in param_grid.keys()]
    n_iters = len(multi_idx[0])
    outs = []
    for out_name in out_names:
        outs += [out_name] * n_iters
    multi_idx = [list(idx) * len(out_names) for idx in multi_idx]
    multi_idx.append(outs)
    index = pd.MultiIndex.from_arrays(multi_idx, names=list(param_grid.keys()) + ['out_var'])
    index = pd.MultiIndex.from_tuples(list(set(index)), names=list(param_grid.keys()) + ['out_var'])
    results_final = pd.DataFrame(columns=index, data=np.zeros_like(results.values), index=results.index)
    for col in results.keys():
        params = col[0].split(param_split)
        indices = [None] * len(results_final.columns.names)
        for param in params:
            var, val = param.split(val_split)[:2]
            idx = list(results_final.columns.names).index(var)
            try:
                indices[idx] = float(val)
            except ValueError:
                indices[idx] = val
        results_final.loc[:, tuple(indices)] = results[col].values

    return results_final


def linearize_grid(grid: dict, permute=False):
    """
    Parameters
    ----------
    grid
    permute
    Returns
    -------
    """

    arg_lengths = [len(arg) for arg in grid.values()]

    if len(list(set(arg_lengths))) == 1 and not permute:
        return pd.DataFrame(grid)
    else:
        vals, keys = [], []
        for key, val in grid.items():
            vals.append(val)
            keys.append(key)
        new_grid = np.stack(np.meshgrid(*tuple(vals)), -1).reshape(-1, len(grid))
        return pd.DataFrame(new_grid, columns=keys)


def adapt_circuit(circuit, params, param_map):
    """
    Parameters
    ----------
    circuit
    params
    param_map
    Returns
    -------
    """

    for key in params.keys():
        val = params[key]
        for op, var in param_map[key]['var']:
            nodes = param_map[key]['nodes'] if 'nodes' in param_map[key] else []
            edges = param_map[key]['edges'] if 'edges' in param_map[key] else []
            for node in nodes:
                if op in circuit.nodes[node]['node'].op_graph.nodes:
                    circuit.nodes[node]['node'].op_graph.nodes[op]['variables'][var]['value'] = float(val)
            for source, target, edge in edges:
                if op in circuit.edges[source, target, edge]:
                    circuit.edges[source, target, edge][op][var] = float(val)
                elif var in circuit.edges[source, target, edge]:
                    circuit.edges[source, target, edge][var] = float(val)

    return circuit
