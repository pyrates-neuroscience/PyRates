
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


def grid_search(circuit_template, param_grid, simulation_time, inputs, outputs, dt, sampling_step_size=None, **kwargs):
    """

    Parameters
    ----------
    circuit
    param_grid
    simulation_time
    inputs
    outputs
    sampling_step_size

    Returns
    -------

    """

    # linearize parameter grid if necessary
    if type(param_grid) is dict:
        param_grid = linearize_grid(param_grid)

    # assign parameter updates to each circuit and combine them to unconnected network
    circuit = CircuitIR()
    circuit_names = []
    for n in range(param_grid.shape[0]):
        circuit_tmp = CircuitTemplate.from_yaml(circuit_template).apply()
        circuit_names.append(f'{circuit_tmp.label}_{n}')
        circuit_tmp = adapt_circuit(circuit_tmp, param_grid.iloc[n, :])
        circuit.add_circuit(circuit_names[-1], circuit_tmp)

    # create backend graph
    net = ComputeGraph(circuit, dt=dt, **kwargs)

    # adjust input and output of simulation
    for inp_key, inp in inputs.items():
        inputs[inp_key] = np.tile(inp, (1, len(circuit_names)))

    outputs_new = {}
    for name in circuit_names:
        for out_key, out in outputs.items():
            outputs_new[f'{name}/{out_key}'] = out

    # simulate the circuits behavior
    results, _ = net.run(simulation_time=simulation_time,
                         inputs=inputs,
                         outputs=outputs_new,
                         sampling_step_size=sampling_step_size)

    return results


def linearize_grid(grid: dict):
    """

    Parameters
    ----------
    grid

    Returns
    -------

    """

    arg_lengths = [len(arg) for arg in grid.values()]

    if len(list(set(arg_lengths))) == 1:
        return pd.DataFrame(grid)
    else:
        new_grid = np.meshgrid(tuple([arg for arg in grid.values()]))
        return pd.DataFrame(new_grid, columns=grid.keys())


def adapt_circuit(circuit, params):
    """

    Parameters
    ----------
    circuit
    params

    Returns
    -------

    """

    for keys in params.keys():
        val = params[keys]
        node, op, var = keys.split('/')
        circuit.nodes[node]['node'].op_graph.nodes[op]['variables'][var]['value'] = float(val)

    return circuit
