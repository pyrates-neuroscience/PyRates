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


def grid_search(circuit_template, param_grid, simulation_time, inputs, outputs, dt, sampling_step_size=None):
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

    # assign parameter updates to each circuit
    circuits = {}
    for n in range(param_grid.shape[0]):
        circuits[f'{circuit_template.label}_{n}'] = {'template': circuit_template, 'values': param_grid[n, :]}

    # combine circuits to backend
    circuit_comb = CircuitIR.from_circuits('combined', circuits=circuits)
    net = ComputeGraph(circuit_comb, dt=dt, vectorize='nodes', key='combined')

    # simulate the circuits behavior
    results = net.run(simulation_time=simulation_time,
                      inputs=inputs,
                      outputs=outputs,
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
