from pyrates.frontend.template.circuit.circuit import CircuitTemplate
from pyrates.backend import ComputeGraph
from pyrates.utility import plot_timeseries, grid_search
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

# parameters
dt = 1e-4
T = 2.
inp = np.random.randn(int(T/dt), 2) * 2.0

params = {'D': np.arange(0., 0.05, 0.01), 'C': np.arange(0.1, 0.5, 0.1)}
param_map = {'D': {'var': [(None, 'delay')],
                   'edges': [('PC1.0', 'PC2.0', 0), ('PC2.0', 'PC1.0', 0)]},
             'C': {'var': [(None, 'weight')],
                   'edges': [('PC1.0', 'PC2.0', 0), ('PC2.0', 'PC1.0', 0)]}
             }

# perform simulation
results = grid_search(circuit_template="pyrates.examples.simple_nextgen_NMM.Net6",
                      param_grid=params, param_map=param_map,
                      inputs={("PC", "Op_e.0", "inp"): inp}, outputs={"r": ("PC", "Op_e.0", "r")},
                      dt=dt, simulation_time=T, sampling_step_size=1e-3)

# plotting
for d in params['D']:
    for c in params['C']:
        ax = plot_timeseries(results[d][c], title=f"D={d}, C={c}")
        plt.show()
