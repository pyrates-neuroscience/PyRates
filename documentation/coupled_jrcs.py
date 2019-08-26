# pyrates imports
from pyrates.utility import plot_timeseries, grid_search, plot_psd, plot_connectivity, create_cmap

# additional imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

dt = 1e-4                                                       # integration step-size of the forward euler solver in s
T = 1.0                                                         # overall simulation time in s
inp1 = np.random.uniform(120., 320., (int(T/dt), 2))            # white noise input to the pyramidal cells in Hz.
inp2 = np.random.uniform(120., 320., (int(T/dt), 1))

N = 10                                                          # grid-size
C = np.linspace(0., 150., N)                                    # bi-directional connection strength
D = np.linspace(0., 1e-3, N)                                    # bi-directional coupling delay

params = {'C': C}
param_map = {'C': {'vars': ['weight'],
                   'edges': [('PC', 'EIN'), ('EIN', 'PC')]},
             'D': {'vars': ['delay'],
                   'edges': [('PC', 'EIN'), ('EIN', 'PC')]}}

results = grid_search(circuit_template="model_templates.jansen_rit.simple_jansenrit.JRC",
                      param_grid=params, param_map=param_map,
                      inputs={"PC/RPO_e_pc/u": inp1},
                      outputs={"v": "PC/OBS/V"},
                      dt=dt, simulation_time=T, permute_grid=True, sampling_step_size=1e-3)

results.plot()
plt.show()
