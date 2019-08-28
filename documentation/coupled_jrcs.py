# pyrates imports
from pyrates.utility import plot_timeseries, grid_search, plot_psd, plot_connectivity, create_cmap

# additional imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import njit, config, threading_layer
config.THREADING_LAYER = 'default'

dt = 1e-4                                                       # integration step-size of the forward euler solver in s
T = 1.0                                                         # overall simulation time in s
inp1 = np.random.uniform(120., 320., (int(T/dt), 1))            # white noise input to the pyramidal cells in Hz.
inp2 = np.random.uniform(120., 320., (int(T/dt), 1))

N = 5                                                          # grid-size
C = np.linspace(0., 200., N)                                    # bi-directional connection strength
D = np.linspace(0., 1e-2, N)                                    # bi-directional coupling delay

params = {'C': C, 'D': D}
param_map = {'C': {'vars': ['weight'],
                   'edges': [('JRC1/PC', 'JRC2/PC'), ('JRC2/PC', 'JRC1/PC')]},
             'D': {'vars': ['delay'],
                   'edges': [('JRC1/PC', 'JRC2/PC'), ('JRC2/PC', 'JRC1/PC')]}}

results, param_map, _, _ = grid_search(circuit_template="model_templates.jansen_rit.simple_jansenrit.JRC_delaycoupled",
                                       param_grid=params, param_map=param_map,
                                       inputs={"JRC1/PC/RPO_e_pc/u": np.asarray(inp1, dtype=np.float32),
                                               "JRC2/PC/RPO_e_pc/u": np.asarray(inp2, dtype=np.float32)},
                                       outputs={"v": "all/PC/OBS/V"},
                                       dt=dt, simulation_time=T, permute_grid=True, sampling_step_size=1e-3,
                                       init_kwargs={'solver': 'euler', 'backend': 'numpy'}, profile='t')

#print('Numba threading layer: ' + threading_layer())

results.plot()
plt.show()
