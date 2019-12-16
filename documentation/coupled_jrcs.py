# pyrates imports
from pyrates.utility import grid_search, functional_connectivity
from pyrates.utility.grid_search import linearize_grid

# additional imports
import numpy as np
from numba import njit, config
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# threading configs
# config.THREADING_LAYER = 'tbb'
os.environ["KMP_BLOCKTIME"] = '0'
os.environ["KMP_SETTINGS"] = 'true'
os.environ["KMP_AFFINITY"] = 'granularity=fine,verbose,compact,1,0'
os.environ["OMP_NUM_THREADS"] = '2'
tf.config.threading.set_inter_op_parallelism_threads(3)
tf.config.threading.set_intra_op_parallelism_threads(2)
#tf.config.optimizer.set_jit(True)
# tf.config.experimental.set_synchronous_execution(False)
#tf.debugging.set_log_device_placement(True)

# parameters
############

# general parameters
dt = 1e-5                                                       # integration step-size of the forward euler solver in s
T = 1.0                                                         # overall simulation time in s
inp = np.random.uniform(120., 320., (int(T/dt)+1, 2))           # white noise input to the pyramidal cells in Hz.

N = 10                                                            # grid-size
C = np.linspace(0., 5.0, N)                                    # bi-directional connection strength
D = np.linspace(0., 5e-3, N)                                    # bi-directional coupling delay

# parameter grid
params = {'C': C,
          'D': D
          }
param_map = {'C': {'vars': ['weight'],
                   'edges': [('JRC1', 'JRC2'), ('JRC2', 'JRC1')]},
             'D': {'vars': ['delay'],
                   'edges': [('JRC1', 'JRC2'), ('JRC2', 'JRC1')]}
             }

# grid searches
###############

# numpy backend grid-search
results, param_map, _ = grid_search(circuit_template="model_templates.jansen_rit.simple_jansenrit.JRC_delaycoupled",
                                    param_grid=params, param_map=param_map,
                                    inputs={"all/JRC_op/u": np.asarray(inp, dtype=np.float32)},
                                    outputs={"v": "all/JRC_op/PSP_ein"},
                                    dt=dt, simulation_time=T, permute_grid=True, sampling_step_size=1e-3,
                                    init_kwargs={'backend': 'numpy', 'matrix_sparseness': 0.9, 'step_size': dt,
                                                 'solver': 'scipy', 'dde_approximation_order': 20},
                                    profile=True)

# tensorflow backend grid-search
# results, param_map, _ = grid_search(circuit_template="model_templates.jansen_rit.simple_jansenrit.JRC_delaycoupled",
#                                     param_grid=params, param_map=param_map,
#                                     inputs={"JRC1/PC/RPO_e_pc/u": np.asarray(inp1, dtype=np.float32),
#                                             "JRC2/PC/RPO_e_pc/u": np.asarray(inp2, dtype=np.float32)},
#                                     outputs={"v": "all/PC/OBS/V"},
#                                     step_size=step_size, simulation_time=T, permute_grid=True, sampling_step_size=1e-3,
#                                     init_kwargs={'solver': 'euler', 'backend': 'tensorflow'}, profile='t',
#                                     )

results.plot()

# coherence evaluation
######################

# coherences = np.zeros((len(C), len(D)))
# for circuit_name in param_map.index:
#
#     results_tmp = results[circuit_name]
#     fc = functional_connectivity(results_tmp, metric='coh', fmin=8.0, fmax=12.0, faverage=True, tmin=1.0, verbose=False)
#     row = np.argwhere(C == param_map.loc[circuit_name, 'C'])
#     col = np.argwhere(D == param_map.loc[circuit_name, 'D'])
#     coherences[row, col] = fc[1, 0]
#
# plt.matshow(coherences)
# plt.tight_layout()
plt.show()
