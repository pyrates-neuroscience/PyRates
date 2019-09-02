# pyrates imports
from pyrates.utility import grid_search

# additional imports
import numpy as np
from numba import njit, config
# import tensorflow as tf
import os
import matplotlib.pyplot as plt

# threading configs
config.THREADING_LAYER = 'tbb'
os.environ["KMP_BLOCKTIME"] = '0'
os.environ["KMP_SETTINGS"] = 'true'
os.environ["KMP_AFFINITY"] = 'granularity=fine,verbose,compact,1,0'
os.environ["OMP_NUM_THREADS"] = '2'
# tf.config.threading.set_inter_op_parallelism_threads(4)
# tf.config.threading.set_intra_op_parallelism_threads(2)
# tf.config.optimizer.set_jit(True)
# tf.config.experimental.set_synchronous_execution(False)
#tf.debugging.set_log_device_placement(True)

# parameters
############

# general parameters
dt = 1e-4                                                       # integration step-size of the forward euler solver in s
T = 1.0                                                         # overall simulation time in s
inp1 = np.random.uniform(120., 320., (int(T/dt), 1))            # white noise input to the pyramidal cells in Hz.
inp2 = np.random.uniform(120., 320., (int(T/dt), 1))

N = 10                                                         # grid-size
C = np.linspace(0., 200., N)                                    # bi-directional connection strength
D = np.linspace(0., 1e-2, N)                                    # bi-directional coupling delay

# parameter grid
params = {'C': C, 'D': D}
param_map = {'C': {'vars': ['weight'],
                   'edges': [('JRC1/PC', 'JRC2/PC'), ('JRC2/PC', 'JRC1/PC')]},
             'D': {'vars': ['delay'],
                   'edges': [('JRC1/PC', 'JRC2/PC'), ('JRC2/PC', 'JRC1/PC')]}}


# grid searches
###############

# numpy backend grid-search
results, param_map, _ = grid_search(circuit_template="model_templates.jansen_rit.simple_jansenrit.JRC_delaycoupled",
                                    param_grid=params, param_map=param_map,
                                    inputs={"JRC1/PC/RPO_e_pc/u": np.asarray(inp1, dtype=np.float32),
                                            "JRC2/PC/RPO_e_pc/u": np.asarray(inp2, dtype=np.float32)},
                                    outputs={"v": "all/PC/OBS/V"},
                                    dt=dt, simulation_time=T, permute_grid=True, sampling_step_size=1e-3,
                                    init_kwargs={'solver': 'euler', 'backend': 'numpy'}, profile='t',
                                    njit=True
                                    )

# tensorflow backend grid-search
# results, param_map, _ = grid_search(circuit_template="model_templates.jansen_rit.simple_jansenrit.JRC_delaycoupled",
#                                     param_grid=params, param_map=param_map,
#                                     inputs={"JRC1/PC/RPO_e_pc/u": np.asarray(inp1, dtype=np.float32),
#                                             "JRC2/PC/RPO_e_pc/u": np.asarray(inp2, dtype=np.float32)},
#                                     outputs={"v": "all/PC/OBS/V"},
#                                     dt=dt, simulation_time=T, permute_grid=True, sampling_step_size=1e-3,
#                                     init_kwargs={'solver': 'euler', 'backend': 'tensorflow'}, profile='t',
#                                     )

#results.plot()
#plt.show()
