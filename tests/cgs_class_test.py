import time
import numpy as np
from pyrates.utility.cluster_grid_search import ClusterGridSearch, create_cgs_config


print("Start!")
start = time.time()

# parameters
dt = 1e-4
T = 20.0

inp = (2. + np.random.randn(int(T/dt), 1) * 1.0).tolist()
params = {'J_e': np.arange(8., 28., 2.), 'J_i': np.arange(2., 22., 2.)}

# params = {'J_e': np.arange(8., 16., 2.), 'J_i': np.arange(2., 12., 2.)}
param_map = {'J_e': {'var': [('Op_e.0', 'J_e'), ('Op_i.0', 'J_e')],
                     'nodes': ['PC.0', 'IIN.0']},
             'J_i': {'var': [('Op_e.0', 'J_i'), ('Op_i.0', 'J_i')],
                     'nodes': ['PC.0', 'IIN.0']}
             }

config_name = "CGSConfig_1"

global_config = f'/data/hu_salomon/Documents/ClusterGridSearch/{config_name}.json'

create_cgs_config(fp=global_config, circuit_template="pyrates.examples.simple_nextgen_NMM.Net5",
                  param_map=param_map, dt=dt, simulation_time=T, inputs={("PC", "Op_e.0", "inp"): inp},
                  outputs={"r": ("PC", "Op_e.0", "r")}, permute_grid=True)

host_config = {
    'hostnames': [
        'animals',
        # 'spanien',
        'carpenters'
        # 'tschad'
        # 'rihanna',
        # 'unheilig'
        # 'styx',
        # 'spliff',
        # 'springsteen',
        # 'ufo',
        # 'roxette'
    ],
    # Python executable in conda environment with installed packages 'pandas' and 'pyrates'
    'host_env_cpu': "/data/u_salomon_software/anaconda3/envs/PyRates/bin/python",
    'host_env_gpu': "",
    # Python script to call on the remote hosts
    'host_file': "/data/hu_salomon/PycharmProjects/PyRates/pyrates/utility/cluster_worker.py",
    # Directory on the host to copy host_file and host_env to, if no shared filesystem is available
    'host_dir': ""
}

# param_grid = linearize_grid(params, permute=True)
# param_grid = "/data/hu_salomon/Documents/ClusterGridSearch/CGS_TestDir/Grids/CGSTestGrid.csv"

dir = f'/data/hu_salomon/Documents/ClusterGridSearch/{config_name}'

cgs = ClusterGridSearch(global_config, dir)
cgs.create_cluster(host_config)
res_dir = cgs.compute_grid(params)

elapsed = time.time() - start
print("Overall elapsed time: {0:.3f} seconds".format(elapsed))

# # plotting
# for j_e in params['J_e']:
#     for j_i in params['J_i']:
#         ax = plot_timeseries(results[j_e][j_i], title=f"J_e={j_e}, J_i={j_i}")
#         plt.show()