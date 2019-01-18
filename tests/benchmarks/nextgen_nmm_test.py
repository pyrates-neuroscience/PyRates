# from pyrates.frontend.template.circuit.circuit import CircuitTemplate
# from pyrates.backend import ComputeGraph
# from pyrates.utility import plot_timeseries, grid_search
from pyrates.utility.grid_search import cluster_grid_search
# from pyrates.utility.grid_search import create_cgs_config_file
import numpy as np
# import matplotlib.pyplot as plt
# from pandas import DataFrame
# import os
import time

start = time.time()

# parameters
# dt = 1e-4
# T = 2.

# inp needs to be of type 'list', instead of type 'ndarray', to store it in JSON file
# has no effect on original grid_search()
# inp = (2. + np.random.randn(int(T/dt), 1) * 1.0).tolist()

params = {'J_e': np.arange(8., 14., 2.), 'J_i': np.arange(2., 10., 2.)}

# params = {'J_e': np.arange(8., 12., 2.), 'J_i': np.arange(2., 6., 2.)}
# param_map = {'J_e': {'var': [('Op_e.0', 'J_e'), ('Op_i.0', 'J_e')],
#                      'nodes': ['PC.0', 'IIN.0']},
#              'J_i': {'var': [('Op_e.0', 'J_i'), ('Op_i.0', 'J_i')],
#                      'nodes': ['PC.0', 'IIN.0']}
#              }

hosts = {
    'host_names': [
        'animals',
        'spanien',
        'tschad'
        # 'osttimor',
        # 'tiber',
    ],
    # Python executable in conda environment with pandas and pyrates installed
    'host_env': "/data/u_salomon_software/anaconda3/envs/PyRates/bin/python",
    # Python script to call on the remote hosts
    'host_file': "/data/hu_salomon/PycharmProjects/PyRates/pyrates/utility/cluster_worker.py"
}

config = "/data/hu_salomon/Documents/ClusterGridSearch/CGS_config_2019-01-17_default.json"

# string = f'{os.getcwd()}/CGS_config_{str(date.today())}_default.json'
# create_cgs_config_file(string, circuit_template="pyrates.examples.simple_nextgen_NMM.Net5", param_grid=params,
#                        param_map=param_map, dt=dt, simulation_time=T, inputs={("PC", "Op_e.0", "inp"): inp},
#                        outputs={"r": ("PC", "Op_e.0", "r")}, permute_grid=True)

print("Starting cluster grid search")

results = cluster_grid_search(hosts, config_file=config, param_grid=params)

end = time.time()

print('Time elapsed: %.2f seconds' % (end-start))


# # perform simulation
# results = grid_search(circuit_template="pyrates.examples.simple_nextgen_NMM.Net5",
#                       param_grid=params, param_map=param_map,
#                       inputs={("PC", "Op_e.0", "inp"): inp}, outputs={"r": ("PC", "Op_e.0", "r")},
#                       dt=dt, simulation_time=T, permute_grid=True)

# print(results)

# with open('/data/hu_salomon/Documents/GridSearch_Results/test.csv', 'w') as f:
#     results.to_csv(f)
    # for row in results:
    #     f.write("%s\n" % str(row))

#
# plotting
# for j_e in params['J_e']:
#     for j_i in params['J_i']:
#         ax = plot_timeseries(results[j_e][j_i], title=f"J_e={j_e}, J_i={j_i}")
#         plt.show()
