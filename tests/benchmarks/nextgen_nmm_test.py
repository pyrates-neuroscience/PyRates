from pyrates.frontend.template.circuit.circuit import CircuitTemplate
from pyrates.backend import ComputeGraph
from pyrates.utility import plot_timeseries, grid_search
from pyrates.utility.grid_search import cluster_grid_search
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

# parameters
dt = 1e-4
T = 2.
# inp needs to be of type 'list', not of type 'ndarray', since ndarrays can't be recreated from their string
# representation via std.literal_eval() due to EOL flags inside an ndarray.
# Recreation from string representation is necessary after parsing the data to a remote script during
# cluster_grid_search(). Converting inp to type 'list' has no effect on grid_search()

inp = (2. + np.random.randn(int(T/dt), 1) * 1.0).tolist()

# params = {'J_e': np.arange(8., 16., 2.), 'J_i': np.arange(2., 12., 2.)}

params = {'J_e': np.arange(8., 12., 2.), 'J_i': np.arange(2., 6., 2.)}

param_map = {'J_e': {'var': [('Op_e.0', 'J_e'), ('Op_i.0', 'J_e')],
                     'nodes': ['PC.0', 'IIN.0']},
             'J_i': {'var': [('Op_e.0', 'J_i'), ('Op_i.0', 'J_i')],
                     'nodes': ['PC.0', 'IIN.0']}
             }

hosts = ['animals', 'tschad']

fp = "/data/hu_salomon/Documents/ClusterGridSearch/config.json"

results = cluster_grid_search(circuit_template="pyrates.examples.simple_nextgen_NMM.Net5",
                      param_grid=params, param_map=param_map,
                      inputs={("PC", "Op_e.0", "inp"): inp}, outputs={"r": ("PC", "Op_e.0", "r")},
                      dt=dt, simulation_time=T, hostnames=hosts, permute_grid=True)

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
