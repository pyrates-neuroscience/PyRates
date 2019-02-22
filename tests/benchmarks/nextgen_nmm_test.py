from pyrates.frontend.template.circuit.circuit import CircuitTemplate
from pyrates.backend import ComputeGraph
from pyrates.utility import plot_timeseries, grid_search
from pyrates.utility.grid_search import grid_search_2, linearize_grid
import numpy as np
import matplotlib.pyplot as plt
# from pandas import DataFrame
import time
import pandas as pd

# disable TF-gpu warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# parameters
dt = 1e-4
T = 2.
inp = 2. + np.random.randn(int(T/dt), 1) * 1.0

params = {'J_e': np.arange(8., 12., 2.), 'J_i': np.arange(2., 12., 2.)}
param_map = {'J_e': {'var': [('Op_e.0', 'J_e'), ('Op_i.0', 'J_e')],
                     'nodes': ['PC.0', 'IIN.0']},
             'J_i': {'var': [('Op_e.0', 'J_i'), ('Op_i.0', 'J_i')],
                     'nodes': ['PC.0', 'IIN.0']}
             }

print("Start")
# perform simulation
t0 = time.time()
results, t, _ = grid_search_2(circuit_template="pyrates.examples.simple_nextgen_NMM.Net5",
                              param_grid=params, param_map=param_map,
                              inputs={("PC", "Op_e.0", "inp"): inp}, outputs={"r": ("PC", "Op_e.0", "r")},
                              dt=dt, simulation_time=T, permute_grid=False, profile="t")


print(type(results.columns.values[0][0]))
print(type(results.columns.values[0][1]))
print(type(results.columns.values[0][2]))


# df_col = pd.DataFrame(results.columns)
# print(df_col)


file = "/data/hu_salomon/Documents/testresult_new_2.csv"
results.to_csv(file, index=True)
results_1 = pd.read_csv(file, index_col=0, header=[0,1,2])
# results_1.columns = results.columns

# for col in results_1.columns
results_1.columns.values[0][0] = float(results_1.columns.values[0][0])
print(type(results_1.columns.values[0][0]))
print(type(results_1.columns.values[0][1]))
print(type(results_1.columns.values[0][2]))

# print(results_1.columns.values)

# store = pd.HDFStore('/data/hu_salomon/Documents/testresult_new_2.h5')
# store['df'] = results

# print(results.columns.values)
# # Create result file
# # results.reset_index().to_csv("/data/hu_salomon/Documents/testresult_new_2.csv", index=False)
# results.to_hdf("/data/hu_salomon/Documents/testresult_new_2.h5", key='df', mode='w')

# results_1 = pd.read_hdf("/data/hu_salomon/Documents/testresult_new_2.csv", index_col=0, header=[0,1,2])
#
# print(results_1.head())
# print(results_1.columns.values)
# params = pd.read_csv("/data/hu_salomon/Documents/testgrid_2.csv",
#                      index_col=0, header=[0])


# param_grid = linearize_grid(params)
# param_grid.to_csv("/data/hu_salomon/Documents/testgrid_2.csv")
# # print(results)
# # print(t)
# print(f'Peak memory usage: {m}')
# print('Total elapsed time: %.2f seconds' % (time.time()-t0))

# print(results.loc[:, (params['J_e'][0], params['J_i'][0])])
# print(results.columns)
# plotting
# for j_e in params['J_e']:
#     for j_i in params['J_i']:
#         print(j_e, j_i)
#         ax = plot_timeseries(results[j_e][j_i], title=f"J_e={j_e}, J_i={j_i}")
# plt.show()