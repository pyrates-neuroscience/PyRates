import time
import socket
import numpy as np
import pandas as pd

from pyrates.utility.grid_search import grid_search_2

print("Start")

t0 = time.time()

dt = 1e-4

param_map = {'J_e': {'var': [('Op_e.0', 'J_e'), ('Op_i.0', 'J_e')],
                     'nodes': ['PC.0', 'IIN.0']},
             'J_i': {'var': [('Op_e.0', 'J_i'), ('Op_i.0', 'J_i')],
                     'nodes': ['PC.0', 'IIN.0']}
             }

# T = [1., 2., 5., 10.]
T = [2.]

# grid_size = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
grid_size = [10000]

t_list = []
times = []

for n, T_ in enumerate(T):
    inp = 2. + np.random.randn(int(T_/dt), 1) * 1.0
    for i, size in enumerate(grid_size):
        print("")
        print(f'Simulation time: {T_} seconds')
        print(f'Timestep: {dt} seconds')
        print(f'Grid size: {size}')

        params = {'J_e': np.linspace(8., 12., size), 'J_i': np.linspace(2., 12., size)}

        _, t_dict, t_, m = grid_search_2(
            circuit_template="pyrates.examples.simple_nextgen_NMM.Net5",
            param_grid=params, param_map=param_map,
            inputs={("PC", "Op_e.0", "inp"): inp},
            outputs={"r": ("PC", "Op_e.0", "r")},
            dt=dt, simulation_time=T_, permute_grid=False,
            profile="tm", timestamps=True)

        t_list.append(t_dict)
        print(f'Total peak memory: {m} MByte')

    print(t_list)
    times.append(pd.DataFrame(t_list))
    t_list = []

times_total = pd.concat(times, axis=1)
times_total.index = grid_size

midx = []
for t in T:
    for i in range(len(t_dict)):
        midx.append(t)
index = pd.MultiIndex.from_arrays([midx, times_total.columns.values], names=['simulation time', 'time stamp', ])
times_total.columns = index

times_total.to_csv(f'/nobackup/spanien1/salomon/time_gridsize_dependancy_{socket.gethostname()}.csv',
                   index=True)

print(times_total)

print(f'Overall elapsed time: {time.time()-t0} seconds')