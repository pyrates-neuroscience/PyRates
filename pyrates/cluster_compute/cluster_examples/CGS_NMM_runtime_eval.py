# -*- coding: utf-8 -*-
#
#
# PyRates software framework for flexible implementation of neural
# network models and simulations. See also:
# https://github.com/pyrates-neuroscience/PyRates
#
# Copyright (C) 2017-2018 the original authors (Richard Gast and
# Daniel Rose), the Max-Planck-Institute for Human Cognitive Brain
# Sciences ("MPI CBS") and contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#
# CITATION:
#
# Richard Gast and Daniel Rose et. al. in preparation

# PyRates internal imports
from pyrates.utility.grid_search import grid_search_2
from pyrates.cluster_compute.cluster_compute import *

# meta infos
__author__ = "Christoph Salomon"
__status__ = "development"

# disable TF-gpu warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

grid_size = [1,    2,    3,    4,    5,    6,    7,    8,    9,
             10,   20,   30,   40,   50,   60,   70,   80,   90,
             100,  200,  300,  400,  500,  600,  700,  800,  900,
             1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
             10000]

num_nodes = [5]

node_lst = [
        'animals',
        'spanien',
        'tschad',
        'carpenters',
        'osttimor',
        'lech',
        'tiber',
        'uganda',
        'main'
        ]

circuit_template = "pyrates.examples.simple_nextgen_NMM.Net5"

dt = 1e-4
T = 2.0
inp = 2. + np.random.randn(int(T/dt), 1) * 1.0

param_map = {'J_e': {'var': [('Op_e.0', 'J_e'), ('Op_i.0', 'J_e')],
                     'nodes': ['PC.0', 'IIN.0']},
             'J_i': {'var': [('Op_e.0', 'J_i'), ('Op_i.0', 'J_i')],
                     'nodes': ['PC.0', 'IIN.0']}
             }

inputs = {("PC", "Op_e.0", "inp"): inp.tolist()}
outputs = {"r": ("PC", "Op_e.0", "r")}

worker_env = "/data/u_salomon_software/anaconda3/envs/PyRates/bin/python3"
worker_file = "/data/hu_salomon/PycharmProjects/PyRates/pyrates/cluster_compute/cluster_workers/cluster_worker.py"


runtime_total = {}
runtime_cgs = {}
runtime_files = {}
time_res_dir = f'/data/hu_salomon/Documents/CGS_eval/animals_chunk_400'
global_compute_dir = "/nobackup/spanien1/salomon/ClusterGridSearch/CGS_nextgen_NMM_runtime_eval/animals_chunk_400"

for n in num_nodes:
    nodes = node_lst[:n]

    # Create compute directory and connect to nodes
    compute_dir = f'{global_compute_dir}/nodes_{n}'
    cgs = ClusterGridSearch(nodes, compute_dir=compute_dir)
    config_file = f'{compute_dir}/Config/simple_nextgen_NMM.json'
    create_cgs_config(fp=config_file, circuit_template=circuit_template,
                      param_map=param_map, dt=dt, simulation_time=T, inputs=inputs,
                      outputs=outputs, sampling_step_size=1e-3)

    time_total = []
    time_cgs = []
    time_files = []
    for i, size in enumerate(grid_size):

        params = {'J_e': np.linspace(8., 12., size), 'J_i': np.linspace(2., 12., size)}

        print("")
        print(f'Simulation time: {2.0} seconds')
        print(f'Timestep: {dt} seconds')
        print(f'Grid size: {size}')
        print(f'Nodes: {nodes}')

        t0_total = t.time()
        t0_cgs = t.time()
        res_file, grid_file = cgs.run(thread_kwargs={
                                        "worker_env": worker_env,
                                        "worker_file": worker_file
                                        },
                                      config_file=config_file,
                                      chunk_size=400,
                                      param_grid=params,
                                      permute=False)
        t_cgs = t.time() - t0_cgs

        # Read result files
        ###################
        t0_files = t.time()
        results = read_cgs_results(res_file, key='Data')
        t_files = t.time() - t0_files

        t_total = t.time() - t0_total

        print(f'Total elapsed time: {t_total} seconds')

        time_total.append(t_total)
        time_cgs.append(t_cgs)
        time_files.append(t_files)

    runtime_total[f'{n}'] = time_total
    runtime_cgs[f'{n}'] = time_cgs
    runtime_files[f'{n}'] = time_files

    ##############################
    # Create Dict and save files #
    ##############################
    rows = list(np.arange(n).astype(str))

    dir_ = f'{time_res_dir}/nodes_{n}'
    os.makedirs(dir_, exist_ok=True)

    df_runtime_total = pd.DataFrame.from_dict(runtime_total)
    df_runtime_total.to_csv(f'{dir_}/runtime_total.csv', index=True)

    df_runtime_cgs = pd.DataFrame.from_dict(runtime_cgs)
    df_runtime_cgs.to_csv(f'{dir_}/runtime_cgs.csv', index=True)

    df_runtime_files = pd.DataFrame.from_dict(runtime_files)
    df_runtime_files.to_csv(f'{dir_}/runtime_files.csv', index=True)

#######################################################################################################################

# runtime_total = {}
# runtime_cgs = {}
# runtime_files = {}
# time_res_dir = f'/data/hu_salomon/Documents/CGS_eval/animals_chunk_500'
# global_compute_dir = "/nobackup/spanien1/salomon/ClusterGridSearch/CGS_nextgen_NMM_runtime_eval/animals_chunk_500"
#
# for n in num_nodes:
#     nodes = node_lst[:n]
#
#     # Create compute directory and connect to nodes
#     compute_dir = f'{global_compute_dir}/nodes_{n}'
#     cgs = ClusterGridSearch(nodes, compute_dir=compute_dir)
#     config_file = f'{compute_dir}/Config/simple_nextgen_NMM.json'
#     create_cgs_config(fp=config_file, circuit_template=circuit_template,
#                       param_map=param_map, dt=dt, simulation_time=T, inputs=inputs,
#                       outputs=outputs, sampling_step_size=1e-3)
#
#     time_total = []
#     time_cgs = []
#     time_files = []
#     for i, size in enumerate(grid_size):
#
#         params = {'J_e': np.linspace(8., 12., size), 'J_i': np.linspace(2., 12., size)}
#
#         print("")
#         print(f'Simulation time: {2.0} seconds')
#         print(f'Timestep: {dt} seconds')
#         print(f'Grid size: {size}')
#         print(f'Nodes: {nodes}')
#
#         t0_total = t.time()
#         t0_cgs = t.time()
#         res_file, grid_file = cgs.run(thread_kwargs={
#                                         "worker_env": worker_env,
#                                         "worker_file": worker_file
#                                         },
#                                       config_file=config_file,
#                                       chunk_size='dist_equal_add_mod',
#                                       param_grid=params,
#                                       permute=False)
#         t_cgs = t.time() - t0_cgs
#
#         # Read result files
#         ###################
#         t0_files = t.time()
#         results = read_cgs_results(res_file, key='Data')
#         t_files = t.time() - t0_files
#
#         t_total = t.time() - t0_total
#
#         print(f'Total elapsed time: {t_total} seconds')
#
#         time_total.append(t_total)
#         time_cgs.append(t_cgs)
#         time_files.append(t_files)
#
#     runtime_total[f'{n}'] = time_total
#     runtime_cgs[f'{n}'] = time_cgs
#     runtime_files[f'{n}'] = time_files
#
#     ##############################
#     # Create Dict and save files #
#     ##############################
#     rows = list(np.arange(n).astype(str))
#
#     dir_ = f'{time_res_dir}/nodes_{n}'
#     os.makedirs(dir_, exist_ok=True)
#
#     df_runtime_total = pd.DataFrame.from_dict(runtime_total)
#     df_runtime_total.to_csv(f'{dir_}/runtime_total.csv', index=True)
#
#     df_runtime_cgs = pd.DataFrame.from_dict(runtime_cgs)
#     df_runtime_cgs.to_csv(f'{dir_}/runtime_cgs.csv', index=True)
#
#     df_runtime_files = pd.DataFrame.from_dict(runtime_files)
#     df_runtime_files.to_csv(f'{dir_}/runtime_files.csv', index=True)
