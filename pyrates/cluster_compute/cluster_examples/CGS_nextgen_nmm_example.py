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

# External imports
import matplotlib.pyplot as plt

# PyRates internal imports
from pyrates.utility import plot_timeseries
from pyrates.utility.grid_search import grid_search_2
from pyrates.cluster_compute.cluster_compute import *

# np.set_printoptions(precision=2)

#####################################
# Create ClusterGridSearch instance #
#####################################
nodes = [
        'animals',
        'spanien',
        'carpenters',
        'osttimor'
        ]

compute_dir = "/nobackup/spanien1/salomon/ClusterGridSearch/CGS_nextgen_NMM_example_Spec"

# Create compute directory and connect to nodes
cgs = ClusterGridSearch(nodes, compute_dir=compute_dir)

############################
# Global config parameters #
############################
config_file = f'{compute_dir}/Config/simple_nextgen_NMM.json'

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

create_cgs_config(fp=config_file, circuit_template=circuit_template,
                  param_map=param_map, dt=dt, simulation_time=T, inputs=inputs,
                  outputs=outputs)

#########################
# Run ClusterGridSearch #
#########################
worker_env = "/data/u_salomon_software/anaconda3/envs/PyRates/bin/python3"
worker_file = "/data/hu_salomon/PycharmProjects/PyRates/pyrates/cluster_compute/cluster_workers/gridsearch_worker.py"

# Parameter grid
################
params = {'J_e': np.arange(1., 11., 1.), 'J_i': np.arange(1., 11., 1.)}
# params = {'J_e': np.arange(8., 12., 2.), 'J_i': np.arange(2., 12., 2.)}
# params = {'J_e': np.linspace(8., 12., 20), 'J_i': np.linspace(2., 12., 20)}

print("Starting cluster grid search!")
t0 = t.time()
res_dir, grid_file = cgs.run(thread_kwargs={
                                "worker_env": worker_env,
                                "worker_file": worker_file
                                 },
                             config_file=config_file,
                             chunk_size="dist_equal_add_mod",
                             # chunk_size=500,
                             param_grid=params,
                             permute=True)

print(f'Cluster grid search elapsed time: {t.time()-t0:.3f} seconds')

##################
# GATHER RESULTS #
##################

results = read_cgs_results(res_dir)

########################################
# EVALUATE AGAINST GRID_SEARCH RESULTS #
########################################
# print("Starting grid search!")
# t0 = t.time()
# results_gs, t_, _ = grid_search_2(circuit_template="pyrates.examples.simple_nextgen_NMM.Net5",
#                                   param_grid=params, param_map=param_map,
#                                   inputs={("PC", "Op_e.0", "inp"): inp}, outputs={"r": ("PC", "Op_e.0", "r")},
#                                   dt=dt, simulation_time=T, permute_grid=True, profile="t")
# print(f'Grid search elapsed times: {t.time()-t0:.3f} seconds')
#
# results_gs.to_hdf(f'{compute_dir}/Results/grid_search_results.hf5', key='Data')
#
# diff = results-results_gs
# diff_sum = diff.values.sum()
# print("")
# print(f'Total sum of errors: {diff_sum}')


################
# PLOT RESULTS #
################
# print(results.loc[:, (params['J_e'][0], params['J_i'][0])])
# print(results.columns)
# # plotting
# for j_e in params['J_e']:
#     for j_i in params['J_i']:
#         print(j_e, j_i)
#         ax = plot_timeseries(results[str(j_e)][str(j_i)], title=f"J_e={j_e}, J_i={j_i}")
# plt.show()
