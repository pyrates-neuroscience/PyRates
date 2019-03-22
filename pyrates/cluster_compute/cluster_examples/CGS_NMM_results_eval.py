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

from pyrates.utility import plot_timeseries, grid_search, plot_psd, plot_connectivity
# External imports
import matplotlib.pyplot as plt
from seaborn import cubehelix_palette

# PyRates internal imports
from pyrates.utility.grid_search import grid_search_2
from pyrates.cluster_compute.cluster_compute import *

__author__ = "Christoph Salomon"
__status__ = "development"

#####################################
# Create ClusterGridSearch instance #
#####################################
nodes = [
        'animals',
        'spanien'
        # 'carpenters',
        # 'osttimor'
        ]

compute_dir = "/nobackup/spanien1/salomon/ClusterGridSearch/CGS_nextgen_NMM_Eval"

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
                  outputs=outputs, sampling_step_size=1e-3)

#########################
# Run ClusterGridSearch #
#########################
worker_env = "/data/u_salomon_software/anaconda3/envs/PyRates/bin/python3"
worker_file = "/data/hu_salomon/PycharmProjects/PyRates/pyrates/cluster_compute/cluster_workers/cluster_worker.py"

# Parameter grid
################
# params = {'J_e': np.arange(8., 12., 2.), 'J_i': np.arange(2., 12., 2.)}
params = {'J_e': np.linspace(8., 12., 10), 'J_i': np.linspace(2., 12., 10)}

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

######################
# Gather CGS Results #
######################

cgs_results = read_cgs_results(res_dir, key='Data')

##################
# Run GridSearch #
##################
print("Starting grid search!")
t0 = t.time()
gs_results, t_, _ = grid_search_2(circuit_template="pyrates.examples.simple_nextgen_NMM.Net5",
                                  param_grid=params, param_map=param_map,
                                  inputs={("PC", "Op_e.0", "inp"): inp}, outputs={"r": ("PC", "Op_e.0", "r")},
                                  dt=dt, simulation_time=T, permute_grid=True, profile="t",
                                  sampling_step_size=1e-3)

gs_results.to_hdf(f'{compute_dir}/Results/grid_search_results.hf5', key='Data')

print(f'Grid search elapsed times: {t.time()-t0:.3f} seconds')

########################################
# Evaluate Results #
########################################
err = cgs_results-gs_results
err = err.abs()
err_sum = err.values.sum()


################
# PLOT RESULTS #
################
num_yticks = 100

# Plot Grid Search and Cluster Grid Search
fig1, ax1 = plt.subplots(ncols=2, nrows=1, figsize=(25, 15), gridspec_kw={})

cm1 = cubehelix_palette(n_colors=int(gs_results.size), as_cmap=True, start=-2.0, rot=-0.1)
gs_results.index = np.round(gs_results.index, decimals=3)
cax1 = plot_connectivity(gs_results, ax=ax1[0], yticklabels=num_yticks,
                         xticklabels=False, cmap=cm1)
cax1.set_xlabel('Parameter grid index', size=20)
cax1.set_ylabel('Time in s', size=20)
cax1.set_title(f'Grid Search result', size=25)
for tick in cax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(15)
    tick.label.set_rotation('horizontal')

cm2 = cubehelix_palette(n_colors=int(cgs_results.size), as_cmap=True, start=2.5, rot=-0.1)
cgs_results.index = np.round(cgs_results.index, decimals=3)
cax2 = plot_connectivity(gs_results, ax=ax1[1], yticklabels=num_yticks,
                         xticklabels=False, cmap=cm2)
cax2.set_xlabel('Parameter grid index', size=20)
cax2.set_ylabel('Time in s', size=20)
cax2.set_title(f'Cluster Grid Search result', size=25)
for tick in cax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(15)
    tick.label.set_rotation('horizontal')

fig1.savefig("/data/hu_salomon/Pictures/Eval_plots/CGS_eval_results", format="svg")

# Plot Error
fig2, ax2 = plt.subplots(ncols=1, nrows=1, figsize=(20, 15), gridspec_kw={})
cm = cubehelix_palette(n_colors=int(err.size), as_cmap=True, start=-2.0, rot=-0.1)
err.index = np.round(err.index, decimals=3)
cax3 = plot_connectivity(err, ax=ax2, yticklabels=num_yticks,
                         xticklabels=False, cmap=cm2)
cax3.set_xlabel('Parameter grid index', size=20)
cax3.set_ylabel('Time in s', size=20)
cax3.set_title(f'Error', size=25)
for tick in cax3.yaxis.get_major_ticks():
    tick.label.set_fontsize(15)
    tick.label.set_rotation('horizontal')

fig2.savefig("/data/hu_salomon/Pictures/Eval_plots/CGS_eval_err", format="svg")

plt.show()
