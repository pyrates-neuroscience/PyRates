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
import matplotlib.pyplot as plt
from seaborn import cubehelix_palette
from seaborn import set
from pyrates.utility import plot_connectivity, grid_search
from pyrates.cluster_compute.cluster_compute import *

# meta infos
__author__ = "Christoph Salomon"
__status__ = "development"


def plot_peaks_per_second(data, parameters, tick_size=5):
    num_peaks = np.zeros([len(parameters['k_e']), len(parameters['k_i'])])
    for m, k_e in enumerate(parameters['k_e']):
        for n, k_i in enumerate(parameters['k_i']):
            # data already contains the number of peaks. No further postprocessing is needed
            num_peaks[m, n] = np.array(data[k_e][k_i])

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 15), gridspec_kw={})
    plot_connectivity(num_peaks, ax=ax, xticklabels=list(parameters['k_i']), yticklabels=list(parameters['k_e']))

    # Show only every 25th tick (steps of 5.0 on the axis label)
    step_size_x = parameters['k_e'][1] - parameters['k_e'][0]
    step_size_y = parameters['k_i'][1] - parameters['k_i'][0]
    step_tick_x = tick_size/step_size_x
    step_tick_y = tick_size/step_size_y
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % 2 != 0:
            label.set_visible(False)
    for n, label in enumerate(ax.yaxis.get_ticklabels()):
        label.font_size = 20
        if n % 2 != 0:
            label.set_visible(False)

    plt.tick_params(labelsize=20)
    axis_font = {'fontname': 'Arial', 'size': '25'}
    ax.set_xlabel('k_i', fontdict=axis_font)
    ax.set_ylabel('k_e', fontdict=axis_font)
    ax.set_title('Average spikes per second', fontdict=axis_font)

    fig.savefig("/nobackup/spanien1/salomon/ClusterGridSearch/Montbrio_EIC/Computation_1/Results/DefaultGrid_0/Montbrio_EIC_spike_rate_2", format="svg")
    plt.show()


# Create CGS instance
#####################
# nodes = [
#         'animals',
#         'spanien',
#         # 'tschad',
#         'carpenters',
#         'osttimor'
#         # 'lech',
#         # 'tiber',
#         # 'uganda',
#         # 'main'
#         ]
#
# compute_dir = "/nobackup/spanien1/salomon/ClusterGridSearch/Montbrio_EIC/Computation_1"
# cgs = ClusterGridSearch(nodes, compute_dir=compute_dir)
#
# # Create configuration file
# ###########################
# param_map = {'k_e': {'var': [('Op_e.0', 'k_ee'), ('Op_i.0', 'k_ie')],
#                      'nodes': ['E.0', 'I.0']},
#              'k_i': {'var': [('Op_e.0', 'k_ei'), ('Op_i.0', 'k_ii')],
#                      'nodes': ['E.0', 'I.0']}
#              }
#
# config_file = f'{compute_dir}/Config/Montbrio_EIC_config.json'
# create_cgs_config(fp=config_file,
#                   circuit_template="/data/hu_salomon/PycharmProjects/PyRates/models/Montbrio/Montbrio.EI_Circuit",
#                   param_map=param_map,
#                   dt=1e-5,
#                   simulation_time=10.,
#                   inputs={},
#                   outputs={"r": ("E", "Op_e.0", "r")},
#                   sampling_step_size=1e-3)
#
# # Create parameter grid
# #######################
# params = {'k_e': np.linspace(10., 30., 3), 'k_i': np.linspace(10., 30., 3)}
# param_grid = linearize_grid(params, permute=True)
#
# # Run cluster grid search
# #########################
# res_file, grid_file = cgs.run(config_file=config_file,
#                               param_grid=param_grid,
#                               chunk_size=100,
#                               thread_kwargs={
#                                   "worker_env": "/data/u_salomon_software/anaconda3/envs/PyRates/bin/python3",
#                                   "worker_file": "/data/hu_salomon/PycharmProjects/PyRates/"
#                                                  "pyrates/cluster_compute/cluster_workers/gridsearch_worker.py"
#                               })

# Worker saved the number of peaks for each parameter combination in the result file under the key 'Num_Peaks'
params = {'k_e': np.linspace(10., 30., 101), 'k_i': np.linspace(10., 30., 101)}
res_file = "/nobackup/spanien1/salomon/ClusterGridSearch/CGS_EIC_spike_rate/Results/DefaultGrid_0/CGS_result_DefaultGrid_0.h5"
peaks = read_cgs_results(res_file, key='Num_Peaks')

# Create plots
##############

plot_peaks_per_second(peaks, parameters=params, tick_size=5)
