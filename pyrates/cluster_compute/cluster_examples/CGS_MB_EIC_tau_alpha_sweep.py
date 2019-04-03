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

# external imports
import scipy.signal as sp
import matplotlib.pyplot as plt
from seaborn import cubehelix_palette

# PyRates internal imports
from pyrates.cluster_compute.cluster_compute import *
from pyrates.utility import plot_timeseries, plot_psd, plot_connectivity

t0 = t.time()
#####################################
# Create ClusterGridSearch instance #
#####################################
nodes = [
    'animals',
    'spanien',
    'tschad',
    # 'carpenters',
    'osttimor'
]

compute_dir = "/nobackup/spanien1/salomon/ClusterGridSearch/Montbrio/EIC/C20_tau_alpha_sweep_high_res"

cgs = ClusterGridSearch(nodes, compute_dir=compute_dir)

############################
# Create configuration file
###########################
param_map = {'k_ee': {'var': [('Op_e.0', 'k_ee')],
                      'nodes': ['E.0']},
             'k_ei': {'var': [('Op_e.0', 'k_ei')],
                      'nodes': ['E.0']},
             'k_ii': {'var': [('Op_i.0', 'k_ii')],
                      'nodes': ['I.0']},
             'k_ie': {'var': [('Op_i.0', 'k_ie')],
                      'nodes': ['I.0']},
             'alpha': {'var': ['Op_e.0', 'alpha'],
                       'nodes': ['E.0']},
             'tau': {'var': ['Op_e.0', 'tau'],
                     'nodes': ['E.0']}
             }

dts = 1e-2

config_file = f'{compute_dir}/Config/Montbrio_EIC_coupling_config.json'

create_cgs_config(fp=config_file,
                  circuit_template="/data/hu_salomon/PycharmProjects/PyRates/models/Montbrio/Montbrio.EI_Circuit",
                  param_map=param_map,
                  simulation_time=200.,
                  dt=5e-4,
                  inputs={},
                  outputs={"r": ("E", "Op_e.0", "r")},
                  sampling_step_size=dts)

# Create parameter grid
#######################
tau_temp = np.linspace(1.0, 3.0, 101)
alpha_temp = np.linspace(0.0, 0.2, 101)
tau, alpha = np.meshgrid(tau_temp, alpha_temp)
tau = tau.reshape(tau_temp.size*alpha_temp.size, 1).squeeze()
alpha = alpha.reshape(tau_temp.size*alpha_temp.size, 1).squeeze()
Cs = 20.0

ei_ratio = [1.0, 2.4, 3.6]
io_ratio = [1.2, 1.7, 2.2]


for idx, (ei, io) in enumerate(zip(ei_ratio, io_ratio)):
    k_ee = [Cs]*(tau_temp.size*alpha_temp.size)
    k_ei = [Cs / (ei * io)]*(tau_temp.size*alpha_temp.size)
    k_ie = [Cs / io]*(tau_temp.size*alpha_temp.size)
    k_ii = [Cs / ei]*(tau_temp.size*alpha_temp.size)

    params = {'k_ee': k_ee, 'k_ei': k_ei, 'k_ie': k_ie, 'k_ii': k_ii, 'alpha': alpha, 'tau': tau}

    # Run cluster grid search
    #########################
    res_file, _ = cgs.run(config_file=config_file,
                          param_grid=params,
                          permute=False,
                          chunk_size=1000,
                          thread_kwargs={
                              "worker_env": "/data/u_salomon_software/anaconda3/envs/PyRates/bin/python3",
                              "worker_file": "/data/hu_salomon/PycharmProjects/PyRates/"
                                             "pyrates/cluster_compute/cluster_worker.py"
                          })

    # results = pd.read_hdf(res_file, key='/Results/r_E0_df')

#     peaks_freq = np.zeros((len(ei_ratio), len(io_ratio)))
#     peaks_amp = np.zeros_like(peaks_freq)
#     for k_ee_, k_ei_, k_ie_, k_ii_ in zip(params['k_ee'], params['k_ei'], params['k_ie'], params['k_ii']):
#         if not results[k_ee_][k_ei_][k_ie_][k_ii_].isnull().any().any():
#             r, c = np.argmin(np.abs(ei_ratio - k_ee_ / k_ii_)), np.argmin(np.abs(io_ratio - k_ee_ / k_ie_))
#             data = np.array(results[k_ee_][k_ei_][k_ie_][k_ii_].loc[100.0:])
#             peaks, props = sp.find_peaks(data.squeeze(), prominence=0.6 * (np.max(data) - np.mean(data)),
#                                          distance=1 / dts)
#             if len(peaks) > 1:
#                 diff = np.mean(np.diff(peaks)) * dts * 0.01
#                 peaks_freq[r, c] = 1 / diff
#                 peaks_amp[r, c] = np.mean(props['prominences'])
#
#     step_tick_x, step_tick_y = int(peaks_freq.shape[1] / 10), int(peaks_freq.shape[0] / 10)
#     cm1 = cubehelix_palette(n_colors=int(len(ei_ratio) * len(io_ratio)), as_cmap=True, start=2.5, rot=-0.1)
#     cax1 = plot_connectivity(peaks_freq, ax=ax[0, idx], yticklabels=list(np.round(ei_ratio, decimals=2)),
#                              xticklabels=list(np.round(io_ratio, decimals=2)), cmap=cm1)
#     for n, label in enumerate(ax[0, idx].xaxis.get_ticklabels()):
#         if n % step_tick_x != 0:
#             label.set_visible(False)
#     for n, label in enumerate(ax[0, idx].yaxis.get_ticklabels()):
#         if n % step_tick_y != 0:
#             label.set_visible(False)
#     cax1.set_xlabel('intra/inter pcs')
#     cax1.set_ylabel('exc/inh pcs')
#     cax1.set_title(f'max freq (C = {C})')
#
#     cm2 = cubehelix_palette(n_colors=int(len(ei_ratio) * len(io_ratio)), as_cmap=True, start=-2.0, rot=-0.1)
#     cax2 = plot_connectivity(peaks_amp, ax=ax[1, idx], yticklabels=list(np.round(ei_ratio, decimals=2)),
#                              xticklabels=list(np.round(io_ratio, decimals=2)), cmap=cm2)
#     for n, label in enumerate(ax[1, idx].xaxis.get_ticklabels()):
#         if n % step_tick_x != 0:
#             label.set_visible(False)
#     for n, label in enumerate(ax[1, idx].yaxis.get_ticklabels()):
#         if n % step_tick_y != 0:
#             label.set_visible(False)
#     cax2.set_xlabel('intra/inter pcs')
#     cax2.set_ylabel('exc/inh pcs')
#     cax2.set_title(f'mean peak amp (C = {C})')
#
# plt.suptitle('EI-circuit sensitivity to population Coupling strengths (pcs)')
# print(f'Elapsed time: {t.time()-t0} seconds')
# # plt.tight_layout(pad=2.5, rect=(0.01, 0.01, 0.99, 0.96))
# fig.savefig('/data/hu_salomon/Documents/EIC_Coupling_alpha_0_high_res', format="svg")
#
# plt.show()

