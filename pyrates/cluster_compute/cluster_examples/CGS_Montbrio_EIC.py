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
from pyrates.cluster_compute.montbrio_eval_funcs import *
from pyrates.cluster_compute.cluster_compute import *

# meta infos
__author__ = "Christoph Salomon"
__status__ = "development"


if __name__ == "__main__":

    # Create CGS instance
    #####################
    # Directory, where the CGS project with all its files is stored
    compute_dir = "/nobackup/spanien1/salomon/ClusterGridSearch/Montbrio/EIC/T10s_dt10us"

    nodes = [
        'animals',
        'spanien',
        'tschad',
        # 'carpenters',
        'osttimor'
        # 'lech',
        # 'tiber',
        # 'uganda',
        # 'main'
    ]

    cgs = ClusterGridSearch(nodes, compute_dir=compute_dir)

    # Create configuration file
    ###########################
    param_map = {'k_e': {'var': [('Op_e.0', 'k_ee'), ('Op_i.0', 'k_ie')],
                         'nodes': ['E.0', 'I.0']},
                 'k_i': {'var': [('Op_e.0', 'k_ei'), ('Op_i.0', 'k_ii')],
                         'nodes': ['E.0', 'I.0']}
                 }

    config_file = f'{compute_dir}/Config/Montbrio_EIC_config.json'
    create_cgs_config(fp=config_file,
                      circuit_template="/data/hu_salomon/PycharmProjects/PyRates/models/Montbrio/Montbrio.EI_Circuit",
                      param_map=param_map,
                      simulation_time=10.,
                      dt=1e-5,
                      inputs={},
                      outputs={"r": ("E", "Op_e.0", "r")},
                      sampling_step_size=1e-3)

    # Create parameter grid
    #######################
    params = {'k_e': np.linspace(10., 30., 101), 'k_i': np.linspace(10., 30., 101)}

    # Run cluster grid search
    #########################
    res_file, grid_file = cgs.run(config_file=config_file,
                                  param_grid=params,
                                  permute=True,
                                  chunk_size=1000,
                                  thread_kwargs={
                                      "worker_env": "/data/u_salomon_software/anaconda3/envs/PyRates/bin/python3",
                                      "worker_file": "/data/hu_salomon/PycharmProjects/PyRates/"
                                                     "pyrates/cluster_compute/cluster_worker.py"
                                  })

    # results = pd.read_hdf(res_file, key="/Results/r_E0_df")
    # print(results)

    # Create plots
    ##############
    # plot_avrg_peak_dist(results, parameters=param_map, tick_size=1)
    # plot_time_series(results)
