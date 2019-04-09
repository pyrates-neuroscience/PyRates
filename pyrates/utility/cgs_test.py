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
from pyrates.utility.grid_search import *

# meta infos
__author__ = "Christoph Salomon"
__status__ = "development"

# Create CGS instance
#####################
# Directory that is used to store the CGS project and all its files
compute_dir = "/nobackup/spanien1/salomon/ClusterGridSearch/Montbrio/EIC/Test_1"

nodes = [
    'animals',
    # 'spanien',
    'tschad',
    # 'carpenters',
    # 'osttimor'
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

dts = 1e-3

config_file = f'{compute_dir}/Config/Montbrio_EIC_config.json'

create_cgs_config(fp=config_file,
                  circuit_template="/data/hu_salomon/PycharmProjects/MasterThesis/Montbrio/Montbrio/EI_Circuit",
                  param_map=param_map,
                  simulation_time=5.,
                  dt=5e-4,
                  inputs={},
                  outputs={"r": ("E", "Op_e.0", "r")},
                  sampling_step_size=dts)

# Create parameter grid
#######################
params = {'k_e': np.linspace(1., 100., 101), 'k_i': np.linspace(1., 100., 101)}

# Run cluster grid search
#########################
res_file = cgs.run(config_file=config_file,
                   params=params,
                   permute=True,
                   chunk_size=1000,
                   worker_env="/data/u_salomon_software/anaconda3/envs/PyRates/bin/python3",
                   worker_file="/data/hu_salomon/PycharmProjects/PyRates/pyrates/utility/cluster_worker.py"
                   )

try:
    results = pd.read_hdf(res_file, key="/Results/r_E0_df")
except (KeyError, FileNotFoundError) as e:
    print(e)
