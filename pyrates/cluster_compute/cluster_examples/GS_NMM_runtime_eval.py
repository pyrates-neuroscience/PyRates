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

# meta infos
__author__ = "Christoph Salomon"
__status__ = "development"

# disable TF-gpu warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

grid_size = [1,    2,    3,    4,    5,    6,    7,    8,    9,
             10,   20,   30,   40,   50,   60,   70,   80,   90,
             100,  200,  300,  400,  500,  600,  700,  800,  900,
             1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
             10000]

##################
# Run GridSearch #
##################
runtime = {}

for i, size in enumerate(grid_size):
    print("")
    print(f'Simulation time: {2.0} seconds')
    print(f'Timestep: {dt} seconds')
    print(f'Grid size: {size}')

    params = {'J_e': np.linspace(8., 12., size), 'J_i': np.linspace(2., 12., size)}

    t0 = t.time()
    _, t_dict, = grid_search_2(
        circuit_template="pyrates.examples.simple_nextgen_NMM.Net5",
        param_grid=params, param_map=param_map,
        inputs={("PC", "Op_e.0", "inp"): inp},
        outputs={"r": ("PC", "Op_e.0", "r")},
        dt=dt, simulation_time=T, permute_grid=False, timestamps=True, prec=3)
    runtime[f'{size}'] = [t_dict['graph'], t_dict['run'], t_dict['total']]

    t_total = np.round(t.time()-t0, decimals=3)
    print(f'Elapsed time: {t_total:.3f} seconds')

# Create DataFrame and save results
###################################
cols = ['Graph (s)', 'Run (s)', 'Total (s)']
df_runtime = pd.DataFrame.from_dict(runtime, orient='index', columns=cols)
df_runtime.to_csv("/data/hu_salomon/Documents/GridSearch_runtime_eval.csv", index=True)


