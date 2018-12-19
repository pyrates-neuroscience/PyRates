
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
from pyrates.frontend.circuit import CircuitTemplate
from pyrates.ir.circuit import CircuitIR
from pyrates.backend import ComputeGraph
from pyrates.utility import plot_timeseries, grid_search
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

# circuit = CircuitTemplate.from_yaml("pyrates.examples.jansen_rit.simple_jr.JRC").apply()
#
# c = CircuitIR()
# for i in range(1):
#    c.add_circuit(f"jrc.{i}", circuit)
#
# compute_graph = ComputeGraph(c, vectorization="none")
# inp = 220. + np.random.randn(int(1./1e-3), 1) * 22.
# result, _ = compute_graph.run(1., outputs={"V": ("all", "PRO.0", "PSP")}, inputs={("PC", "RPO_e_pc.0", "u"): inp},
#                               out_dir="/tmp/log")


# parameter definition
dt = 5e-4
T = 3.
C = np.array([135.])
inp = np.random.uniform(120., 320., (int(T/dt), 1))

def adjust_weights(c, net):
    for s, t, e in net.edges:
        net.edges[s, t, e]['weight'] *= c

# pyrates simulation
results = []
for c in C:
    jrc_config = CircuitTemplate.from_yaml("pyrates.examples.jansen_rit.simple_jr.JRC").apply()
    adjust_weights(c/135., jrc_config)
    jrc_model = ComputeGraph(jrc_config, dt=dt, vectorization='full')
    result = jrc_model.run(simulation_time=T, outputs={f'C_{c}': ('PC', 'PRO.0', 'PSP')},
                           inputs={('PC', 'RPO_e_pc.0', 'u'): inp}, verbose=False)
    results.append(result)

df = DataFrame()
for r, c in zip(results, C):
    df[f'C = {c}'] = r.iloc[r.index > 1.0, 0]
plot_timeseries(df, plot_style='ridge_plot', demean=True, light=0.6, dark=0.3, hue=.95, n_colors=6, hspace=-.07,
                fontsize=12, start=-1.85, rot=-0.2)
plt.show()
