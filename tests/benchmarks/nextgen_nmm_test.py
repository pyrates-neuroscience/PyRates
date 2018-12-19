
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
from pyrates.backend import ComputeGraph
from pyrates.utility import plot_timeseries
import numpy as np
import matplotlib.pyplot as plt

# parameters
dt = 1e-4
T = 30.0
inp = np.zeros((int(T/dt), 1))
inp[int(1./dt):int((T-9.)/dt)] = 3.

# network set-up
circuit = CircuitTemplate.from_yaml("pyrates.examples.simple_nextgen_NMM.Net1").apply()
compute_graph = ComputeGraph(circuit, vectorization="none", dt=dt)

# simulation
result, _ = compute_graph.run(T,
                              inputs={("Pop1.0", "OP1.0", "inp"): inp},
                              outputs={"r": ("Pop1.0", "OP1.0", "r"),
                                       "v": ("Pop1.0", "OP1.0", "v")})

# plotting
fig, axes = plt.subplots(nrows=2, figsize=(15, 8))
plot_timeseries(result['r'], ax=axes[0])
plot_timeseries(result['v'], ax=axes[1])
plt.show()
