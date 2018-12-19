
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
from pyrates.frontend import CircuitTemplate
from pyrates.ir.circuit import CircuitIR
from pyrates.backend import ComputeGraph
from matplotlib.pyplot import *

# parameters
n_jrcs = 2

# circuit IR setup
circuit_temp = CircuitTemplate.from_yaml("pyrates.examples.jansen_rit.circuit.JansenRitCircuit")
circuits = {}
for n in range(n_jrcs):
    circuits['jrc.' + str(n)] = circuit_temp
circuit_ir = CircuitIR.from_circuits('jrc_net', circuits=circuits).network_def(revert_node_names=True)

# create backend
net = ComputeGraph(circuit_ir, vectorization='ops')
#inp_pc = 220. + np.random.randn(3000, n_jrcs) * 0.
#inp_in = np.zeros((3000, n_jrcs))
results, _ = net.run(simulation_time=3.,
                     outputs={'v': ('all', 'JansenRitPRO.0', 'V')},
 #                    inputs={('JR_PC', 'JansenRitExcitatorySynapseRCO.0', 'u'): inp_pc,
 #                            ('JR_IIN', 'JansenRitExcitatorySynapseRCO.0', 'u'): inp_in,
 #                            ('JR_EIN', 'JansenRitExcitatorySynapseRCO.0', 'u'): inp_in})
                     )
results.pop('time')
results.plot()
show()
