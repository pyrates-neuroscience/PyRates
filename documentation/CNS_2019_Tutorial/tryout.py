from pyrates.frontend import CircuitTemplate
from pyrates.backend import ComputeGraph
from pyrates.utility import plot_timeseries

import numpy as np
import matplotlib.pyplot as plt


dt = 1e-4                                      # integration step size in s
dts = 1e-2                                     # variable storage sub-sampling step size in s
sub = int(dts/dt)                              # sub-sampling rate
T = 22.                                        # total simulation time in s
inp = np.zeros((int(T/dt), 1))                 # external input to the population
inp[int(6./dt):int((T-12.)/dt)] = 3.

# basic montbrio population
# ec_config = CircuitTemplate.from_yaml("model_templates.CNS_2019_Tutorial.templates.EC").apply()
# ec = ComputeGraph(ec_config, dt, vectorization='none', name='EC', backend='numpy')
# results_ec = ec.run(T, inputs={("E.0", "Op_exc.0", "inp"): inp}, outputs={'r': ('E.0', 'Op_exc.0', 'r')})
#
# plot_timeseries(results_ec)
# plt.show()

# three population model (JRC architecture)
# jrc_config = CircuitTemplate.from_yaml("model_templates.CNS_2019_Tutorial.templates.JRC_nosyns").apply()
# jrc_nosyns = ComputeGraph(jrc_config, dt, vectorization='none', name='JRC_nosyns', backend='numpy')
# results_jrc_nosyns = jrc_nosyns.run(T, inputs={("PC.0", "Op_exc.0", "inp"): inp},
#                                     outputs={'r_PC': ('PC.0', 'Op_exc.0', 'r'),
#                                              'r_EIN': ('EIN.0', 'Op_exc.0', 'r'),
#                                              'r_IIN': ('IIN.0', 'Op_inh.0', 'r')})
#
# plot_timeseries(results_jrc_nosyns)
# plt.show()

# three population model (JRC architecture) with alpha-kernel synapses
jrc_config = CircuitTemplate.from_yaml("model_templates.CNS_2019_Tutorial.templates.JRC").apply()
jrc = ComputeGraph(jrc_config, dt, vectorization='none', name='JRC', backend='numpy')
results_jrc = jrc.run(T, inputs={("PC.0", "Op_exc_syns.0", "inp"): inp},
                      outputs={'r_PC': ('PC.0', 'Op_exc_syns.0', 'r'),
                               'r_EIN': ('EIN.0', 'Op_exc_syns.0', 'r'),
                               'r_IIN': ('IIN.0', 'Op_inh_syns.0', 'r')})

plot_timeseries(results_jrc)
plt.show()
