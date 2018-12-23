from pyrates.frontend.template.circuit.circuit import CircuitTemplate
from pyrates.ir import CircuitIR
from pyrates.backend import ComputeGraph
from pyrates.utility import plot_timeseries, grid_search
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

circuit = CircuitTemplate.from_yaml("pyrates.examples.jansen_rit.simple_jr.JRC").apply()

c = CircuitIR()
N = 5
for i in range(N):
    c.add_circuit(f"jrc.{i}", circuit)

T = 1.
dt = 1e-3
compute_graph = ComputeGraph(c, vectorization="nodes", dt=dt)
inp = 220. + np.random.randn(int(T/dt), N) * 0.
result = compute_graph.run(T, outputs={"V": ("all", "PRO.0", "PSP")}, inputs={("PC", "RPO_e_pc.0", "u"): inp})
#plot_timeseries(result['V']
df = result['V']
plot_timeseries(df)
#df['input'] = inp
#plot_timeseries(df, plot_style='ridge_plot', demean=True, light=0.6, dark=0.3, hue=.95,
#                n_colors=6, hspace=-.01, fontsize=28, start=-3.0, rot=-0.2, aspect=20, height=2.0)
plt.show()

# parameter definition
# dt = 1e-4
# T = 82.
# taus = np.array([0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 1., 1., 1., 1., 10., 10., 10., 10.])
# taus2 = np.array([0.01, 0.1, 1., 10., 0.01, 0.1, 1., 10., 0.01, 0.1, 1., 10., 0.01, 0.1, 1., 10.])
# inp1 = np.zeros((int(T/dt), 1))
# inp1[int(6./dt):int((T-20.)/dt)] = 3.
# inp2 = np.zeros((int(T/dt), 1))
# inp2[int(4./dt):, 0] = 1. * np.sin(np.pi/20. * np.arange(4., T, dt))
#
# from pyrates.utility import grid_search
# params = {'Pop1.0/Op_tau_e.0/tau': taus, 'Pop1.0/Op_syn_e.0/tau': taus2}
#
# # pyrates simulation
# df = grid_search("pyrates.examples.simple_nextgen_NMM.Net3", params, T,
#                  inputs={("Pop", "Op_tau_e.0", "inp"): inp1, ("Pop", "Op_syn_e.0", "r_in"): inp2},
#                  outputs={"v": ("all", "Op_tau_e.0", "v")},
#                  dt=dt)
#
# # plotting
# plot_timeseries(df['v'].iloc[df.index > 1.0, :], plot_style='ridge_plot', demean=True, light=0.6, dark=0.3, hue=.95,
#                 n_colors=6, hspace=-.01, fontsize=28, start=-3.0, rot=-0.2, aspect=20, height=2.0)
# plt.show()
