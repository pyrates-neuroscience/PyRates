from pyrates.frontend import CircuitTemplate
from pyrates.ir import CircuitIR
from pyrates.backend import ComputeGraph
import numpy as np
from scipy.io import loadmat
from pandas import DataFrame
from pyrates.utility import functional_connectivity, plot_connectivity, plot_timeseries, plot_psd
import matplotlib.pyplot as plt

# connectivity
k = 1.5
v = 6.2
C = loadmat("/home/rgast/PycharmProjects/PyRates/tests/resources/SC.mat")['SC']
D = loadmat("/home/rgast/PycharmProjects/PyRates/tests/resources/D.mat")['D']
C *= k
D /= v * 1e3
nodes = [f'circuit_{i}' for i in range(C.shape[0])]
svar = 'Op_e.0/r'
tvar = 'Op_e.0/r_e'
snodes = [f'{node}/PC.0/{svar}' for node in nodes]
tnodes = [f'{node}/PC.0/{tvar}' for node in nodes]
conn = DataFrame(columns=snodes, index=tnodes)
delay = DataFrame(columns=snodes, index=tnodes)
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        conn.loc[tnodes[j], snodes[i]] = C[j, i]
        delay.loc[tnodes[j], snodes[i]] = D[j, i]

# create network in pyrates
dt = 1e-4
circuits = {nodes[i]: CircuitTemplate.from_yaml("pyrates.examples.simple_nextgen_NMM.Net5").apply()
            for i in range(C.shape[0])}
net_config = CircuitIR.from_circuits('net', circuits=circuits, connectivity={'weight': conn, 'delay': delay})
net = ComputeGraph(net_config=net_config, dt=dt, vectorization='nodes')

# define input and run simulation
T = 10.
inp = 2. + np.random.randn(int(T/dt), C.shape[0]) * 1.0
results = net.run(simulation_time=T, inputs={('PC', 'Op_e.0', 'inp'): inp}, outputs={'r': ('PC', 'Op_e.0', 'r')})

# calculate and plot connectivity
plot_psd(results['r'], fmin=6., fmax=100., estimate='power', average=False, area_mode=None, spatial_colors=False)
#plot_timeseries(results['r'])
plt.show()
fc = functional_connectivity(data=results['r'], metric='coh', fmin=25., fmax=40., faverage=True)
fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
plot_connectivity(fc=fc, ax=axes[0], vmin=0., vmax=1.)
plot_connectivity(fc=C, ax=axes[1])
plt.show()
