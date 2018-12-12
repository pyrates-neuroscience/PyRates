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
