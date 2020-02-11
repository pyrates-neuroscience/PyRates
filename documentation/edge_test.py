# pyrates imports
from pyrates.frontend import CircuitTemplate
from pyrates.backend import ComputeGraph

# additional imports
import numpy as np
import matplotlib.pyplot as plt

dt = 1e-1                                      # integration step size in s
dts = 5e-1                                     # variable storage sub-sampling step size in s
sub = int(dts/dt)                              # sub-sampling rate
T = 1000.0                                     # total simulation time in ms

inp = np.zeros((int(T/dt), 1), dtype='float32')                 # external input to the population
dur = 100.0
inp[int(dur/dt):int(dur/dt), :] = 10.0

circuit = CircuitTemplate.from_yaml("model_templates.wilson_cowan.simple_wilsoncowan.RC").apply()
compute_graph = ComputeGraph(circuit, vectorization=True, backend='numpy', name='wc_net', step_size=dt,
                             solver='euler')

result, t = compute_graph.run(T,
                              inputs={"P1/Op_rate/I_ext": inp},
                              outputs={"r1": "P1/Op_rate/r", "r2": "P2/Op_rate/r"},
                              sampling_step_size=dts,
                              profile=True,
                              verbose=True,
                              )
result.plot()
plt.show()
