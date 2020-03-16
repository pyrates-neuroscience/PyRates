# pyrates imports
from pyrates.frontend import CircuitTemplate

# additional imports
import numpy as np
import matplotlib.pyplot as plt

dt = 1e-3                                      # integration step size in s
dts = 1e-3                                     # variable storage sub-sampling step size in s
sub = int(dts/dt)                              # sub-sampling rate
T = 80.0                                        # total simulation time in s
inp = np.zeros((int(T/dt), 1), dtype='float32')                 # external input to the population
inp[int(20./dt):int((T-20.)/dt)] = 5.

# circuit = CircuitTemplate.from_yaml("model_templates.montbrio.simple_montbrio.Net1").apply()
#
# compute_graph = ComputeGraph(circuit, vectorization=True, backend='numpy', name='montbrio', step_size=dt,
#                              solver='euler')
#
# result, t = compute_graph.run(T,
#                               inputs={"Pop1/Op_e/inp": inp},
#                               outputs={"r_e": "Pop1/Op_e/r"},
#                               sampling_step_size=dts,
#                               profile=True,
#                               verbose=True,
#                               )
#
# result.plot()
# plt.show()

circuit2 = CircuitTemplate.from_yaml("model_templates.montbrio.simple_montbrio.Net1").apply()
compute_graph2 = circuit2.compile(vectorization=True, backend='tensorflow', name='montbrio')

result2, t = compute_graph2.run(T,
                                step_size=dt,
                                inputs={"Pop1/Op_e/inp": inp},
                                outputs={"r": "Pop1/Op_e/r"},
                                sampling_step_size=dts,
                                solver='euler',
                                profile='t',
                                verbose=True,
                                )

#dir = compute_graph2.generate_auto_def(None)

result2.plot()
print(f"r: {result2['r'].iloc[-1]}")
plt.show()
