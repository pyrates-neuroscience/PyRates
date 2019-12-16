# pyrates imports
from pyrates.frontend import CircuitTemplate
from pyrates.backend import ComputeGraph

# additional imports
import numpy as np
import matplotlib.pyplot as plt

dt = 1e-3                                      # integration step size in s
dts = 1e-2                                     # variable storage sub-sampling step size in s
sub = int(dts/dt)                              # sub-sampling rate
T = 20.                                        # total simulation time in s
inp = np.zeros((int(T/dt), 1), dtype='float32')                 # external input to the population
inp[int(5./dt):int((T-5.)/dt)] = 1.0

circuit = CircuitTemplate.from_yaml("model_templates.wilson_cowan.simple_wilsoncowan.WC_stp_net").apply()

compute_graph = ComputeGraph(circuit, vectorization=True, backend='numpy', name='wc', step_size=dt,
                             solver='scipy')

result, t = compute_graph.run(T,
                              inputs={"E/WC_e_op/I": inp},
                              outputs={"v_e": "E/WC_e_op/ve", "v_i": "I/WC_i_op/vi"},
                              sampling_step_size=dts,
                              profile=True,
                              verbose=True,
                              )

result.loc[1.0:, :].plot()
plt.show()
