# pyrates imports
from pyrates.frontend import EdgeTemplate, CircuitTemplate
from pyrates.backend import ComputeGraph
from pyrates.ir import CircuitIR

# additional imports
import numpy as np
import matplotlib.pyplot as plt

dt = 1e-1                                   # integration step size in s
dts = 1.0                                     # variable storage sub-sampling step size in s
sub = int(dts/dt)                              # sub-sampling rate
T = 20000                                        # total simulation time in ms
delay = 5000
N = 15
m = 5
inp = np.zeros((int(T/dt), N), dtype='float32')                 # external input to the population
dur = 50.0
i = 0
while (i+1)*dur < T-delay:
    if i*dur > delay:
        i_tmp = i % m
        inp[int(i*dur/dt):int(((i+1)*dur)/dt), i_tmp] = 1.0
    i += 1

#circuit = CircuitTemplate.from_yaml("model_templates.wilson_cowan.simple_wilsoncowan.WC_stp_net").apply()
C_ee = np.zeros((N, N))
C_ee[np.eye(N) == 0] = 3.5/N

C_ie = np.zeros((N, N))
C_ie[np.eye(N) == 0] = 6.0/N

circuit = CircuitIR()
edge1 = EdgeTemplate.from_yaml("model_templates.wilson_cowan.simple_wilsoncowan.WC_ee")
edge2 = EdgeTemplate.from_yaml("model_templates.wilson_cowan.simple_wilsoncowan.WC_ie")
for idx in range(N):
    circuit.add_circuit(f'wc_{idx}', CircuitIR.from_yaml("model_templates.wilson_cowan.simple_wilsoncowan.WC_stp_net")
                        )
circuit.add_edges_from_matrix(source_var="WC_e_op/re", target_var="WC_e_op/re_in", template=edge1,
                              nodes=[f'wc_{idx}/E' for idx in range(N)], weight=C_ee)
circuit.add_edges_from_matrix(source_var="E/WC_e_op/re", target_var="I/WC_i_op/re_in", template=edge2,
                              nodes=[f'wc_{idx}' for idx in range(N)], weight=C_ie)

compute_graph = ComputeGraph(circuit, vectorization=True, backend='numpy', name='wc_net', step_size=dt,
                             solver='euler')

result, t = compute_graph.run(T,
                              inputs={"all/E/WC_e_op/I": inp},
                              outputs={"meg": "all/E/WC_e_op/meg"},
                              sampling_step_size=dts,
                              profile=True,
                              verbose=True,
                              )
result.loc[1.0:, :].mean(axis=1).plot()
plt.show()
