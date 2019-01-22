from pyrates.frontend.template.circuit.circuit import CircuitTemplate
from pyrates.ir import CircuitIR
from pyrates.backend import ComputeGraph
from pyrates.utility import plot_timeseries
import matplotlib.pyplot as plt
import numpy as np

circuit = CircuitIR()
template = CircuitTemplate.from_yaml("pyrates.examples.jansen_rit.simple_jr.JRC").apply()
circuit.add_circuit('jrc', template)

T = 1.
dt = 1e-5
inp = 220. + np.random.randn(int(T/dt), 1) * 0.

fig, axes = plt.subplots(figsize=(15, 5))
for dt_sampling, cmap in zip([1e-3, 1e-4, 1e-5], ['Reds', 'Blues', 'Greens']):
    compute_graph = ComputeGraph(circuit, vectorization="nodes", dt=dt, build_in_place=False)
    results = compute_graph.run(T, outputs={"V": ("PC", "PRO.0", "PSP")}, inputs={("PC", "RPO_e_pc.0", "u"): inp},
                                sampling_step_size=dt_sampling)
    axes = plot_timeseries(results, ax=axes, cmap=cmap)

plt.legend([1e-3, 1e-4, 1e-5])
plt.show()
