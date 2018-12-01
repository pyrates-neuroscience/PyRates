from pyrates.frontend.circuit import CircuitTemplate
from pyrates.ir.circuit import CircuitIR
from pyrates.backend import ComputeGraph
import numpy as np
import matplotlib.pyplot as plt

# parameters
dt = 1e-4
sim_time = 1.
inp = np.zeros((int(sim_time/dt), 3)) + 0.

# network set-up
circuit = CircuitTemplate.from_yaml("pyrates.examples.simple_nextgen_NMM.JRC").apply()
c = CircuitIR()
for i in range(1):
    c.add_circuit(f"jrc.{i}", circuit)
compute_graph = ComputeGraph(c, vectorize="none", dt=dt)

# simulation
result, _ = compute_graph.run(sim_time,
                              inputs={("all", "PRO.0", "I"): inp},
                              outputs={"V": ("PC", "PRO.0", "PSP")},
                              out_dir="/tmp/log")

# plotting
result.pop("time")
result.plot()
plt.show()
