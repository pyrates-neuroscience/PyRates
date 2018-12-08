from pyrates.frontend.circuit import CircuitTemplate
from pyrates.ir.circuit import CircuitIR
from pyrates.backend import ComputeGraph
import numpy as np
import matplotlib.pyplot as plt

# parameters
dt = 1e-3
sim_time = 10.0
inp = np.zeros((int(sim_time/dt), 3)) + 0.

# network set-up
circuit = CircuitTemplate.from_yaml("pyrates.examples.simple_nextgen_NMM.JRC").apply()
c = CircuitIR()
for i in range(4):
    c.add_circuit(f"jrc.{i}", circuit)
compute_graph = ComputeGraph(c, vectorization="none", dt=dt)

# simulation
result, _ = compute_graph.run(sim_time,
                              #inputs={("all", "PRO.0", "I"): inp},
                              outputs={"r": ("PC", "PRO.0", "r")},
                              out_dir="/tmp/log")

# plotting
result.pop("time")
result.plot()
plt.show()
