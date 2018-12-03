from pyrates.frontend.circuit import CircuitTemplate
from pyrates.ir.circuit import CircuitIR
from pyrates.backend import ComputeGraph
import matplotlib.pyplot as plt

circuit = CircuitTemplate.from_yaml("pyrates.examples.jansen_rit.simple_jr.JRC").apply()

c = CircuitIR()
for i in range(3):
    c.add_circuit(f"jrc.{i}", circuit)

compute_graph = ComputeGraph(c, vectorize="nodes")

result, _ = compute_graph.run(1., outputs={"V": ("all", "PRO.0", "PSP")}, out_dir="/tmp/log")

result.pop("time")
result.plot()
plt.show()
