from pyrates.frontend.circuit import CircuitTemplate
from pyrates.ir.circuit import CircuitIR
from pyrates.backend import ComputeGraph
import matplotlib.pyplot as plt

circuit = CircuitTemplate.from_yaml("pyrates.examples.jansen_rit.JRC").apply()

c = CircuitIR()
for i in range(4):
    c.add_circuit(f"jrc.{i}", circuit)

compute_graph = ComputeGraph(c, vectorize="none")

result, _ = compute_graph.run(10., outputs={"V": ("PC", "PRO.0", "PSP")})

result.pop("time")
result.plot()
plt.show()
