from pyrates.frontend.circuit import CircuitTemplate
from pyrates.backend import ComputeGraph
import matplotlib.pyplot as plt

circuit = CircuitTemplate.from_yaml("pyrates.examples.jansen_rit.JRC").apply()

nd = circuit.network_def()

compute_graph = ComputeGraph(nd, vectorize="nodes")

result, _ = compute_graph.run(10., outputs={"V": ("PC", "PRO.0", "PSP")})

result.pop("time")
result.plot()
plt.show()
