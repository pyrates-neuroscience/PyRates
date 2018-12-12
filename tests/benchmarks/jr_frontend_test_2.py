from pyrates.frontend.circuit import CircuitTemplate
from pyrates.ir.circuit import CircuitIR
from pyrates.backend import ComputeGraph
from pyrates.utility import plot_timeseries, grid_search
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

#circuit = CircuitTemplate.from_yaml("pyrates.examples.jansen_rit.simple_jr.JRC").apply()

#c = CircuitIR()
#for i in range(1):
#    c.add_circuit(f"jrc.{i}", circuit)

#compute_graph = ComputeGraph(c, vectorization="full")
#inp = 220. + np.random.randn(int(1./1e-3), 1) * 22.
# result, _ = compute_graph.run(1., outputs={"V": ("all", "PRO.0", "PSP")}, inputs={("PC.0", "RPO_e_pc.0", "u"): inp})


grid = DataFrame(np.zeros((2, 1)), columns=['EIN.0/RPO_e.0/tau'])
grid.iloc[0, 0] = 10e-3
grid.iloc[1, 0] = 15e-3
inp = 220. + np.random.randn(int(1./1e-3), 1) * 22.
result = grid_search("pyrates.examples.jansen_rit.simple_jr.JRC", param_grid=grid, simulation_time=1.,
                     outputs={"V": ("all", "PRO.0", "PSP")}, inputs={("PC", "RPO_e_pc.0", "u"): inp},
                     dt=1e-3, vectorization='nodes')

result.plot()
plt.show()
