import numpy as np
from pyrates import CircuitTemplate, NodeTemplate
import matplotlib.pyplot as plt

julia = ""

# node definition
li = NodeTemplate.from_yaml("model_templates.base_templates.li_node")
N = 10
nodes = [f"p{i+1}" for i in range(N)]
net = CircuitTemplate(name="li_coupled", nodes={key: li for key in nodes})

# edge definition
C = np.random.uniform(low=-5.0, high=5.0, size=(N, N))
D = np.random.uniform(low=1.0, high=3.0, size=(N, N))
S = np.random.uniform(low=0.5, high=2.0, size=(N, N))
net.add_edges_from_matrix(source_var="li_op/r", target_var="li_op/m_in", nodes=nodes, weight=C,
                          edge_attr={'delay': D, 'spread': S})

# define input
T = 100.0
dt = 1e-4
inp = np.sin(2*np.pi*0.2*np.linspace(0, T, int(np.round(T/dt)))) * 0.1

# simulate time series
res = net.run(simulation_time=T, step_size=dt, sampling_step_size=1e-2, solver="scipy", method="DOP853",
              outputs={"all/li_op/r"}, inputs={"all/li_op/u": inp}, backend="julia", clear=False,
              constants_file_name="li_params.npz", julia_path=julia, func_name="li_eval")

# plotting
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(res)
plt.legend(nodes)
plt.show()
