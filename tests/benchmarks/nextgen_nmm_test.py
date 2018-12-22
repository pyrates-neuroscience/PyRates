from pyrates.frontend.circuit import CircuitTemplate
from pyrates.backend import ComputeGraph
from pyrates.utility import plot_timeseries
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

# parameters
dt = 5e-5
T = 2.
inp = np.zeros((int(T/dt), 2))
#inp[:, 0] = 3.
D = np.arange(0.0, 0.10, 0.002)


def adjust_delays(d, net, edges):
    for s, t, e in net.edges:
        if (s, t) in edges:
            net.edges[s, t, e]['delay'] = d


# network set-up
results = DataFrame()
for d in D:
    model_config = CircuitTemplate.from_yaml("pyrates.examples.simple_nextgen_NMM.Net6").apply()
    adjust_delays(d, model_config, [('PC1.0', 'PC2.0'), ('PC2.0', 'PC1.0')])
    montbrio = ComputeGraph(model_config, dt=dt, vectorization='full')
    result = montbrio.run(simulation_time=T,
                          inputs={("PC", "Op_e.0", "inp"): inp},
                          outputs={"r1": ("PC1.0", "Op_e.0", "r"),
                                   "r2": ("PC2.0", "Op_e.0", "r")},
                          verbose=False)

    plot_timeseries(result)
    plt.show()
    results[f"r1_d_{d}"] = result['r1']['PC1.0']
    results[f"r2_d_{d}"] = result['r2']['PC2.0']

# plotting
results['input'] = inp[:, 0]
plot_timeseries(results, plot_style='ridge_plot', demean=True, light=0.6, dark=0.3, hue=.95, n_colors=6, hspace=-.01,
                fontsize=28, start=-3.0, rot=-0.2, aspect=10, height=2.0)
plt.show()
