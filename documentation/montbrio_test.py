# pyrates imports
from pyrates.frontend import CircuitTemplate
from pyrates.utility.visualization import plot_timeseries, create_cmap

# additional imports
import numpy as np
import matplotlib.pyplot as plt

dt = 1e-3                                      # integration step size in s
dts = 1e-3                                     # variable storage sub-sampling step size in s
sub = int(dts/dt)                              # sub-sampling rate
T = 80.0                                        # total simulation time in s
#inp = np.zeros((int(T/dt), 1), dtype='float32')                 # external input to the population
#inp[int(20./dt):int((T-20.)/dt)] = 5.

circuit = CircuitTemplate.from_yaml("model_templates.montbrio.simple_montbrio.Net5").apply()
compute_graph = circuit.compile(vectorization=True, backend='numpy', name='montbrio', solver='scipy')

result, t = compute_graph.run(T,
                              step_size=dt,
                              #inputs={"E/Op_e_adapt/inp": inp},
                              outputs={"r": "E/Op_e_adapt/r",
                                       "v": "E/Op_e_adapt/v",
                                       "A": "E/Op_e_adapt/I_a"},
                              sampling_step_size=dts,
                              profile='t',
                              verbose=True,

                              )

result.plot()
cmap = create_cmap("inferno", as_cmap=False, n_colors=3)
plot_timeseries(result, plot_style="ridge_plot", demean=True, hspace=-.01, fontsize=28, aspect=6, height=2.0,
                cmap=cmap)
plt.show()
