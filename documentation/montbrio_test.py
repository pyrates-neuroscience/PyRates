# pyrates _imports
from pyrates.frontend import CircuitTemplate
from pyrates.utility.visualization import plot_timeseries, create_cmap

# additional _imports
import numpy as np
import matplotlib.pyplot as plt

dt = 1e-3                                      # integration step size in s
dts = 1e-3                                     # variable storage sub-sampling step size in s
sub = int(dts/dt)                              # sub-sampling rate
T = 800.0                                        # total simulation time in s
#inp = np.zeros((int(T/dt), 1), dtype='float32')                 # external input to the population
#inp[int(20./dt):int((T-20.)/dt)] = 5.

circuit = CircuitTemplate.from_yaml("../model_templates/montbrio/simple_montbrio.QIF_sfa_exp"
                                    ).apply(node_values={'p/Op_sfa_exp/eta': -3.96,
                                                         'p/Op_sfa_exp/J': 15.0*np.sqrt(2.0),
                                                         'p/Op_sfa_exp/alpha': 0.7}
                                            )
compute_graph = circuit.compile(vectorization=True, backend='numpy', name='montbrio', solver='scipy')

result, t = compute_graph.run(T,
                              step_size=dt,
                              #inputs={"E/Op_e_adapt/inp": inp},
                              outputs={"r": "p/Op_sfa_exp/r",
                                       "v": "p/Op_sfa_exp/v",
                                       "A": "p/Op_sfa_exp/I_a"},
                              sampling_step_size=dts,
                              profile='t',
                              verbose=True,
                              method='LSODA'
                              )

result.plot()
cmap = create_cmap("inferno", as_cmap=False, n_colors=3)
plot_timeseries(result.loc[10.0:, :], plot_style="ridge_plot", demean=True, hspace=-.01, fontsize=28, aspect=6,
                height=2.0, cmap=cmap)
plt.show()
