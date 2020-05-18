from pyrates.utility.visualization import plot_timeseries, plot_psd, plot_connectivity
from pyrates.utility.grid_search import grid_search

import numpy as np
import matplotlib.pyplot as plt
from seaborn import cubehelix_palette

__author__ = "Richard Gast"
__status__ = "Development"


# parameters
dt = 1e-4
T = 5.
inp = 3. + np.random.randn(int(T/dt), 1) * 1.0

params = {'H_e': np.arange(0.1, 2.01, 0.1)[::-1], 'H_i': np.arange(0.1, 4.01, 0.1)}
param_map = {'H_e': {'var': [('Op_e.0', 'H_e'), ('Op_i.0', 'H_e')],
                     'nodes': ['PC.0', 'IIN.0']},
             'H_i': {'var': [('Op_e.0', 'H_i'), ('Op_i.0', 'H_i')],
                     'nodes': ['PC.0', 'IIN.0']},
             }

# perform simulation
results = grid_search(circuit_template="EI_circuit.Net",
                      param_grid=params, param_map=param_map,
                      inputs={("PC", "Op_e.0", "i_in"): inp}, outputs={"r": ("PC", "Op_e.0", "r")},
                      dt=dt, simulation_time=T, permute_grid=True, sampling_step_size=1e-3)

# plotting
cut_off = 1.
max_freq = np.zeros((len(params['H_e']), len(params['H_i'])))
freq_pow = np.zeros_like(max_freq)
for i, H_e in enumerate(params['H_e']):
    for j, H_i in enumerate(params['H_i']):
        if not results[H_e][H_i].isnull().any().any():
            _ = plot_psd(results[H_e][H_i], tmin=cut_off, show=False)
            pow = plt.gca().get_lines()[-1].get_ydata()
            freqs = plt.gca().get_lines()[-1].get_xdata()
            max_freq[i, j] = freqs[np.argmax(pow)]
            freq_pow[i, j] = np.max(pow)
            plt.close('all')

fig, ax = plt.subplots(ncols=2, figsize=(15, 5), gridspec_kw={})
cm1 = cubehelix_palette(n_colors=int(len(params['H_e'])*len(params['H_i'])), as_cmap=True, start=2.5, rot=-0.1)
cm2 = cubehelix_palette(n_colors=int(len(params['H_e'])*len(params['H_i'])), as_cmap=True, start=-2.0, rot=-0.1)
cax1 = plot_connectivity(max_freq, ax=ax[0], yticklabels=list(np.round(params['H_e'], decimals=2)),
                         xticklabels=list(np.round(params['H_i'], decimals=2)), cmap=cm1)
cax1.set_xlabel('H_i')
cax1.set_ylabel('H_e')
cax1.set_title(f'max freq')
cax2 = plot_connectivity(freq_pow, ax=ax[1], yticklabels=list(np.round(params['H_e'], decimals=2)),
                         xticklabels=list(np.round(params['H_i'], decimals=2)), cmap=cm2)
cax2.set_xlabel('H_i')
cax2.set_ylabel('H_e')
cax2.set_title(f'freq pow')
plt.suptitle('EI-circuit sensitivity to synaptic efficacies (H)')
plt.tight_layout(pad=2.5, rect=(0.01, 0.01, 0.99, 0.96))
fig.savefig("/home/rgast/Documents/Studium/PhD_Leipzig/Figures/BGTCS/eic_efficacies", format="svg")
plt.show()
