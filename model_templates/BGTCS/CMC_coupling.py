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
Cs = [1., 2., 4.]
ei_ratio = np.arange(0.5, 2., 0.1)[::-1]
l3l5_ratio = np.arange(0.5, 2., 0.1)
k_l3l5_e = np.zeros((int(len(ei_ratio) * len(l3l5_ratio))))
k_l3l5_i = np.zeros_like(k_l3l5_e)

fig, ax = plt.subplots(ncols=len(Cs), nrows=2, figsize=(20, 15), gridspec_kw={})
for idx, C in enumerate(Cs):

    k_l5l3_i = np.zeros_like(k_l3l5_e) + C
    n = 0
    for r1 in ei_ratio:
        for r2 in l3l5_ratio:
            k_l3l5_e[n] = C * r1 * r2
            k_l3l5_i[n] = C * r2
            n += 1

    params = {'k_l3l5_e': k_l3l5_e, 'k_l3l5_i': k_l3l5_i, 'k_l5l3_i': k_l5l3_i}
    param_map = {'k_l3l5_e': {'var': [(None, 'weight')],
                              'edges': [('L3/PC.0', 'L5/PC.0', 0)]},
                 'k_l3l5_i': {'var': [(None, 'weight')],
                              'edges': [('L3/PC.0', 'L5/IIN.0', 0)]},
                 'k_l5l3_i': {'var': [(None, 'weight')],
                              'edges': [('L5/PC.0', 'L3/IIN.0', 0)]},
                 }

    # perform simulation
    results = grid_search(circuit_template="EI_circuit.CMC",
                          param_grid=params, param_map=param_map,
                          inputs={("L3/PC.0", "Op_e.0", "i_in"): inp}, outputs={"r": ("L3/PC.0", "Op_e.0", "r")},
                          dt=dt, simulation_time=T, permute_grid=False, sampling_step_size=1e-3)

    # plotting
    cut_off = 1.
    max_freq = np.zeros((len(ei_ratio), len(l3l5_ratio)))
    freq_pow = np.zeros_like(max_freq)
    for k1, k2, k3 in zip(params['k_l3l5_e'], params['k_l3l5_i'], params['k_l5l3_i']):
        if not results[k1][k2][k3].isnull().any().any():
            _ = plot_psd(results[k1][k2][k3], tmin=cut_off, show=False)
            pow = plt.gca().get_lines()[-1].get_ydata()
            freqs = plt.gca().get_lines()[-1].get_xdata()
            r, c = np.argmin(np.abs(ei_ratio - k1/k2)), np.argmin(np.abs(l3l5_ratio - k2/k3))
            max_freq[r, c] = freqs[np.argmax(pow)]
            freq_pow[r, c] = np.max(pow)
            plt.close(plt.gcf())

    cm1 = cubehelix_palette(n_colors=int(len(ei_ratio)*len(l3l5_ratio)), as_cmap=True, start=2.5, rot=-0.1)
    cm2 = cubehelix_palette(n_colors=int(len(ei_ratio)*len(l3l5_ratio)), as_cmap=True, start=-2.0, rot=-0.1)
    cax1 = plot_connectivity(max_freq, ax=ax[0, idx], yticklabels=list(np.round(ei_ratio, decimals=2)),
                             xticklabels=list(np.round(l3l5_ratio, decimals=2)), cmap=cm1)
    cax1.set_xlabel('L3/L5 coupling strength')
    cax1.set_ylabel('exc/inh coupling strength')
    cax1.set_title(f'max freq (C = {C})')
    cax2 = plot_connectivity(freq_pow, ax=ax[1, idx], yticklabels=list(np.round(ei_ratio, decimals=2)),
                             xticklabels=list(np.round(l3l5_ratio, decimals=2)), cmap=cm2)
    cax2.set_xlabel('L3/L5 coupling strength')
    cax2.set_ylabel('exc/inh coupling strength')
    cax2.set_title(f'freq power (C = {C})')

plt.suptitle('Cortical microcircuit sensitivity to inter-laminar coupling')
plt.tight_layout(pad=2.5, rect=(0.01, 0.01, 0.99, 0.96))
fig.savefig("/home/rgast/Documents/Studium/PhD_Leipzig/Figures/BGTCS/cmc_coupling", format="svg")
plt.show()
