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
ei_ratio = np.arange(0.1, 3., 0.1)[::-1]
io_ratio = np.arange(0.1, 2., 0.1)
J_e = np.zeros((int(len(ei_ratio) * len(io_ratio))))
J_i = np.zeros_like(J_e)
k_ei = np.zeros_like(J_e)

fig, ax = plt.subplots(ncols=len(Cs), nrows=2, figsize=(20, 15), gridspec_kw={})
for idx, C in enumerate(Cs):

    k_ie = np.zeros_like(J_e) + C
    n = 0
    for r1 in ei_ratio:
        for r2 in io_ratio:
            J_e[n] = C * r1 * r2
            J_i[n] = C * r2
            k_ei[n] = C * r1
            n += 1

    params = {'J_e': J_e, 'J_i': J_i, 'k_ei': k_ei, 'k_ie': k_ie}
    param_map = {'J_e': {'var': ['Op_e/J'],
                         'nodes': ['PC']},
                 'J_i': {'var': ['Op_i/J'],
                         'nodes': ['IIN']},
                 'k_ei': {'var': [(None, 'weight')],
                          'edges': [('PC/IIN', 0)]},
                 'k_ie': {'var': [(None, 'weight')],
                          'edges': [('IIN', 'PC', 0)]}
                 }

    # perform simulation
    results, result_map = grid_search(circuit_template="EI_circuit.Net",
                                      param_grid=params, param_map=param_map,
                                      inputs={("PC", "Op_e", "i_in"): inp}, outputs={"r": ("PC", "Op_e", "r")},
                                      dt=dt, simulation_time=T, permute_grid=False, sampling_step_size=1e-3)

    # plotting
    cut_off = 1.
    max_freq = np.zeros((len(ei_ratio), len(io_ratio)))
    freq_pow = np.zeros_like(max_freq)
    for j_e, j_i, k1, k2 in zip(params['J_e'], params['J_i'], params['k_ei'], params['k_ie']):
        if not results[j_e][j_i][k1][k2].isnull().any().any():
            _ = plot_psd(results[j_e][j_i][k1][k2], tmin=cut_off, show=False)
            pow = plt.gca().get_lines()[-1].get_ydata()
            freqs = plt.gca().get_lines()[-1].get_xdata()
            r, c = np.argmin(np.abs(ei_ratio - k1/k2)), np.argmin(np.abs(io_ratio - j_i/k2))
            max_freq[r, c] = freqs[np.argmax(pow)]
            freq_pow[r, c] = np.max(pow)
            plt.close(plt.gcf())

    cm1 = cubehelix_palette(n_colors=int(len(ei_ratio)*len(io_ratio)), as_cmap=True, start=2.5, rot=-0.1)
    cm2 = cubehelix_palette(n_colors=int(len(ei_ratio)*len(io_ratio)), as_cmap=True, start=-2.0, rot=-0.1)
    cax1 = plot_connectivity(max_freq, ax=ax[0, idx], yticklabels=list(np.round(ei_ratio, decimals=2)),
                             xticklabels=list(np.round(io_ratio, decimals=2)), cmap=cm1)
    cax1.set_xlabel('intra/inter pcs')
    cax1.set_ylabel('exc/inh pcs')
    cax1.set_title(f'max freq (C = {C})')
    cax2 = plot_connectivity(freq_pow, ax=ax[1, idx], yticklabels=list(np.round(ei_ratio, decimals=2)),
                             xticklabels=list(np.round(io_ratio, decimals=2)), cmap=cm2)
    cax2.set_xlabel('intra/inter pcs')
    cax2.set_ylabel('exc/inh pcs')
    cax2.set_title(f'freq power (C = {C})')

plt.suptitle('EI-circuit sensitivity to population Coupling strengths (pcs)')
plt.tight_layout(pad=2.5, rect=(0.01, 0.01, 0.99, 0.96))
fig.savefig("/home/rgast/Documents/Studium/PhD_Leipzig/Figures/BGTCS/eic_coupling", format="svg")
plt.show()
