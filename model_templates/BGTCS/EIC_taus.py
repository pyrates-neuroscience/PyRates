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

taus = [0.01, 0.02, 0.04]
tau_e_scaling = np.arange(0.1, 2., 0.1)[::-1]
ei_ratio = np.arange(0.1, 1., 0.05)

fig, ax = plt.subplots(ncols=len(taus), nrows=2, figsize=(20, 15), gridspec_kw={})
for i, t in enumerate(taus):

    n = 0
    tau = np.zeros((int(len(tau_e_scaling) * len(ei_ratio)))) + t
    tau_i = np.zeros_like(tau)
    tau_e = np.zeros_like(tau)
    for ts in tau_e_scaling:
        for r in ei_ratio:
            tau_e[n] = t * ts * r
            tau_i[n] = t * ts
            n += 1
    params = {'tau_e': tau_e, 'tau_i': tau_i, 'tau': tau}
    param_map = {'tau_e': {'var': [('Op_e.0', 'tau_e'), ('Op_i.0', 'tau_e')],
                           'nodes': ['PC.0', 'IIN.0']},
                 'tau_i': {'var': [('Op_e.0', 'tau_i'), ('Op_i.0', 'tau_i')],
                           'nodes': ['PC.0', 'IIN.0']},
                 'tau': {'var': [('Op_e.0', 'tau'), ('Op_i.0', 'tau')],
                         'nodes': ['PC.0', 'IIN.0']}
                 }

    # perform simulation
    results = grid_search(circuit_template="EI_circuit.Net",
                          param_grid=params, param_map=param_map,
                          inputs={("PC", "Op_e.0", "i_in"): inp}, outputs={"r": ("PC", "Op_e.0", "r")},
                          dt=dt, simulation_time=T, permute_grid=False, sampling_step_size=1e-3)

    # plotting
    cut_off = 1.
    max_freq = np.zeros((len(tau_e_scaling), len(ei_ratio)))
    freq_pow = np.zeros_like(max_freq)
    for t_e, t_i, t_tmp in zip(params['tau_e'], params['tau_i'], params['tau']):
        if not results[t_e][t_i][t_tmp].isnull().any().any():
            _ = plot_psd(results[t_e][t_i][t_tmp], tmin=cut_off, show=False)
            pow = plt.gca().get_lines()[-1].get_ydata()
            freqs = plt.gca().get_lines()[-1].get_xdata()
            r, c = np.argmin(np.abs(tau_e_scaling - t_i/t)), np.argmin(np.abs(ei_ratio - t_e/t_i))
            max_freq[r, c] = freqs[np.argmax(pow)]
            freq_pow[r, c] = np.max(pow)
            plt.close(plt.gcf())

    cm1 = cubehelix_palette(n_colors=int(len(tau_e_scaling)*len(ei_ratio)), as_cmap=True, start=2.5, rot=-0.1)
    cm2 = cubehelix_palette(n_colors=int(len(tau_e_scaling)*len(ei_ratio)), as_cmap=True, start=-2.0, rot=-0.1)
    cax1 = plot_connectivity(max_freq, ax=ax[0, i], yticklabels=list(np.round(tau_e_scaling, decimals=2)),
                             xticklabels=list(np.round(ei_ratio, decimals=2)), cmap=cm1)
    cax1.set_xlabel('tau_e/tau_i')
    cax1.set_ylabel('tau_e/tau')
    cax1.set_title(f'max freq (tau = {t})')
    cax2 = plot_connectivity(freq_pow, ax=ax[1, i], yticklabels=list(np.round(tau_e_scaling, decimals=2)),
                             xticklabels=list(np.round(ei_ratio, decimals=2)), cmap=cm2)
    cax2.set_xlabel('tau_e/tau_i')
    cax2.set_ylabel('tau_e/tau')
    cax2.set_title(f'freq power (tau = {t})')

plt.suptitle('EI-circuit sensitivity to synaptic and membrane time constants (tcs)')
plt.tight_layout(pad=2.5, rect=(0.01, 0.01, 0.99, 0.96))
fig.savefig("/home/rgast/Documents/Studium/PhD_Leipzig/Figures/BGTCS/eic_taus", format="svg")
plt.show()
