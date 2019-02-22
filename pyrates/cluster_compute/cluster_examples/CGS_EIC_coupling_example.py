#e External imports
import matplotlib.pyplot as plt
from seaborn import cubehelix_palette

# PyRates internal imports
from pyrates.cluster_compute.cluster_compute import *
from pyrates.utility import plot_timeseries, plot_psd, plot_connectivity

# np.set_printoptions(precision=3)
# pd.set_option('precision', 3)

#####################################
# Create ClusterGridSearch instance #
#####################################
nodes = [
        # 'animals',
        'spanien',
        'carpenters',
        'osttimor'
        ]

compute_dir = "/nobackup/spanien1/salomon/ClusterGridSearch/CGS_EIC_coupling_example_3"

cgs = ClusterGridSearch(nodes, compute_dir=compute_dir)

############################
# Global config parameters #
############################
config_file = f'{compute_dir}/Config/EI_circuit_example.json'

circuit_template = "/data/hu_salomon/PycharmProjects/PyRates/models/BGTCS/EI_circuit.Net"

dt = 1e-4
T = 5.
inp = (3. + np.random.randn(int(T/dt), 1) * 1.0).tolist()

param_map = {'J_e': {'var': [('Op_e.0', 'J')],
                     'nodes': ['PC.0']},
             'J_i': {'var': [('Op_i.0', 'J')],
                     'nodes': ['IIN.0']},
             'k_ei': {'var': [(None, 'weight')],
                      'edges': [('PC.0', 'IIN.0', 0)]},
             'k_ie': {'var': [(None, 'weight')],
                      'edges': [('IIN.0', 'PC.0', 0)]}
             }
inputs = {("PC", "Op_e.0", "i_in"): inp}
outputs = {"r": ("PC", "Op_e.0", "r")}

create_cgs_config(fp=config_file, circuit_template=circuit_template,
                  param_map=param_map, dt=dt, simulation_time=T, inputs=inputs,
                  outputs=outputs, sampling_step_size=1e-3)

#########################
# Run ClusterGridSearch #
#########################
worker_env = "/data/u_salomon_software/anaconda3/envs/PyRates/bin/python"
worker_file = "/data/hu_salomon/PycharmProjects/PyRates/pyrates/cluster_compute/cluster_workers/gridsearch_worker.py"

# Parameter grid #
##################
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

    print("Starting cluster grid search!")
    t0 = t.time()
    res_dir, grid_file = cgs.run(thread_kwargs={
                                    "worker_env": worker_env,
                                    "worker_file": worker_file
                                 },
                                 config_file=config_file,
                                 chunk_size="dist_equal_add_mod",
                                 # chunk_size=50,
                                 param_grid=params,
                                 permute=False)
    print(f'Cluster grid search elapsed time: {t.time() - t0:.3f} seconds')

    results = gather_cgs_results(res_dir)

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
# fig.savefig("/home/rgast/Documents/Studium/PhD_Leipzig/Figures/BGTCS/eic_coupling", format="svg")
plt.show()
# fig.savefig("/data/hu_salomon/Pictures/EIC_Coupling", format="svg")
