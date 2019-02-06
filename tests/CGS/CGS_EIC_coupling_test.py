from pyrates.utility import plot_timeseries, grid_search, plot_psd, plot_connectivity
import matplotlib.pyplot as plt
from seaborn import cubehelix_palette
from pyrates.utility.cluster_grid_search import *

__author__ = "Richard Gast"
__status__ = "Development"

print("Start!")
start = time.time()

home_dir = str(Path.home())

# Optional: Directory to use as compute directory for current CGS instance.
# If none is specified, default directory is created
compute_dir = f'{home_dir}/Documents/ClusterGridSearch/CGS_EIC_coupling_test'

############################
# Global config parameters #
############################
config_name = "EI_circuit_test1"
config_file = f'{home_dir}/Documents/ClusterGridSearch/{config_name}.json'

circuit_template = "/data/hu_salomon/PycharmProjects/PyRates/models/BGTCS/EI_circuit.Net"

# parameters
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
# Cluster configuration #
#########################
host_config = {
    'hostnames': [
        'animals',
        # 'spanien',
        # 'carpenters',
        'osttimor'
        ],
    'host_env_cpu': "/data/u_salomon_software/anaconda3/envs/PyRates/bin/python",
    'host_env_gpu': "",
    'host_file': "/data/hu_salomon/PycharmProjects/PyRates/pyrates/utility/cluster_worker.py",
    'host_dir': ""
}

# Create ClusterGridSearch instance
cgs = ClusterGridSearch(config_file, compute_dir=compute_dir)

# Create compute cluster
cgs.create_cluster(host_config)


##################
# Parameter grid #
##################
Cs = [1., 2., 4.]
ei_ratio = np.arange(0.1, 3., 0.1)[::-1]
io_ratio = np.arange(0.1, 2., 0.1)
J_e = np.zeros((int(len(ei_ratio) * len(io_ratio))))
J_i = np.zeros_like(J_e)
k_ei = np.zeros_like(J_e)

for idx, C in enumerate(Cs):

    k_ie = np.zeros_like(J_e) + C
    n = 0
    for r1 in ei_ratio:
        for r2 in io_ratio:
            J_e[n] = C * r1 * r2
            J_i[n] = C * r2
            k_ei[n] = C * r1
            n += 1

    param_grid = {'J_e': J_e, 'J_i': J_i, 'k_ei': k_ei, 'k_ie': k_ie}

    # Compute grid inside the cluster
    res_dir, grid_file = cgs.compute_grid(param_grid, num_params=300, permute=False)

elapsed = time.time() - start
print("Overall elapsed time: {0:.3f} seconds".format(elapsed))



# # plotting
# cut_off = 1.
# max_freq = np.zeros((len(ei_ratio), len(io_ratio)))
# freq_pow = np.zeros_like(max_freq)
# for j_e, j_i, k1, k2 in zip(params['J_e'], params['J_i'], params['k_ei'], params['k_ie']):
#     if not results[j_e][j_i][k1][k2].isnull().any().any():
#         _ = plot_psd(results[j_e][j_i][k1][k2], tmin=cut_off, show=False)
#         pow = plt.gca().get_lines()[-1].get_ydata()
#         freqs = plt.gca().get_lines()[-1].get_xdata()
#         r, c = np.argmin(np.abs(ei_ratio - k1/k2)), np.argmin(np.abs(io_ratio - j_i/k2))
#         max_freq[r, c] = freqs[np.argmax(pow)]
#         freq_pow[r, c] = np.max(pow)
#         plt.close(plt.gcf())
#
# cm1 = cubehelix_palette(n_colors=int(len(ei_ratio)*len(io_ratio)), as_cmap=True, start=2.5, rot=-0.1)
# cm2 = cubehelix_palette(n_colors=int(len(ei_ratio)*len(io_ratio)), as_cmap=True, start=-2.0, rot=-0.1)
# cax1 = plot_connectivity(max_freq, ax=ax[0, idx], yticklabels=list(np.round(ei_ratio, decimals=2)),
#                          xticklabels=list(np.round(io_ratio, decimals=2)), cmap=cm1)
# cax1.set_xlabel('intra/inter pcs')
# cax1.set_ylabel('exc/inh pcs')
# cax1.set_title(f'max freq (C = {C})')
# cax2 = plot_connectivity(freq_pow, ax=ax[1, idx], yticklabels=list(np.round(ei_ratio, decimals=2)),
#                          xticklabels=list(np.round(io_ratio, decimals=2)), cmap=cm2)
# cax2.set_xlabel('intra/inter pcs')
# cax2.set_ylabel('exc/inh pcs')
# cax2.set_title(f'freq power (C = {C})')
#
# plt.suptitle('EI-circuit sensitivity to population Coupling strengths (pcs)')
# plt.tight_layout(pad=2.5, rect=(0.01, 0.01, 0.99, 0.96))
# fig.savefig("/home/rgast/Documents/Studium/PhD_Leipzig/Figures/BGTCS/eic_coupling", format="svg")
# plt.show()
