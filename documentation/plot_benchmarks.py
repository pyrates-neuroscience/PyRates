import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.reload_library()
plt.style.use('ggplot')
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.labelcolor'] = 'black'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['legend.fontsize'] = 12

from pyrates.utility import plot_connectivity, create_cmap
import numpy as np

gpu_results = np.mean(np.load('gpu_benchmarks.npy'), axis=3)
cpu_results = np.mean(np.load('cpu_benchmarks.npy'), axis=3)
N = np.load('n_jrcs.npy')
p = np.load('conn_prob.npy')
diff = gpu_results[:, :, 0] / cpu_results[:, :, 0]

# create colormaps
n_colors = 20
cm_red = create_cmap('pyrates_red', as_cmap=True, n_colors=n_colors, reverse=True)
cm_green = create_cmap('pyrates_green', as_cmap=True, n_colors=n_colors, reverse=True)
div_cols = np.linspace(np.min(diff), np.max(diff), n_colors)
n_red = np.sum(div_cols > 1.)
n_blue = n_colors - n_red
cm_div = create_cmap('pyrates_red/pyrates_blue', as_cmap=True, n_colors=(n_blue, n_red),
                     pyrates_blue={'reverse': True}, pyrates_red={'reverse': False}, vmin=(0., 0.), vmax=(1.0, 1.0))

# plot results
dpi = 400
fig, axes = plt.subplots(ncols=3, figsize=(int(5000/dpi), int(2000/dpi)), dpi=400)

# plot simulation times of benchmarks run on the GPU
plot_connectivity(gpu_results[:, :, 0], ax=axes[0], yticklabels=N, xticklabels=p, cmap=cm_red,
                  cbar_kws={'label': 'T in s'})
axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation='horizontal')
axes[0].set_ylabel('number of JRCs', labelpad=15.)
#axes[0].set_xlabel('coupling density', labelpad=15.)
axes[0].set_title('A: Simulation time T', pad=20.)

# plot simulation time differene between GPU and CPU
ax = plot_connectivity(diff, ax=axes[1], yticklabels=N, xticklabels=p, cmap=cm_div,
                       cbar_kws={'label': r'$\mathbf{T_{GPU}} / \mathbf{T_{CPU}}$'})
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation='horizontal')
#axes[1].set_ylabel('number of JRCs', labelpad=15.)
axes[1].set_xlabel('coupling density', labelpad=15.)
axes[1].set_title('B: GPU vs. CPU', pad=20.)

# plot memory consumption of benchmarks run on the GPU
plot_connectivity(cpu_results[:, :, 1], ax=axes[2], yticklabels=N, xticklabels=p, cmap=cm_green,
                  cbar_kws={'label': 'M in MB'})
axes[2].set_yticklabels(axes[2].get_yticklabels(), rotation='horizontal')
#axes[2].set_ylabel('number of JRCs', labelpad=15.)
#axes[2].set_xlabel('coupling density', labelpad=15.)
axes[2].set_title('C: Peak memory M', pad=20.)

plt.tight_layout()
plt.savefig('/nobackup/spanien1/rgast/PycharmProjects/PyRates/documentation/img/Gast_2018_PyRates_benchmarks.svg',
            format='svg')
plt.show()
