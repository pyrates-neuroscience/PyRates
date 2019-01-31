# pyrates imports
from pyrates.ir.circuit import CircuitIR
from pyrates.backend import ComputeGraph
from pyrates.utility import plot_connectivity, create_cmap
from pyrates.frontend import CircuitTemplate

# additional imports
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.reload_library()
plt.style.use('ggplot')
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 24
mpl.rcParams['axes.labelcolor'] = 'black'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['legend.fontsize'] = 20
from copy import deepcopy

# define parameters and functions
dt = 1e-3                                       # integration step-size of the forward euler solver in s
T = 1.0                                         # simulation time in s
c = 20.                                         # global connection strength scaling
N =[1, 10, 100][::-1]                           # network sizes, each of which will be run a benchmark for
p = [0.81, 0.27, 0.09, 0.03, 0.01][::-1]        # global coupling probabilities, each of which will be run a benchmark for


def benchmark(Ns, Ps, T, dt, init_kwargs, run_kwargs):
    """Function that will run a benchmark simulation for each combination of N and P.
    Each benchmark simulation simulates the behavior of a neural population network, where the Jansen-Rit model is used for each of the N nodes and
    connections are drawn randomly, such that on overall coupling density of P is established.

    Parameters
    ----------
    Ns
        Vector with network sizes.
    Ps
        Vector with coupling densities.
    T
        Overall simulation time.
    dt
        Integration step-size.
    init_kwargs
        Additional key-word arguments for the model initialization.
    run_kwargs
        Additional key-word arguments for running the simulation.

    Returns
    -------
    tuple
        Simulation times, peak memory consumptions

    """

    times = np.zeros((len(Ns), len(Ps)))
    peak_mem = np.zeros_like(times)

    for i, n in enumerate(Ns):
        for j, p in enumerate(Ps):

            print(f'Running benchmark for n = {n} and p = {p}.')
            print("Setting up the network in PyRates...")

            # define inter-JRC connectivity
            C = np.random.uniform(size=(n, n))
            C[C > p] = 0.
            conns = DataFrame(C, columns=[f'jrc_{idx}/PC.0/PRO.0/m_out' for idx in range(n)])
            conns.index = [f'jrc_{idx}/PC.0/RPO_e_pc.0/m_in' for idx in range(n)]

            # define input
            inp = 220 + np.random.randn(int(T / dt), n) * 22.

            # set up template
            template = CircuitTemplate.from_yaml("pyrates.examples.jansen_rit.simple_jr.JRC")

            # set up intermediate representation
            circuits = {}
            for idx in range(n):
                circuits[f'jrc_{idx}'] = deepcopy(template)
            circuit = CircuitIR.from_circuits(label='net', circuits=circuits, connectivity=conns)

            # set up compute graph
            net = ComputeGraph(circuit, dt=dt, **init_kwargs)

            print("Starting the benchmark simulation...")

            # run simulations
            _, t, m = net.run(T, inputs={('PC', 'RPO_e_pc.0', 'u'): inp}, outputs={'V': ('PC', 'PRO.0', 'PSP')},
                              verbose=False, **run_kwargs)
            times[i, j] = t
            peak_mem[i, j] = m

            print("Finished!")

    return times, peak_mem


# simulate benchmarks
results = np.zeros((len(N), len(p), 3))                       # array in which results will be stored

t, m = benchmark(N, p, T, dt,
                 init_kwargs={'vectorization': 'full'},
                 run_kwargs={'profile': 'mt'})                # runs benchmark function on GPU
results[:, :, 0] = t
results[:, :, 1] = m

t, _ = benchmark(N, p, T, dt,                                 # runs benchmark function on CPU
                 init_kwargs={'vectorization': 'full'},
                 run_kwargs={'profile': 't'})
results[:, :, 3] = t

# create colormaps
n_colors = 16
cm_red = create_cmap('pyrates_red', as_cmap=True, n=n_colors)
cm_green = create_cmap('pyrates_green', as_cmap=True, n=n_colors)
cm_div = create_cmap('pyrates_blue/pyrates_yellow', as_cmap=True, n=n_colors, reverse=True)

# plot results
fig, axes = plt.subplots(ncols=3, figsize=(20, 7))

# plot simulation times of benchmarks run on the GPU
plot_connectivity(results[:, :, 0], ax=axes[0], yticklabels=N, xticklabels=p, cmap=cm_red)
axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation='horizontal')
axes[0].set_ylabel('number of JRCs', labelpad=15.)
axes[0].set_xlabel('coupling density', labelpad=15.)
axes[0].set_title('Simulation time T in s', pad=20.)

# plot memory consumption of benchmarks run on the GPU
plot_connectivity(results[:, :, 1], ax=axes[1], yticklabels=N, xticklabels=p, cmap=cm_green)
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation='horizontal')
axes[1].set_ylabel('number of JRCs', labelpad=15.)
axes[1].set_xlabel('coupling density', labelpad=15.)
axes[1].set_title('Peak memory in MB', pad=20.)

# plot simulation time differene between GPU and CPU
plot_connectivity(results[:, :, 2] - results[:, :, 0], ax=axes[2], yticklabels=N, xticklabels=p, cmap=cm_div)
axes[2].set_yticklabels(axes[2].get_yticklabels(), rotation='horizontal')
axes[2].set_ylabel('number of JRCs', labelppad=15.)
axes[2].set_xlabel('coupling density', labelpad=15.)
axes[2].set_title(r'$\mathbf{T_{CPU}} - \mathbf{T_{CPU}}$', pad=20.)

plt.tight_layout()
#plt.savefig('/img/Gast_2018_PyRates_benchmarks.svg', format='svg')
