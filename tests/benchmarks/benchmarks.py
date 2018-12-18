"""
Includes various benchmarks for the NMM framework.

Profiling:
##########
http://softwaretester.info/python-profiling-with-pycharm-community-edition/

# install packages
pip install snakeviz
pip install cprofilev

# Change “Benchmark Configuration”:
Interpreter options: -B -m cProfile -o profile.prof

# show results (terminal):
> snakeviz profile.prof

"""

# pyrates imports
from pyrates.ir.circuit import CircuitIR
from pyrates.backend import ComputeGraph
from pyrates.utility import plot_connectivity
from pyrates.frontend import CircuitTemplate

# additional imports
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from seaborn import cubehelix_palette
from copy import deepcopy
import time

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"

#########################
# Jansen-Rit benchmarks #
#########################


def benchmark(Ns, Ps, T, dt, init_kwargs, run_kwargs):
    """

    Parameters
    ----------
    Ns
    Ps
    T
    dt
    init_kwargs
    run_kwargs

    Returns
    -------

    """

    times = np.zeros((len(Ns), len(Ps)))
    peak_mem = np.zeros_like(times)

    for i, n in enumerate(Ns):
        for j, p in enumerate(Ps):

            print(f'Running benchmark for n = {n} and p = {p}.')

            # define inter-JRC connectivity
            print('connectivity setup')
            t0 = time.time()
            C = np.random.uniform(size=(n, n))
            C[C > p] = 0.
            conns = DataFrame(C, columns=[f'jrc_{idx}/PC.0/PRO.0/m_out' for idx in range(n)])
            conns.index = [f'jrc_{idx}/PC.0/RPO_e_pc.0/m_in' for idx in range(n)]
            print(f'...finished after {time.time() - t0} seconds.')

            # define input
            inp = 220 + np.random.randn(int(T / dt), n) * 22.

            # set up template
            print('Frontend template setup')
            t0 = time.time()
            template = CircuitTemplate.from_yaml("pyrates.examples.jansen_rit.simple_jr.JRC")
            print(f'...finished after {time.time() - t0} seconds.')

            # set up intermediate representation
            print('IR setup')
            t0 = time.time()
            circuits = {}
            for idx in range(n):
                circuits[f'jrc_{idx}'] = deepcopy(template)
            circuit = CircuitIR.from_circuits(label='net', circuits=circuits, connectivity=conns)
            print(f'...finished after {time.time() - t0} seconds.')

            # set up compute graph
            print('Compute graph setup')
            t0 = time.time()
            net = ComputeGraph(circuit, dt=dt, **init_kwargs)
            print(f'...finished after {time.time() - t0} seconds.')

            # run simulations
            _, t, m = net.run(T, inputs={('PC', 'RPO_e_pc.0', 'u'): inp}, outputs={'V': ('PC', 'PRO.0', 'PSP')},
                              verbose=False, **run_kwargs)
            times[i, j] = t
            peak_mem[i, j] = m

    return times, peak_mem


# parameter definitions
dt = 1e-3
T = 1.0
n_jrcs = [1, 10, 100, 1000, 2000]
p_conn = [0.81, 0.27, 0.09, 0.03, 0.01]
sim_times = np.zeros((len(n_jrcs), len(p_conn), 4))

# pyrates simulation
t, _ = benchmark(n_jrcs, p_conn, T, dt, init_kwargs={'vectorization': 'nodes'}, run_kwargs={'profile': 't'})
sim_times[:, :, 0] = t
t, _ = benchmark(n_jrcs, p_conn, T, dt, init_kwargs={'vectorization': 'full'}, run_kwargs={'profile': 't'})
sim_times[:, :, 1] = t
_, m = benchmark(n_jrcs, p_conn, T, dt, init_kwargs={'vectorization': 'nodes'}, run_kwargs={'profile': 'm'})
sim_times[:, :, 2] = m
_, m = benchmark(n_jrcs, p_conn, T, dt, init_kwargs={'vectorization': 'full'}, run_kwargs={'profile': 'm'})
sim_times[:, :, 3] = m

# plotting
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15, 15))
cm1 = cubehelix_palette(n_colors=int(len(n_jrcs)**2), as_cmap=True, start=2.5, rot=-0.1)
cm2 = cubehelix_palette(n_colors=int(len(n_jrcs)**2), as_cmap=True, start=-2.0, rot=-0.1)
axes[0, 0].set_aspect(1.0)
axes[0, 1].set_aspect(1.0)
axes[1, 0].set_aspect(1.0)
axes[1, 1].set_aspect(1.0)
plot_connectivity(sim_times[:, :, 0], ax=axes[0, 0], yticklabels=n_jrcs, xticklabels=p_conn, cmap=cm1)
plot_connectivity(sim_times[:, :, 1], ax=axes[0, 1], yticklabels=n_jrcs, xticklabels=p_conn, cmap=cm1)
plot_connectivity(sim_times[:, :, 2], ax=axes[1, 0], yticklabels=n_jrcs, xticklabels=p_conn, cmap=cm2)
plot_connectivity(sim_times[:, :, 3], ax=axes[1, 1], yticklabels=n_jrcs, xticklabels=p_conn, cmap=cm2)
plt.savefig('PyRates_benchmarks_1.svg', format='svg')
plt.show()

#######################
# benchmark functions #
#######################


# def run_JR_circuit_benchmark(simulation_time=1.0, step_size=1 / 2048, param_names=None, param_values=None,
#                              synaptic_inputs=None, synaptic_input_pops=None, synaptic_input_syns=None,
#                              verbose=False):
#     """
#     Runs a benchmark on a single Jansen-Rit type microcircuit (3 interconnected neural populations).
#
#     :param simulation_time: length of the simulation [unit = s] (default = 60.0).
#     :param step_size: simulation update-size [unit = s] (default = 1e-4).
#     :param param_names: list with name of JR parameters to alter (see JR_parameter_study function).
#     :param param_values: list with parameter values (see JR_parameter_study function).
#     :param synaptic_inputs: input fed to the microcircuit (length must be simulation_time/step_size).
#     :param verbose: If true, simulation progress will be displayed.
#     :param variable_step_size: If true, variable update solver will be used.
#
#     :return: simulation length in seconds (real-time).
#     """
#
#     #############################
#     # set simulation parameters #
#     #############################
#
#     if synaptic_inputs is None:
#         # synaptic inputs
#         mu_stim = 200.0
#         std_stim = 20.0
#         synaptic_inputs = np.zeros((int(simulation_time / step_size), 2))
#         synaptic_inputs[:, 0] = std_stim * np.random.random(synaptic_inputs.shape[0]) + mu_stim
#         synaptic_inputs[:, 1] = 0 * np.random.random(synaptic_inputs.shape[0]) + mu_stim / 2.
#         synaptic_input_pops = ['JR_EINs', 'JR_PCs']
#         synaptic_input_syns = ['excitatory', 'excitatory']
#
#     #########################
#     # initialize JR circuit #
#     #########################
#
#     nmm = JansenRitCircuit(step_size=step_size)
#
#     if param_names:
#
#         for i, p in enumerate(param_names):
#             setattr(nmm, p, param_values[i])
#
#     #####################
#     # perform benchmark #
#     #####################
#
#     print('Starting simulation of single Jansen Rit Circuit...')
#
#     start_time = time.clock()
#
#     nmm.run(simulation_time=simulation_time,
#             synaptic_inputs=synaptic_inputs,
#             synaptic_input_pops=synaptic_input_pops,
#             synaptic_input_syns=synaptic_input_syns,
#             verbose=verbose)
#
#     end_time = time.clock()
#
#     simulation_duration = end_time - start_time
#     print("%.2f" % simulation_time, 's simulation of Jansen-Rit circuit finished after ',
#           "%.2f" % simulation_duration, ' s.')
#
#     return simulation_duration
#
#
# def run_JR_network_benchmark(simulation_time=60.0, step_size=1e-4, N=33, C=None, connectivity_scaling=100.0, D=True,
#                              velocity=1.0, synaptic_input=None, synaptic_input_pops=None,
#                              synaptic_input_syns=None, verbose=False):
#     """
#     Runs benchmark for a number of JR circuits connected in a backend.
#
#     :param simulation_time:
#     :param step_size:
#     :param N:
#     :param C:
#     :param connectivity_scaling:
#     :param D:
#     :param velocity:
#     :param synaptic_input:
#     :param verbose:
#
#     :return: simulation duration [unit = s]
#
#     """
#
#     #############################
#     # set simulation parameters #
#     #############################
#
#     # number of JRCs
#     n_circuits = 33
#
#     # connectivity matrix
#     if C is None:
#
#         # load connectivity matrix
#         C = loadmat('../resources/SC')['SC']
#         C *= connectivity_scaling
#
#         # hack because, currently a third dimension is expected
#         C = np.reshape(C, (C.shape[0], C.shape[1], 1))
#         C = np.concatenate((C, np.zeros_like(C)), axis=2)
#
#         # create full connectivity matrix
#         # n_pops = 3
#         # C = np.zeros([N * n_pops, N * n_pops, 2])
#         # for i in range(N):
#         #     for j in range(N):
#         #         if i == j:
#         #             C[i * n_pops:i * n_pops + n_pops, j * n_pops:j * n_pops + n_pops, 0] = \
#         #                 [[0, 0.8 * 135, 0], [1.0 * 135, 0, 0], [0.25 * 135, 0, 0]]
#         #             C[i * n_pops:i * n_pops + n_pops, j * n_pops:j * n_pops + n_pops, 1] = \
#         #                 [[0, 0, 0.25 * 135], [0, 0, 0], [0, 0, 0]]
#         #         C[i * n_pops, j * n_pops, 0] = C_tmp[i, j]
#
#     else:
#
#         C *= connectivity_scaling
#
#     # backend delays
#     if D:
#
#         D = loadmat('../resources/D')['D']
#         D /= velocity * 1e3
#
#         # D = np.zeros([N * n_pops, N * n_pops])
#         # for i in range(N):
#         #     for j in range(N):
#         #         D[i * n_pops:i * n_pops + n_pops, j * n_pops:j * n_pops + n_pops] = D_tmp[i, j]
#
#     else:
#
#         D = np.zeros((n_circuits, n_circuits))
#
#     # backend input
#     if synaptic_input is None:
#         simulation_steps = int(np.ceil(simulation_time / step_size))
#         synaptic_input = np.random.uniform(120, 320, (simulation_steps, n_circuits))
#         synaptic_input_pops = ['Circ' + str(i) + '_JR_PCs' for i in range(n_circuits)]
#         synaptic_input_syns = ['excitatory' for _ in range(n_circuits)]
#
#     # circuits
#     circuits = [JansenRitCircuit(step_size=step_size) for _ in range(n_circuits)]
#
#     ################
#     # set up backend #
#     ################
#
#     nmm = CircuitFromCircuit(circuits=circuits,
#                              connectivity=C,
#                              delays=D,
#                              synapse_keys=['excitatory', 'inhibitory']
#                              )
#
#     #####################
#     # perform benchmark #
#     #####################
#
#     print('Starting simulation of 33 connected Jansen-Rit circuits...')
#
#     start_time = time.clock()
#
#     nmm.run(synaptic_inputs=synaptic_input,
#             synaptic_input_pops=synaptic_input_pops,
#             synaptic_input_syns=synaptic_input_syns,
#             simulation_time=simulation_time,
#             verbose=verbose)
#
#     end_time = time.clock()
#
#     simulation_duration = end_time - start_time
#
#     print("%.2f" % simulation_time, 's simulation of Jansen-Rit backend with ', N, 'populations finished after ',
#           "%.2f" % simulation_duration, ' s.')
#
#     return simulation_duration


# if __name__ == "__main__":
#
#     import sys
#
#     class CallError(Exception):
#         pass
#     try:
#         arg = sys.argv[1]
#     except IndexError:
#         arg = "both"
#     if arg == '1':
#         run_first = True
#         run_second = False
#     elif arg == '2':
#         run_second = True
#         run_first = False
#     elif arg == "both":
#         run_first = run_second = True
#     else:
#         run_first = run_second = False
#
#     ######################
#     # perform benchmarks #
#     ######################
#
#     # parameters
#     simulation_duration = 10.0
#     step_size = 5e-4
#     verbose = True
#     D = True
#     velocity = 2.0
#     connectivity_scaling = 50.0
#
#     if run_first:
#         # single JR circuit
#         sim_dur_JR_circuit = run_JR_circuit_benchmark(simulation_time=simulation_duration,
#                                                       step_size=step_size,
#                                                       verbose=verbose)
#
#     if run_second:
#         # JR backend (33 connected JR circuits)
#         sim_dur_JR_network = run_JR_network_benchmark(simulation_time=simulation_duration,
#                                                       step_size=step_size,
#                                                       D=D,
#                                                       velocity=velocity,
#                                                       connectivity_scaling=connectivity_scaling,
#                                                       verbose=verbose)

    ################
    # memory usage #
    ################

    # single JR circuit
    # mem_use_JR_circuit = memory_usage((run_JR_circuit_benchmark, (simulation_duration, step_size)))
    # print("%.2f" % simulation_duration, 's simulation of Jansen-Rit circuit used ',
    #      "%.2f" % (np.sum(mem_use_JR_circuit) * 1e-2), ' MB RAM.')

    # JR backend (33 connected JR circuits)
    # mem_use_JR_network = memory_usage((run_JR_network_benchmark, (simulation_duration, step_size)))
    # print("%.2f" % simulation_duration, 's simulation of backend with 33 JR circuits used ',
    #      "%.2f" % (np.sum(mem_use_JR_network) * 1e-2), ' MB RAM.')
