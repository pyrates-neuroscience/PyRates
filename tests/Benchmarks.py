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

import numpy as np
import time
from memory_profiler import memory_usage
from scipy.io import loadmat

from core.network import NeuralMassModel, JansenRitCircuit

__author__ = "Richard Gast"
__status__ = "Development"

#######################
# benchmark functions #
#######################


def run_JR_circuit_benchmark(simulation_time=60.0, step_size=1e-4, param_names=None, param_values=None,
                             synaptic_inputs=None, verbose=False, variable_step_size=False):
    """
    Runs a benchmark on a single Jansen-Rit type microcircuit (3 interconnected neural populations).

    :param simulation_time: length of the simulation [unit = s] (default = 60.0).
    :param step_size: simulation step-size [unit = s] (default = 1e-4).
    :param param_names: list with name of JR parameters to alter (see JR_parameter_study function).
    :param param_values: list with parameter values (see JR_parameter_study function).
    :param synaptic_inputs: input fed to the microcircuit (length must be simulation_time/step_size).
    :param verbose: If true, simulation progress will be displayed.
    :param variable_step_size: If true, variable step solver will be used.

    :return: simulation length in seconds (real-time).
    """

    #############################
    # set simulation parameters #
    #############################

    if synaptic_inputs is None:

        # synaptic inputs
        mu_stim = 200.0
        std_stim = 20.0
        synaptic_inputs = np.zeros((int(simulation_time / step_size), 3, 2))
        synaptic_inputs[:, 0, 0] = std_stim * np.random.random(synaptic_inputs.shape[0]) + mu_stim

    #########################
    # initialize JR circuit #
    #########################

    nmm = JansenRitCircuit(step_size=step_size)

    if param_names:

        for i, p in enumerate(param_names):

            setattr(nmm, p, param_values[i])

    #####################
    # perform benchmark #
    #####################

    print('Starting simulation...')

    start_time = time.clock()

    nmm.run(simulation_time=simulation_time,
            synaptic_inputs=synaptic_inputs,
            verbose=verbose,
            variable_step_size=variable_step_size)

    end_time = time.clock()

    simulation_duration = end_time - start_time
    print("%.2f" % simulation_time, 's simulation of Jansen-Rit circuit finished after ',
          "%.2f" % simulation_duration, ' s.')

    return simulation_duration


def run_JR_network_benchmark(simulation_time=60.0, step_size=1e-4, N=33, C=None, connectivity_scaling=100.0, D=True,
                             velocity=1.0, synaptic_input=None, verbose=False, variable_step_size=False):
    """
    Runs benchmark for a number of JR circuits connected in a network.

    :param simulation_time:
    :param step_size:
    :param N:
    :param C:
    :param connectivity_scaling:
    :param D:
    :param velocity:
    :param synaptic_input:
    :param verbose:
    :param variable_step_size:

    :return: simulation duration [unit = s]

    """

    #############################
    # set simulation parameters #
    #############################

    # connectivity matrix
    if C is None:

        # load connectivity matrix
        C_tmp = loadmat('SC')['SC']
        C_tmp *= connectivity_scaling

        # create full connectivity matrix
        n_pops = 3
        C = np.zeros([N*n_pops, N*n_pops, 2])
        for i in range(N):
            for j in range(N):
                if i == j:
                    C[i*n_pops:i*n_pops+n_pops, j*n_pops:j*n_pops+n_pops, 0] = \
                        [[0, 0.8 * 135, 0], [1.0 * 135, 0, 0], [0.25 * 135, 0, 0]]
                    C[i*n_pops:i*n_pops+n_pops, j*n_pops:j*n_pops+n_pops, 1] = \
                        [[0, 0, 0.25 * 135], [0, 0, 0], [0, 0, 0]]
                C[i * n_pops, j * n_pops, 0] = C_tmp[i, j]

    else:

        C *= connectivity_scaling

    # network delays
    if D:

        D_tmp = loadmat('D')['D']

        D = np.zeros([N*n_pops, N*n_pops])
        for i in range(N):
            for j in range(N):
                D[i*n_pops:i*n_pops+n_pops, j*n_pops:j*n_pops+n_pops] = D_tmp[i, j]

    else:

        D = None

    # network input
    if synaptic_input is None:

        synaptic_input = np.zeros([int(np.ceil(simulation_time/step_size)), N*n_pops, 2])
        idx_pcs = np.mod(np.arange(N*n_pops), n_pops) == 0
        idx_eins = np.mod(np.arange(N*n_pops), n_pops) == 1
        synaptic_input[:, idx_pcs, 0] = np.random.randn(int(np.ceil(simulation_time/step_size)), N) + 100.0
        synaptic_input[:, idx_eins, 0] = 22 * np.random.randn(int(np.ceil(simulation_time/step_size)), N) + 200.0

    # populations
    populations = list()
    population_types = ['JansenRitPyramidalCells', 'JansenRitExcitatoryInterneurons', 'JansenRitInhibitoryInterneurons']
    for i in range(N*n_pops):
        populations.append(population_types[np.mod(i, n_pops)])

    ################
    # set up model #
    ################

    nmm = NeuralMassModel(connections=C,
                          population_types=populations,
                          distances=D,
                          velocities=velocity,
                          step_size=step_size)

    #####################
    # perform benchmark #
    #####################

    print('Starting simulation...')

    start_time = time.clock()

    nmm.run(synaptic_inputs=synaptic_input,
            simulation_time=simulation_time,
            verbose=verbose,
            variable_step_size=variable_step_size)

    end_time = time.clock()

    simulation_duration = end_time - start_time

    print("%.2f" % simulation_time, 's simulation of Jansen-Rit network with ', N, 'populations finished after ',
          "%.2f" % simulation_duration, ' s.')

    return simulation_duration

######################
# perform benchmarks #
######################

# parameters
simulation_duration = 1.0
step_size = 1e-4
verbose = True
variable_step_size = False
D = False
velocity = 2.0
connectivity_scaling = 100.0

# single JR circuit
# sim_dur_JR_circuit = run_JR_circuit_benchmark(simulation_time=simulation_duration,
#                                               step_size=step_size,
#                                               verbose=verbose,
#                                               variable_step_size=variable_step_size)

# JR network (33 connected JR circuits)
sim_dur_JR_network = run_JR_network_benchmark(simulation_time=simulation_duration,
                                              step_size=step_size,
                                              D=D,
                                              velocity=velocity,
                                              connectivity_scaling=connectivity_scaling,
                                              verbose=verbose,
                                              variable_step_size=variable_step_size)

################
# memory usage #
################

# single JR circuit
#mem_use_JR_circuit = memory_usage((run_JR_circuit_benchmark, (simulation_duration, step_size)))
#print("%.2f" % simulation_duration, 's simulation of Jansen-Rit circuit used ',
#      "%.2f" % (np.sum(mem_use_JR_circuit) * 1e-2), ' MB RAM.')

# JR network (33 connected JR circuits)
#mem_use_JR_network = memory_usage((run_JR_network_benchmark, (simulation_duration, step_size)))
#print("%.2f" % simulation_duration, 's simulation of network with 33 JR circuits used ',
#      "%.2f" % (np.sum(mem_use_JR_network) * 1e-2), ' MB RAM.')