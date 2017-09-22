"""
Includes functions to simulate and plot the behavior of simple neural mass models when varying certain parameters.
"""

import numpy as np
from NMMs.base import nmm_network, populations, synapses, axons
from matplotlib.pyplot import *

__author__ = "Richard Gast"
__status__ = "Development"

#############################
# parameter study functions #
#############################


def JR_parameter_study(param_names, param_values, simulation_time=1.0, step_size=5e-4):
    """
    Function that simulates JR circuit behavior for every parameter combination passed.

    :param param_names: list with name of JR parameters to alter
    :param param_values: list with parameter value lists over which to loop
    :param simulation_time: scalar, indicating the simulation time in seconds of each parameter combination
    :param step_size: simulation step-size for euler formalism

    :return: Array with final pyramidal cell states after simulation time have passed.

    """

    # JansenRit parametrization
    ###########################

    # populations
    population_labels = ['PC', 'EIN', 'IIN']
    N = len(population_labels)
    n_synapses = 2

    # synapses
    connections = np.zeros((N, N, n_synapses))

    # AMPA connections (excitatory)
    connections[:, :, 0] = [[0, 0.8 * 135, 0], [1.0 * 135, 0, 0], [0.25 * 135, 0, 0]]

    # GABA-A connections (inhibitory)
    connections[:, :, 1] = [[0, 0, 0.25 * 135], [0, 0, 0], [0, 0, 0]]

    ampa_dict = {'efficiency': 1.273 * 3e-13,     # A
                 'tau_decay': 0.006,              # s
                 'tau_rise': 0.0006,              # s
                 'conductivity_based': False}

    gaba_a_dict = {'efficiency': 1.273 * -1e-12,    # A
                   'tau_decay': 0.02,               # s
                   'tau_rise': 0.0004,              # s
                   'conductivity_based': False}

    synapse_params = [ampa_dict, gaba_a_dict]
    synaptic_kernel_length = 100                  # in time steps

    # axon
    axon_dict = {'max_firing_rate': 5.,                     # 1/s
                 'membrane_potential_threshold': -0.069,    # V
                 'sigmoid_steepness': 555.56}               # 1/V
    axon_params = [axon_dict for i in range(N)]

    # inter-population delays
    distances = np.zeros((N, N))
    velocities = float('inf')

    # initial state values
    init_states = np.zeros((N, n_synapses))

    # synaptic inputs
    mag_stim = 300.0        # 1/s
    synaptic_inputs = np.zeros((int(simulation_time/step_size), N, n_synapses))
    synaptic_inputs[:, 1, 0] = mag_stim

    # store parameters in dict
    ##########################

    parameters = {'connections': connections,
                  'population_labels': population_labels,
                  'step_size': step_size,
                  'synaptic_kernel_length': synaptic_kernel_length,
                  'distances': distances,
                  'velocities': velocities,
                  'synapse_params': synapse_params,
                  'axon_params': axon_params,
                  'init_states': init_states}

    # alter requested parameters and run simulations for each passed parameter value combination
    ############################################################################################

    N = len(param_values[0])
    final_pc_states = np.zeros(N)

    for i in range(N):

        # alter parameters
        for j, p in enumerate(param_names):
            parameters[p] = param_values[j][i]

        # initialize network
        nmm = nmm_network.NeuralMassModel(connections=parameters['connections'],
                                          population_labels=parameters['population_labels'],
                                          step_size=parameters['step_size'],
                                          synaptic_kernel_length=parameters['synaptic_kernel_length'],
                                          distances=parameters['distances'],
                                          velocities=parameters['velocities'],
                                          synapse_params=parameters['synapse_params'],
                                          axon_params=parameters['axon_params'],
                                          init_states=parameters['init_states'])

        # run simulation

        nmm.run(synaptic_inputs=parameters['synaptic_inputs'],
                simulation_time=simulation_time,
                cutoff_time=0.,
                store_step=1)

        final_pc_states[i] = nmm.neural_mass_states[-1][0]

    return final_pc_states


def JR_AMPAEfficiency_And_InputStrength_Varied(ampa_efficiencies, synaptic_input_strengths, simulation_time=3.0,
                                               step_size=5e-4, grid_simulation=True):
    """

    :param ampa_efficiencies: vector of scalars, indicating the ampa synapse efficiencies to sweep over [unit = A].
    :param synaptic_input_strengths: vector of scalars, indicating the synaptic input strengths delivered to the
           excitatory interneurons to sweep over [unit = Hz].
    :param simulation_time: scalar, indicating the simulation time for each parameter combination [unit = s]
           (default = 3.0).
    :param step_size: scalar, indicating the simulation step-size [unit = s] (default = 5e-4).
    :param grid_simulation: If True, all combination of ampa efficiencies and synaptic input strengths will be
           simulated. Else, only paired combinations will be considered (default = True).

    :return: Vector with final pyramidal cell membrane potentials of each simulation.

    """

    # set parameters
    ################

    gaba_a_dict = {'efficiency': 1.273 * -1e-12,  # A
                   'tau_decay': 0.02,  # s
                   'tau_rise': 0.0004,  # s
                   'conductivity_based': False}

    synapse_params = list()
    synaptic_inputs = list()

    if grid_simulation:

        for i in range(len(ampa_efficiencies)):

            ampa_dict = {'efficiency': float(ampa_efficiencies[i]),  # A
                         'tau_decay': 0.006,  # s
                         'tau_rise': 0.0006,  # s
                         'conductivity_based': False}

            for j in range(len(synaptic_input_strengths)):

                synaptic_inputs_tmp = np.zeros((int(simulation_time / step_size), 3, 2))
                synaptic_inputs_tmp[:, 1, 0] = synaptic_input_strengths[j]

                synapse_params.append([ampa_dict, gaba_a_dict])
                synaptic_inputs.append(synaptic_inputs_tmp)

    else:

        for i in range(len(ampa_efficiencies)):

            ampa_dict = {'efficiency': float(ampa_efficiencies[i]),  # A
                         'tau_decay': 0.006,  # s
                         'tau_rise': 0.0006,  # s
                         'conductivity_based': False}

            synapse_params.append([ampa_dict, gaba_a_dict])

            synaptic_inputs_tmp = np.zeros((int(simulation_time / step_size), 3, 2))
            synaptic_inputs_tmp[:, 1, 0] = synaptic_input_strengths[i]

            synaptic_inputs.append(synaptic_inputs_tmp)

    # start parameter study
    #######################

    results = JR_parameter_study(param_names=['synapse_params', 'synaptic_inputs'],
                                 param_values=[synapse_params, synaptic_inputs],
                                 simulation_time=simulation_time)

    if grid_simulation:

        # re-arrange results from vector to array
        results_final = np.zeros((len(ampa_efficiencies), len(synaptic_input_strengths)))
        for i in range(len(ampa_efficiencies)):
            results_final[i, :] = results[i*len(synaptic_input_strengths):(i+1)*len(synaptic_input_strengths)]
        results = results_final

    return results

######################
# plotting functions #
######################


def parameter_plot_2D(results, param_names, param_vals):
    """

    :param results: Array with membrane potentials
    :param param_names: List with 2 parameter names (character strings).
    :param param_vals: List with 2 vectors, representing the parameter values used to receive the results matrix.

    :return: figure handle
    """

    fig = figure()

    # plot results in matrix
    ax = matshow(results, fignum=0)

    # cosmetics
    colorbar()
    title('PC membrane potential [mV]')
    gca().xaxis.tick_bottom()
    xlabel(param_names[0])
    ylabel(param_names[1])
    xticks(np.arange(len(param_vals[0])), param_vals[0])
    yticks(np.arange(len(param_vals[1])), param_vals[1])

    fig.show()

    return fig

#################################
# apply above defined functions #
#################################

ampa_efficiencies = np.linspace(0.1, 1.0, 10) * 1.273 * 3e-13
synaptic_input_strengths = np.array([50, 100, 150, 200, 250, 300, 350, 400])
results = JR_AMPAEfficiency_And_InputStrength_Varied(ampa_efficiencies=ampa_efficiencies,
                                                     synaptic_input_strengths=synaptic_input_strengths,
                                                     grid_simulation=True)

fig = parameter_plot_2D(results=results.T,
                        param_names=['AMPA synapse efficiency [1e-13 A]', 'Synaptic input strength [Hz]'],
                        param_vals=[np.round(ampa_efficiencies*1e13, decimals=2), synaptic_input_strengths])

fig.show()
