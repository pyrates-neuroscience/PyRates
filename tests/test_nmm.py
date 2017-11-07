"""
Includes unit tests for all classes included in NMMs/base.
"""

import pickle
import unittest
import numpy as np

from core.axon import Axon, JansenRitAxon
from core.network import NeuralMassModel
from core.population import Population
from core.synapse import AMPACurrentSynapse, GABAACurrentSynapse
from core.synapse import Synapse

__author__ = "Richard Gast & Konstantin Weise"
__status__ = "Test"


#####################
# support functions #
#####################


def NMRSE(x, y):
    """
    Calculates the normalized root mean squared error of two vectors of equal length.

    :param x: vector 1
    :param y: vector 2

    :return: NMRSE

    """

    max_val = np.max((np.max(x, axis=0), np.max(y, axis=0)))
    min_val = np.min((np.min(x, axis=0), np.min(y, axis=0)))

    diff = x - y

    return np.sqrt(np.sum(diff ** 2, axis=0)) / (max_val - min_val)


##############
# unit tests #
##############


class TestNMMs(unittest.TestCase):
    """
    Test class that includes unit tests for all components of NMM network.
    """

    def test_0_JR_axon(self):
        """
        Tests whether axon with standard parametrization from Jansen & Rit (1995) shows expected behavior to input in
        form of various membrane potentials.
        """

        # axon parameters
        #################

        max_firing_rate = 5.  # unit = 1
        membrane_potential_threshold = -0.069  # unit = V
        sigmoid_steepness = 555.56  # unit = 1/V

        # initialize axon
        #################

        axon = Axon(max_firing_rate=max_firing_rate,
                    membrane_potential_threshold=membrane_potential_threshold,
                    sigmoid_steepness=sigmoid_steepness)

        # define inputs (unit = V)
        ##########################

        membrane_potential_1 = membrane_potential_threshold
        membrane_potential_2 = membrane_potential_threshold - 0.01
        membrane_potential_3 = membrane_potential_threshold + 0.01
        membrane_potential_4 = membrane_potential_threshold - 0.1
        membrane_potential_5 = membrane_potential_threshold + 0.1

        # get firing rates
        ##################

        firing_rate_1 = axon.compute_firing_rate(membrane_potential_1)
        firing_rate_2 = axon.compute_firing_rate(membrane_potential_2)
        firing_rate_3 = axon.compute_firing_rate(membrane_potential_3)
        firing_rate_4 = axon.compute_firing_rate(membrane_potential_4)
        firing_rate_5 = axon.compute_firing_rate(membrane_potential_5)

        # perform unit tests
        #################################################################################

        print('-----------------')
        print('| Test I - Axon |')
        print('-----------------')

        print('I.1 test whether output firing rate at membrane potential threshold is indeed 0.5 scaled by the max '
              'firing rate.')
        self.assertEqual(firing_rate_1, 0.5 * max_firing_rate)
        print('I.1 done!')

        print('I.2 test whether output firing rate gets smaller for lower membrane potential and the other way around.')
        self.assertLess(firing_rate_2, firing_rate_1)
        self.assertGreater(firing_rate_3, firing_rate_1)
        print('I.2 done!')

        print('I.3 test whether equal amounts of hyperpolarization and depolarization lead to equal changes in membrane'
              ' potential.')
        self.assertAlmostEqual(np.abs(firing_rate_1 - firing_rate_2), np.abs(firing_rate_1 - firing_rate_3), places=4)
        print('I.3 done!')

        print('I.4 test whether extreme depolarization leads to almost zero firing rate.')
        self.assertAlmostEqual(firing_rate_4, 0., places=2)
        print('I.4 done!')

        print('I.5 test whether extreme hyperpolarization leads to almost max firing rate')
        self.assertAlmostEqual(firing_rate_5, max_firing_rate, places=2)
        print('I.5 done!')

    def test_1_AMPA_synapse(self):
        """
        Tests whether synapse with standard AMPA parametrization from Thomas Knoesche shows expected behavior for
        various firing rate inputs.
        """

        # synapse parameters
        ####################

        efficiency = 1.273 * 3e-13  # unit = A
        tau_decay = 0.006  # unit = s
        tau_rise = 0.0006  # unit = s
        step_size = 5.e-4  # unit = s
        synaptic_kernel_length = int(0.05 / step_size)  # unit = 1
        conductivity_based = False

        # initialize synapse
        ####################

        synapse = Synapse(efficiency=efficiency,
                          tau_decay=tau_decay,
                          tau_rise=tau_rise,
                          step_size=step_size,
                          kernel_length=synaptic_kernel_length,
                          conductivity_based=conductivity_based)

        # define firing rate inputs
        ###########################

        firing_rates_1 = np.zeros(synaptic_kernel_length)
        firing_rates_2 = np.ones(synaptic_kernel_length) * 300.0
        firing_rates_3 = np.zeros(3 * synaptic_kernel_length)
        firing_rates_3[synaptic_kernel_length:2 * synaptic_kernel_length] = 300.0

        # calculate synaptic currents
        #############################

        synaptic_current_1 = synapse.get_synaptic_current(firing_rates_1)
        synaptic_current_2 = synapse.get_synaptic_current(firing_rates_2)

        # get synaptic current at each incoming firing rate of firing_rates_3
        idx = np.arange(1, len(firing_rates_3))
        synaptic_current_3 = np.array([synapse.get_synaptic_current(firing_rates_3[0:i]) for i in idx])

        # perform unit tests
        ####################

        print('--------------------------')
        print('| Test II - AMPA Synapse |')
        print('--------------------------')

        print('II.1 test whether zero input to AMPA synapse leads to zero synaptic current.')
        self.assertEqual(synaptic_current_1, 0.)
        print('II.1 done!')

        print('II.2 test whether increased input to AMPA synapse leads to increased synaptic current.')
        self.assertGreater(synaptic_current_2, synaptic_current_1)
        print('II.2 done!')

        print('II.3 test whether synaptic current response to step-function input has a single maximum.')
        pairwise_difference = np.diff(synaptic_current_3)
        response_rise = np.where(pairwise_difference > 0.)
        response_decay = np.where(pairwise_difference < 0.)
        self.assertTrue((np.diff(response_rise) == 1).all())
        self.assertTrue((np.diff(response_decay) == 1).all())
        print('II.3 done!')

        print('')

    def test_2_GABAA_synapse(self):
        """
        Tests whether synapse with standard GABAA parametrization from Thomas Knoesche shows expected behavior for
        various firing rate inputs.
        """

        # synapse parameters
        ####################

        efficiency = 1.273 * -1e-12  # unit = A
        tau_decay = 0.02  # unit = s
        tau_rise = 0.0004  # unit = s
        step_size = 5.e-4  # unit = s
        synaptic_kernel_length = int(0.05 / step_size)  # unit = 1
        conductivity_based = False

        # initialize synapse
        ####################

        synapse = Synapse(efficiency=efficiency,
                          tau_decay=tau_decay,
                          tau_rise=tau_rise,
                          step_size=step_size,
                          kernel_length=synaptic_kernel_length,
                          conductivity_based=conductivity_based)

        # define firing rate inputs
        ###########################

        firing_rates_1 = np.zeros(synaptic_kernel_length)
        firing_rates_2 = np.ones(synaptic_kernel_length) * 300.0
        firing_rates_3 = np.zeros(3 * synaptic_kernel_length)
        firing_rates_3[synaptic_kernel_length:2 * synaptic_kernel_length] = 300.0

        # calculate synaptic currents
        #############################

        synaptic_current_1 = synapse.get_synaptic_current(firing_rates_1)
        synaptic_current_2 = synapse.get_synaptic_current(firing_rates_2)

        # get synaptic current at each incoming firing rate of firing_rates_3
        idx = np.arange(1, len(firing_rates_3))
        synaptic_current_3 = np.array([synapse.get_synaptic_current(firing_rates_3[0:i]) for i in idx])

        # perform unit tests
        ####################

        print('----------------------------')
        print('| Test III - GABAA Synapse |')
        print('----------------------------')

        print('III.1 test whether zero input to GABAA synapse leads to zero synaptic current.')
        self.assertEqual(synaptic_current_1, 0.)
        print('III.1 done!')

        print('III.2 test whether increased input to GABAA synapse leads to decreased synaptic current.')
        self.assertLess(synaptic_current_2, synaptic_current_1)
        print('III.2 done!')

        print('III.3 test whether synaptic current response to step-function input has a single minimum.')
        pairwise_difference = np.diff(synaptic_current_3)
        response_rise = np.where(pairwise_difference > 0.)
        response_decay = np.where(pairwise_difference < 0.)
        self.assertTrue((np.diff(response_rise) == 1).all())
        self.assertTrue((np.diff(response_decay) == 1).all())
        print('III.3 done!')

        print('')

    def test_3_AMPA_conductivity_synapse(self):
        """
        Tests whether conductivity based AMPA synapse shows expected behavior.
        """

        # synapse parameters
        ####################

        efficiency = 1.273 * 7.2e-10  # unit = S
        tau_decay = 0.0015  # unit = s
        tau_rise = 0.000009  # unit = s
        step_size = 5.e-4  # unit = s
        synaptic_kernel_length = int(0.05 / step_size)  # unit = 1
        conductivity_based = True

        # initialize synapse
        ####################

        synapse = Synapse(efficiency=efficiency,
                          tau_decay=tau_decay,
                          tau_rise=tau_rise,
                          step_size=step_size,
                          kernel_length=synaptic_kernel_length,
                          conductivity_based=conductivity_based)

        # define firing rate inputs
        ###########################

        firing_rates_1 = np.zeros(synaptic_kernel_length)
        firing_rates_2 = np.ones(synaptic_kernel_length) * 300.0
        firing_rates_3 = np.zeros(3 * synaptic_kernel_length)
        firing_rates_3[synaptic_kernel_length:2 * synaptic_kernel_length] = 300.0

        # calculate synaptic currents
        #############################

        synaptic_current_1 = synapse.get_synaptic_current(firing_rates_1)
        synaptic_current_2 = synapse.get_synaptic_current(firing_rates_2)

        # get synaptic current at each incoming firing rate of firing_rates_3
        idx = np.arange(1, len(firing_rates_3))
        synaptic_current_3 = np.array([synapse.get_synaptic_current(firing_rates_3[0:i]) for i in idx])

        # perform unit tests
        ####################

        print('--------------------------------------')
        print('| Test IV - AMPA Conductance Synapse |')
        print('--------------------------------------')

        print('IV.1 test whether zero input to AMPA conductance synapse leads to zero synaptic current.')
        self.assertEqual(synaptic_current_1, 0.)
        print('IV.1 done!')

        print('IV.2 test whether increased input to AMPA conductance synapse leads to increased synaptic current.')
        self.assertGreater(synaptic_current_2, synaptic_current_1)
        print('IV.2 done!')

        print('IV.3 test whether synaptic current response to step-function input has a single minimum.')
        pairwise_difference = np.diff(synaptic_current_3)
        response_rise = np.where(pairwise_difference > 0.)
        response_decay = np.where(pairwise_difference < 0.)
        self.assertTrue((np.diff(response_rise) == 1).all())
        self.assertTrue((np.diff(response_decay) == 1).all())
        print('IV.3 done!')

    def test_4_population_init(self):
        """
        Tests whether synapses and axon of initialized population show expected behavior.
        """

        # population parameters
        #######################

        synapse_types = ['AMPA_current', 'GABAA_current']
        axon = 'JansenRit'
        init_state = (-0.07, 0.5)
        step_size = 5.e-4
        synaptic_kernel_length = int(0.05 / step_size)
        tau_leak = 0.016
        resting_potential = -0.07

        # initialize population, synapses and axon
        pop = Population(synapses=synapse_types,
                         axon=axon,
                         init_state=init_state,
                         step_size=step_size,
                         synaptic_kernel_length=synaptic_kernel_length,
                         tau_leak=tau_leak,
                         resting_potential=resting_potential)
        syn1 = AMPACurrentSynapse(step_size=step_size,
                                  kernel_length=synaptic_kernel_length)
        syn2 = GABAACurrentSynapse(step_size=step_size,
                                   kernel_length=synaptic_kernel_length)
        axon = JansenRitAxon()

        # define firing rate input and membrane potential
        #################################################

        firing_rate = np.zeros(synaptic_kernel_length) + 300.0
        membrane_potential = -0.06

        # calculate population, synapse and axon response
        #################################################

        pop_syn1_response = pop.synapses[0].get_synaptic_current(firing_rate)
        pop_syn2_response = pop.synapses[1].get_synaptic_current(firing_rate)
        syn1_response = syn1.get_synaptic_current(firing_rate)
        syn2_response = syn2.get_synaptic_current(firing_rate)

        pop_ax_response = pop.axon.compute_firing_rate(membrane_potential)
        ax_reponse = axon.compute_firing_rate(membrane_potential)

        # perform unit tests
        ####################

        print('----------------------------')
        print('| Test V - Population Init |')
        print('----------------------------')

        print('V.1 test whether population synapses show expected response to firing rate input')
        self.assertEqual(pop_syn1_response, syn1_response)
        self.assertEqual(pop_syn2_response, syn2_response)
        print('V.1 done!')

        print('V.2 test whether population axon shows expected response to membrane potential input')
        self.assertEqual(pop_ax_response, ax_reponse)
        print('V.2 done!')

    def test_5_population_dynamics(self):
        """
        Tests whether population develops as expected over time given some input.
        """

        # set population parameters
        ###########################

        synapse_types = ['AMPA_current', 'GABAA_current']
        axon = 'Knoesche'
        step_size = 5e-4  # unit = s
        synaptic_kernel_length = int(0.05 / step_size)  # unit = 1
        tau_leak = 0.016  # unit = s
        resting_potential = -0.075  # unit = V
        membrane_capacitance = 1e-12  # unit = q/V
        init_state = (resting_potential, 0)  # unit = (V, 1/s)

        # define population input
        #########################

        synaptic_inputs = np.zeros((4, 2, 5 * synaptic_kernel_length))
        synaptic_inputs[1, 0, :] = 300.0
        synaptic_inputs[2, 1, :] = 300.0
        synaptic_inputs[3, 0, 0:synaptic_kernel_length] = 300.0

        extrinsic_inputs = np.zeros((2, 5 * synaptic_kernel_length), dtype=float)
        extrinsic_inputs[1, 0:synaptic_kernel_length] = 1e-14

        # for each combination of inputs calculate state vector of population instance
        ##############################################################################

        states = np.zeros((synaptic_inputs.shape[0], extrinsic_inputs.shape[0], extrinsic_inputs.shape[1]))
        for i in range(synaptic_inputs.shape[0]):
            for j in range(extrinsic_inputs.shape[0]):
                pop = Population(synapses=synapse_types,
                                 axon=axon,
                                 init_state=init_state,
                                 step_size=step_size,
                                 synaptic_kernel_length=synaptic_kernel_length,
                                 tau_leak=tau_leak,
                                 resting_potential=resting_potential,
                                 membrane_capacitance=membrane_capacitance)
                for k in range(synaptic_inputs.shape[2]):
                    pop.state_update(synaptic_input=synaptic_inputs[i, :, k],
                                     extrinsic_current=extrinsic_inputs[j, k])
                    states[i, j, k] = pop.state_variables[-1]

        # perform unit tests
        ####################

        print('---------------------------------')
        print('| Test VI - Population Dynamics |')
        print('---------------------------------')

        print('VI.1 test whether resulting membrane potential for zero input is equal to resting potential')
        self.assertAlmostEqual(states[0, 0, -1], resting_potential, places=2)
        print('VI.1 done!')

        print('VI.2 test whether constant excitatory synaptic input leads to increased membrane potential')
        self.assertGreater(states[1, 0, -1], resting_potential)
        print('VI.2 done!')

        print('VI.3 test whether constant inhibitory input leads to decreased membrane potential')
        self.assertLess(states[2, 0, -1], resting_potential)
        print('VI.3 done!')

        print('VI.4 test whether extrinsic current leads to expected change in membrane potential')
        self.assertAlmostEqual(states[0, 1, 0], init_state[0] + step_size * (1e-14 / membrane_capacitance), places=4)
        print('VI.4 done!')

        print('VI.5 test whether membrane potential goes back to resting potential after step-function input')
        self.assertAlmostEqual(states[3, 0, -1], resting_potential, places=4)
        self.assertAlmostEqual(states[0, 1, -1], resting_potential, places=4)
        print('VI.5 done!')

    def test_6_JR_circuit_I(self):
        """
        Tests whether current implementation shows expected behavior when standard Jansen-Rit circuit is fed with step-
        function input targeted onto the excitatory interneurons.
        """

        # set parameters
        ################

        with open('JR_parameters_I.pickle', 'rb') as f:
            connections, population_labels, step_size, synaptic_kernel_length, distances, velocities, synapse_params, \
            axon_params, init_states, synaptic_inputs, simulation_time, cutoff_time, store_step = pickle.load(f)

        # # simulations parameters
        # simulation_time = 1.0     # s
        # cutoff_time = 0.0         # s
        # step_size = 5.0e-4        # s
        # store_step = 1
        #
        # # populations
        # population_labels = ['PC', 'EIN', 'IIN']
        # N = len(population_labels)
        # n_synapses = 2
        #
        # # synapses
        # connections = np.zeros((N, N, n_synapses))
        #
        # # AMPA connections (excitatory)
        # connections[:, :, 0] = [[0, 0.8 * 135, 0], [1.0 * 135, 0, 0], [0.25 * 135, 0, 0]]
        #
        # # GABA-A connections (inhibitory)
        # connections[:, :, 1] = [[0, 0, 0.25 * 135], [0, 0, 0], [0, 0, 0]]
        #
        # ampa_dict = {'efficiency': 1.273 * 3e-13,     # A
        #              'tau_decay': 0.006,              # s
        #              'tau_rise': 0.0006,              # s
        #              'conductivity_based': False}
        #
        # gaba_a_dict = {'efficiency': 1.273 * -1e-12,    # A
        #                'tau_decay': 0.02,               # s
        #                'tau_rise': 0.0004,              # s
        #                'conductivity_based': False}
        #
        # synapse_params = [ampa_dict, gaba_a_dict]
        # synaptic_kernel_length = 100                  # in time steps
        #
        # # axon
        # axon_dict = {'max_firing_rate': 5.,                     # 1/s
        #              'membrane_potential_threshold': -0.069,    # V
        #              'sigmoid_steepness': 555.56}               # 1/V
        # axon_params = [axon_dict for i in range(N)]
        #
        # distances = np.zeros((N, N))
        # velocities = float('inf')
        #
        # init_states = np.zeros((N, n_synapses))
        #
        # # synaptic inputs
        # start_stim = 0.3        # s
        # len_stim = 0.05         # s
        # mag_stim = 300.0        # 1/s
        #
        # synaptic_inputs = np.zeros((int(simulation_time/step_size), N, n_synapses))
        # synaptic_inputs[int(start_stim/step_size):int(start_stim/step_size+len_stim/step_size), 1, 0] = mag_stim

        # initialize neural mass network
        ################################

        nmm = NeuralMassModel(connections=connections,
                              population_labels=population_labels,
                              step_size=step_size,
                              synaptic_kernel_length=synaptic_kernel_length,
                              distances=distances,
                              velocities=velocities,
                              synapse_params=synapse_params,
                              axon_params=axon_params,
                              init_states=init_states)

        # run network simulation
        ########################

        print('---------------------------------')
        print('| Test VII - Jansen-Rit Circuit |')
        print('---------------------------------')

        nmm.run(synaptic_inputs=synaptic_inputs,
                simulation_time=simulation_time,
                cutoff_time=cutoff_time,
                store_step=store_step)

        states = np.array(nmm.neural_mass_states)

        # load target data
        ###################

        with open('JR_results_I.pickle', 'rb') as f:
            target_states = pickle.load(f)

        # calculate NMRSE between time-series
        #####################################

        error = NMRSE(states, target_states)
        error = np.sum(error)

        # perform unit test
        ###################

        print('VII.1 test response to step-function input to EINs')
        self.assertAlmostEqual(error, 0, places=2)
        print('VII.1 done!')

    def test_7_JR_circuit_II(self):
        """
        Tests whether current implementation shows expected behavior when standard Jansen-Rit circuit is fed with step-
        function input to the excitatory interneurons plus constant input to the pyramidal cells.
        """

        # set parameters
        ################

        with open('JR_parameters_II.pickle', 'rb') as f:
            connections, population_labels, step_size, synaptic_kernel_length, distances, velocities, synapse_params, \
            axon_params, init_states, synaptic_inputs, simulation_time, cutoff_time, store_step = pickle.load(f)

        # # simulations parameters
        # simulation_time = 1.0     # s
        # cutoff_time = 0.0         # s
        # step_size = 5.0e-4        # s
        # store_step = 1
        #
        # # populations
        # population_labels = ['PC', 'EIN', 'IIN']
        # N = len(population_labels)
        # n_synapses = 2
        #
        # # synapses
        # connections = np.zeros((N, N, n_synapses))
        #
        # # AMPA connections (excitatory)
        # connections[:, :, 0] = [[0, 0.8 * 135, 0], [1.0 * 135, 0, 0], [0.25 * 135, 0, 0]]
        #
        # # GABA-A connections (inhibitory)
        # connections[:, :, 1] = [[0, 0, 0.25 * 135], [0, 0, 0], [0, 0, 0]]
        #
        # ampa_dict = {'efficiency': 1.273 * 3e-13,     # A
        #              'tau_decay': 0.006,              # s
        #              'tau_rise': 0.0006,              # s
        #              'conductivity_based': False}
        #
        # gaba_a_dict = {'efficiency': 1.273 * -1e-12,    # A
        #                'tau_decay': 0.02,               # s
        #                'tau_rise': 0.0004,              # s
        #                'conductivity_based': False}
        #
        # synapse_params = [ampa_dict, gaba_a_dict]
        # synaptic_kernel_length = 100                  # in time steps
        #
        # # axon
        # axon_dict = {'max_firing_rate': 5.,                     # 1/s
        #              'membrane_potential_threshold': -0.069,    # V
        #              'sigmoid_steepness': 555.56}               # 1/V
        # axon_params = [axon_dict for i in range(N)]
        #
        # distances = np.zeros((N, N))
        # velocities = float('inf')
        #
        # init_states = np.zeros((N, n_synapses))
        #
        # # synaptic inputs
        # start_stim = 0.3        # s
        # len_stim = 0.05         # s
        # mag_stim = 300.0        # 1/s
        #
        # synaptic_inputs = np.zeros((int(simulation_time/step_size), N, n_synapses))
        # synaptic_inputs[int(start_stim/step_size):int(start_stim/step_size+len_stim/step_size), 1, 0] = mag_stim
        # synaptic_inputs[:, 0, 0] = mag_stim/3.

        # initialize neural mass network
        ################################

        nmm = NeuralMassModel(connections=connections,
                              population_labels=population_labels,
                              step_size=step_size,
                              synaptic_kernel_length=synaptic_kernel_length,
                              distances=distances,
                              velocities=velocities,
                              synapse_params=synapse_params,
                              axon_params=axon_params,
                              init_states=init_states)

        # run network simulation
        ########################

        nmm.run(synaptic_inputs=synaptic_inputs,
                simulation_time=simulation_time,
                cutoff_time=cutoff_time,
                store_step=store_step)

        states = np.array(nmm.neural_mass_states).T

        # load target data
        ###################

        with open('JR_results_II.pickle', 'rb') as f:
            target_states = pickle.load(f)

        # calculate NMRSE between time-series
        #####################################

        error = NMRSE(states, target_states)
        error = np.sum(error)

        # perform unit test
        ###################

        print('VII.2 test response to step-function input to EINs plus constant input to PCs')
        self.assertAlmostEqual(error, 0, places=2)
        print('VII.2 done!')

    def test_8_JR_circuit_III(self):
        """
        Tests whether expected bifurcation occurs when synaptic efficiency of JR circuit is altered (given constant
        input).
        """

        # set parameters
        ################

        with open('JR_parameters_III.pickle', 'rb') as f:
            connections, population_labels, step_size, synaptic_kernel_length, distances, velocities, gaba_a_dict, \
            axon_params, init_states, synaptic_inputs, simulation_time, cutoff_time, store_step = pickle.load(f)

        # # simulations parameters
        # simulation_time = 3.0     # s
        # cutoff_time = 0.0         # s
        # step_size = 5.0e-4        # s
        # store_step = 1
        #
        # # populations
        # population_labels = ['PC', 'EIN', 'IIN']
        # N = len(population_labels)
        # n_synapses = 2
        #
        # # synapses
        # connections = np.zeros((N, N, n_synapses))
        #
        # # AMPA connections (excitatory)
        # connections[:, :, 0] = [[0, 0.8 * 135, 0], [1.0 * 135, 0, 0], [0.25 * 135, 0, 0]]
        #
        # # GABA-A connections (inhibitory)
        # connections[:, :, 1] = [[0, 0, 0.25 * 135], [0, 0, 0], [0, 0, 0]]
        #
        # gaba_a_dict = {'efficiency': 0.5 * 1.273 * -1e-12,    # A
        #                'tau_decay': 0.02,               # s
        #                'tau_rise': 0.0004,              # s
        #                'conductivity_based': False}
        #
        # synaptic_kernel_length = 100                  # in time steps
        #
        # # axon
        # axon_dict = {'max_firing_rate': 5.,                     # 1/s
        #              'membrane_potential_threshold': -0.069,    # V
        #              'sigmoid_steepness': 555.56}               # 1/V
        # axon_params = [axon_dict for i in range(N)]
        #
        # distances = np.zeros((N, N))
        # velocities = float('inf')
        #
        # init_states = np.zeros((N, n_synapses))
        #
        # # synaptic inputs
        # mag_stim = 200.0        # 1/s
        # synaptic_inputs = np.zeros((int(simulation_time/step_size), N, n_synapses))
        # synaptic_inputs[:, 1, 0] = mag_stim

        # loop over different AMPA synaptic efficiencies and simulate network behavior
        ##############################################################################

        ampa_efficiencies = np.linspace(0.1, 1.0, 20) * 1.273 * 3e-13

        final_state = np.zeros(len(ampa_efficiencies))

        for i, efficiency in enumerate(ampa_efficiencies):
            # set ampa parameters
            #####################

            ampa_dict = {'efficiency': float(efficiency),  # A
                         'tau_decay': 0.006,  # s
                         'tau_rise': 0.0006,  # s
                         'conductivity_based': False}

            synapse_params = [ampa_dict, gaba_a_dict]

            # initialize neural mass network
            ################################

            nmm = NeuralMassModel(connections=connections,
                                  population_labels=population_labels,
                                  step_size=step_size,
                                  synaptic_kernel_length=synaptic_kernel_length,
                                  distances=distances,
                                  velocities=velocities,
                                  synapse_params=synapse_params,
                                  axon_params=axon_params,
                                  init_states=init_states)

            # run network simulation
            ########################

            nmm.run(synaptic_inputs=synaptic_inputs,
                    simulation_time=simulation_time,
                    cutoff_time=cutoff_time,
                    store_step=store_step)

            final_state[i] = nmm.neural_mass_states[-1][0]

        # load target data
        ###################

        with open('JR_results_III.pickle', 'rb') as f:
            target_states = pickle.load(f)

        # calculate NMRSE between time-series
        #####################################

        error = NMRSE(final_state, target_states)
        error = np.sum(error)

        # perform unit test
        ###################

        print('VII.3 test response to varying AMPA synapse efficiencies given constant input to EINs.')
        self.assertAlmostEqual(error, 0, places=2)
        print('VII.3 done!')

        # def test_9_JR_network_I(self):
        #     """
        #     tests whether 2 identical connected JR circuits behave as expected.
        #     """
        #
        #     # set parameters
        #     ################
        #
        #     # connectivity matrices
        #     C1 = np.array([[0, 1], [1, 0]]) * 200.
        #     C2 = np.array([[0, 0], [1, 0]]) * 200.
        #     C3 = np.zeros((2, 2))
        #
        #     # distance matrix
        #     D = np.zeros((2, 2))
        #     D[0, 1] = 0.01
        #     D[1, 0] = 0.01
        #
        #     # velocity
        #     v = 10.
        #
        #     # neural mass circuit types
        #     nmm_types = ['JansenRitCircuit', 'JansenRitCircuit']
        #
        #     # simulation step-size
        #     stepsize = 5e-4
        #
        #     # simulation time
        #     #################
        #     T = 1.
        #     timesteps = np.int(T / stepsize)
        #
        #     # synaptic input
        #     stim_time = 0.3
        #     stim_timesteps = np.int(stim_time / stepsize)
        #     synaptic_input = np.zeros((timesteps, 2, 3, 2))
        #     synaptic_input[0:stim_timesteps, 0, 1, 0] = 300.
        #
        #     # initialize nmm network
        #     ########################
        #
        #     nmm1 = NeuralMassNetwork(connections=C1,
        #                                          nmm_types=nmm_types,
        #                                          distances=D,
        #                                          velocities=v,
        #                                          step_size=stepsize)
        #
        #     nmm2 = NeuralMassNetwork(connections=C2,
        #                                          nmm_types=nmm_types,
        #                                          distances=D,
        #                                          velocities=v,
        #                                          step_size=stepsize)
        #
        #     nmm3 = NeuralMassNetwork(connections=C3,
        #                                          nmm_types=nmm_types,
        #                                          distances=D,
        #                                          velocities=v,
        #                                          step_size=stepsize)
        #
        #     # run network simulations
        #     #########################
        #
        #     nmm1.run(synaptic_inputs=synaptic_input, simulation_time=T)
        #     nmm2.run(synaptic_inputs=synaptic_input, simulation_time=T)
        #     nmm3.run(synaptic_inputs=synaptic_input, simulation_time=T)
        #
        #     # perform unit tests
        #     ####################
        #
        #     nmm1.plot_neural_mass_states()
        #     nmm2.plot_neural_mass_states()
        #
        #     self.assertTupleEqual(nmm1.neural_mass_states, nmm2.neural_mass_states)


##################
# run unit tests #
##################


if __name__ == '__main__':
    unittest.main()
