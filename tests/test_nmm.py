"""Unit tests for all modules of BrainNetworks/core.
"""

import pickle
import unittest
import numpy as np

from core.axon import SigmoidAxon, JansenRitAxon
from core.circuit import CircuitFromScratch, CircuitFromPopulations, CircuitFromCircuit, JansenRitCircuit
from core.population import Population
from core.synapse import AMPACurrentSynapse, GABAACurrentSynapse
from core.synapse import Synapse, DoubleExponentialSynapse
from core.utility import NMRSE

__author__ = "Richard Gast & Konstantin Weise"
__status__ = "Test"


##############
# unit tests #
##############


class TestNMMs(unittest.TestCase):
    """Unit tests for all components of NMM network.
    """

    def test_0_JR_axon(self):
        """Tests whether axon with standard parametrization from [1]_ shows expected output to membrane potential input.

        See Also
        --------
        :class:`SigmoidAxon`: Detailed documentation of axon parameters.
        :class:`Axon`: Detailed documentation of axon attributes and methods.

        References
        ----------
        .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
           of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

        """

        ###################
        # axon parameters #
        ###################

        max_firing_rate = 5.  # unit = 1
        membrane_potential_threshold = -0.069  # unit = V
        sigmoid_steepness = 555.56  # unit = 1/V

        ###################
        # initialize axon #
        ###################

        axon = SigmoidAxon(max_firing_rate=max_firing_rate,
                           membrane_potential_threshold=membrane_potential_threshold,
                           sigmoid_steepness=sigmoid_steepness)

        ############################
        # define inputs (unit = V) #
        ############################

        membrane_potential_1 = membrane_potential_threshold
        membrane_potential_2 = membrane_potential_threshold - 0.01
        membrane_potential_3 = membrane_potential_threshold + 0.01
        membrane_potential_4 = membrane_potential_threshold - 0.1
        membrane_potential_5 = membrane_potential_threshold + 0.1

        ####################
        # get firing rates #
        ####################

        firing_rate_1 = axon.compute_firing_rate(membrane_potential_1)
        firing_rate_2 = axon.compute_firing_rate(membrane_potential_2)
        firing_rate_3 = axon.compute_firing_rate(membrane_potential_3)
        firing_rate_4 = axon.compute_firing_rate(membrane_potential_4)
        firing_rate_5 = axon.compute_firing_rate(membrane_potential_5)

        ######################
        # perform unit tests #
        ######################

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
        """Tests whether synapse with standard AMPA parametrization from Thomas Knoesche (corresponding to AMPA synapse
         in [1]_) shows expected output for various firing rate inputs.

        See Also
        --------
        :class:`DoubleExponentialSynapse`: Detailed documentation of synapse parameters.
        :class:`Synapse`: Detailed documentation of synapse attributes and methods.

        References
        ----------
        .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
           of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

        """

        # synapse parameters
        ####################

        efficacy = 1.273 * 3e-13  # unit = A
        tau_decay = 0.006  # unit = s
        tau_rise = 0.0006  # unit = s
        step_size = 5.e-4  # unit = s
        synaptic_kernel_length = 0.05  # unit = s
        conductivity_based = False

        # initialize synapse
        ####################

        synapse = DoubleExponentialSynapse(efficacy=efficacy,
                                           tau_decay=tau_decay,
                                           tau_rise=tau_rise,
                                           bin_size=step_size,
                                           max_delay=synaptic_kernel_length,
                                           conductivity_based=conductivity_based)

        # define firing rate inputs
        ###########################

        time_steps = int(synaptic_kernel_length/step_size)
        firing_rates_1 = np.zeros(time_steps)
        firing_rates_2 = np.ones(time_steps) * 300.0
        firing_rates_3 = np.zeros(3 * time_steps)
        firing_rates_3[time_steps:2 * time_steps] = 300.0

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
        """Tests whether synapse with standard GABAA parametrization from Thomas Knoesche (corresponding to GABAA
        synapse in [1]_) shows expected output for various firing rate inputs.

        See Also
        --------
        :class:`DoubleExponentialSynapse`: Detailed documentation of synapse parameters.
        :class:`Synapse`: Detailed documentation of synapse attributes and methods.

        References
        ----------
        .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
           of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

        """

        # synapse parameters
        ####################

        efficacy = 1.273 * -1e-12  # unit = A
        tau_decay = 0.02  # unit = s
        tau_rise = 0.0004  # unit = s
        step_size = 5.e-4  # unit = s
        synaptic_kernel_length = 0.05  # unit = s
        conductivity_based = False

        # initialize synapse
        ####################

        synapse = DoubleExponentialSynapse(efficacy=efficacy,
                                           tau_decay=tau_decay,
                                           tau_rise=tau_rise,
                                           bin_size=step_size,
                                           max_delay=synaptic_kernel_length,
                                           conductivity_based=conductivity_based)

        # define firing rate inputs
        ###########################
        time_steps = int(0.05 / step_size)
        firing_rates_1 = np.zeros(time_steps)
        firing_rates_2 = np.ones(time_steps) * 300.0
        firing_rates_3 = np.zeros(3 * time_steps)
        firing_rates_3[time_steps:2 * time_steps] = 300.0

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
        """Tests whether synapse with parametrization from Thomas Knoesche corresponding to conductivity based AMPA
        synapse shows expected output for various firing rate inputs.

        See Also
        --------
        :class:`DoubleExponentialSynapse`: Detailed documentation of synapse parameters.
        :class:`Synapse`: Detailed documentation of synapse attributes and methods.

        """

        # synapse parameters
        ####################

        efficacy = 1.273 * 7.2e-10  # unit = S
        tau_decay = 0.0015  # unit = s
        tau_rise = 0.000009  # unit = s
        step_size = 5.e-4  # unit = s
        synaptic_kernel_length = 0.05  # unit = s
        conductivity_based = True

        # initialize synapse
        ####################

        synapse = DoubleExponentialSynapse(efficacy=efficacy,
                                           tau_decay=tau_decay,
                                           tau_rise=tau_rise,
                                           bin_size=step_size,
                                           max_delay=synaptic_kernel_length,
                                           conductivity_based=conductivity_based)

        # define firing rate inputs
        ###########################

        time_steps = int(0.05 / step_size)
        firing_rates_1 = np.zeros(time_steps)
        firing_rates_2 = np.ones(time_steps) * 300.0
        firing_rates_3 = np.zeros(3 * time_steps)
        firing_rates_3[time_steps:2 * time_steps] = 300.0

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
        """Tests whether synapses and axon of initialized population show expected behavior.

        See Also
        --------
        :class:`Population`: Detailed documentation of population parameters, attributes and methods.

        """

        # population parameters
        #######################

        synapse_types = ['AMPACurrentSynapse', 'GABAACurrentSynapse']
        axon = 'JansenRitAxon'
        init_state = -0.07
        step_size = 5.e-4
        synaptic_kernel_length = 0.05
        tau_leak = 0.016
        resting_potential = -0.07

        # initialize population, synapses and axon
        pop = Population(synapses=synapse_types,
                         axon=axon,
                         init_state=init_state,
                         step_size=step_size,
                         max_synaptic_delay=synaptic_kernel_length,
                         tau_leak=tau_leak,
                         resting_potential=resting_potential)
        syn1 = AMPACurrentSynapse(bin_size=step_size,
                                  max_delay=synaptic_kernel_length)
        syn2 = GABAACurrentSynapse(bin_size=step_size,
                                   max_delay=synaptic_kernel_length)
        axon = JansenRitAxon()

        # define firing rate input and membrane potential
        #################################################

        time_steps = int(0.05 / step_size)
        firing_rate = np.zeros(time_steps) + 300.0
        membrane_potential = -0.06

        # calculate population, synapse and axon response
        #################################################

        pop_syn1_response = pop.synapses[0].get_synaptic_current(firing_rate)
        pop_syn2_response = pop.synapses[1].get_synaptic_current(firing_rate)
        syn1_response = syn1.get_synaptic_current(firing_rate)
        syn2_response = syn2.get_synaptic_current(firing_rate)

        pop_ax_response = pop.axon.compute_firing_rate(membrane_potential)
        ax_response = axon.compute_firing_rate(membrane_potential)

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
        self.assertEqual(pop_ax_response, ax_response)
        print('V.2 done!')

    def test_5_population_dynamics(self):
        """Tests whether population develops as expected over time given some input.

        See Also
        --------
        :class:`Population`: Detailed documentation of population parameters, attributes and methods.

        """

        # set population parameters
        ###########################

        synapse_types = ['AMPACurrentSynapse', 'GABAACurrentSynapse']
        axon = 'JansenRitAxon'
        step_size = 5e-4  # unit = s
        synaptic_kernel_length = 0.05  # unit = s
        tau_leak = 0.016  # unit = s
        resting_potential = -0.075  # unit = V
        membrane_capacitance = 1e-12  # unit = q/V
        init_state = resting_potential  # unit = (V, 1/s)

        # define population input
        #########################

        time_steps = int(0.05 / step_size)
        synaptic_inputs = np.zeros((4, 2, 5 * time_steps))
        synaptic_inputs[1, 0, :] = 300.0
        synaptic_inputs[2, 1, :] = 300.0
        synaptic_inputs[3, 0, 0:time_steps] = 300.0

        extrinsic_inputs = np.zeros((2, 5 * time_steps), dtype=float)
        extrinsic_inputs[1, 0:time_steps] = 1e-14

        # for each combination of inputs calculate state vector of population instance
        ##############################################################################

        states = np.zeros((synaptic_inputs.shape[0], extrinsic_inputs.shape[0], extrinsic_inputs.shape[1]))
        for i in range(synaptic_inputs.shape[0]):
            for j in range(extrinsic_inputs.shape[0]):
                pop = Population(synapses=synapse_types,
                                 axon=axon,
                                 init_state=init_state,
                                 step_size=step_size,
                                 max_synaptic_delay=synaptic_kernel_length,
                                 tau_leak=tau_leak,
                                 resting_potential=resting_potential,
                                 membrane_capacitance=membrane_capacitance)
                for k in range(synaptic_inputs.shape[2]):
                    pop.synaptic_input[pop.current_input_idx, :] = synaptic_inputs[i, :, k]
                    pop.state_update(extrinsic_current=extrinsic_inputs[j, k])
                    states[i, j, k] = pop.state_variables[-1][0]

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
        self.assertAlmostEqual(states[0, 1, 0], init_state + step_size * (1e-14 / membrane_capacitance), places=4)
        print('VI.4 done!')

        print('VI.5 test whether membrane potential goes back to resting potential after step-function input')
        self.assertAlmostEqual(states[3, 0, -1], resting_potential, places=4)
        self.assertAlmostEqual(states[0, 1, -1], resting_potential, places=4)
        print('VI.5 done!')

    def test_6_JR_circuit_I(self):
        """Tests whether current implementation shows expected behavior when standard Jansen-Rit circuit ([1]_)is fed
        with step-function input targeted onto the excitatory interneurons.

        See Also
        --------
        :class:`JansenRitCircuit`: Documentation of Jansen-Rit NMM parametrization
        :class:`NeuralMassModel`: Detailed documentation of NMM parameters, attributes and methods.

        References
        ----------
        .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
           of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

        """

        # set parameters
        ################

        # simulations parameters
        simulation_time = 1.0     # s
        cutoff_time = 0.0         # s
        step_size = 5.0e-4        # s

        # populations
        populations = ['JansenRitPyramidalCells',
                       'JansenRitExcitatoryInterneurons',
                       'JansenRitInhibitoryInterneurons']
        population_labels = ['PC', 'EIN', 'IIN']
        N = len(population_labels)
        n_synapses = 2

        # synapses
        connections = np.zeros((N, N, n_synapses))

        # AMPA connections (excitatory)
        connections[:, :, 0] = [[0, 0.8 * 135, 0], [1.0 * 135, 0, 0], [0.25 * 135, 0, 0]]

        # GABA-A connections (inhibitory)
        connections[:, :, 1] = [[0, 0, 0.25 * 135], [0, 0, 0], [0, 0, 0]]

        # other population parameters
        max_synaptic_delay = 0.05                  # s
        init_states = np.zeros(N)                  # V
        resting_potential = -0.075                 # V
        tau_leak = 0.016                           # s
        membrane_capacitance = 1e-12               # q/S
        delays = None
        variable_step_size = False
        synaptic_modulation_direction = None

        # synaptic inputs
        start_stim = 0.3        # s
        len_stim = 0.05         # s
        mag_stim = 300.0        # 1/s

        synaptic_inputs = np.zeros((int(simulation_time/step_size), N, n_synapses))
        synaptic_inputs[int(start_stim/step_size):int(start_stim/step_size+len_stim/step_size), 1, 0] = mag_stim

        # initialize neural mass network
        ################################

        nmm = CircuitFromPopulations(population_types=populations,
                                     connectivity=connections,
                                     step_size=step_size,
                                     max_synaptic_delay=max_synaptic_delay,
                                     init_states=init_states,
                                     delays=delays,
                                     resting_potential=resting_potential,
                                     tau_leak=tau_leak,
                                     membrane_capacitance=membrane_capacitance,
                                     population_labels=population_labels,
                                     variable_step_size=variable_step_size,
                                     synaptic_modulation_direction=synaptic_modulation_direction)

        # run network simulation
        ########################

        print('---------------------------------')
        print('| Test VII - Jansen-Rit Circuit |')
        print('---------------------------------')

        nmm.run(synaptic_inputs=synaptic_inputs,
                simulation_time=simulation_time)

        states = nmm.get_population_states(state_variable_idx=0)

        # load target data
        ###################

        with open('JR_results_I.pickle', 'rb') as f:
            target_states = pickle.load(f)

        # calculate NMRSE between time-series
        #####################################

        error = NMRSE(states[1:, :], target_states)
        error = np.mean(error)

        # perform unit test
        ###################

        print('VII.1 test response to step-function input to EINs')
        self.assertAlmostEqual(error, 0, places=0)
        print('VII.1 done!')

    def test_7_JR_circuit_II(self):
        """
        Tests whether current implementation shows expected behavior when standard Jansen-Rit circuit is fed with step-
        function input to the excitatory interneurons plus constant input to the pyramidal cells.
        """

        # set parameters
        ################

        # circuit parameters
        N = 3
        n_synapses = 2

        # simulations parameters
        simulation_time = 1.0     # s
        cutoff_time = 0.0         # s
        step_size = 5.0e-4        # s

        # synaptic inputs
        start_stim = 0.3        # s
        len_stim = 0.05         # s
        mag_stim = 300.0        # 1/s

        synaptic_inputs = np.zeros((int(simulation_time/step_size), N, n_synapses))
        synaptic_inputs[int(start_stim/step_size):int(start_stim/step_size+len_stim/step_size), 1, 0] = mag_stim
        synaptic_inputs[:, 0, 0] = mag_stim/3.

        # initialize neural mass network
        ################################

        nmm = JansenRitCircuit()

        # run network simulation
        ########################

        nmm.run(synaptic_inputs=synaptic_inputs,
                simulation_time=simulation_time)

        states = nmm.get_population_states(state_variable_idx=0)

        # load target data
        ###################

        with open('JR_results_II.pickle', 'rb') as f:
            target_states = pickle.load(f)

        # calculate NMRSE between time-series
        #####################################

        error = NMRSE(states[1:, :], target_states.T)
        error = np.mean(error)

        # perform unit test
        ###################

        print('VII.2 test response to step-function input to EINs plus constant input to PCs')
        self.assertAlmostEqual(error, 0, places=0)
        print('VII.2 done!')

    def test_8_JR_circuit_III(self):
        """
        Tests whether expected bifurcation occurs when synaptic efficiency of JR circuit is altered (given constant
        input).
        """

        # set parameters
        ################

        # simulations parameters
        simulation_time = 3.0     # s
        cutoff_time = 0.0         # s
        step_size = 5.0e-4        # s

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

        gaba_a_dict = {'efficacy': 0.5 * 1.273 * -1e-12,    # A
                       'tau_decay': 0.02,               # s
                       'tau_rise': 0.0004,              # s
                       'conductivity_based': False}

        max_synaptic_delay = 0.05                  # s

        # axon
        axon_dict = {'max_firing_rate': 5.,                     # 1/s
                     'membrane_potential_threshold': -0.069,    # V
                     'sigmoid_steepness': 555.56}               # 1/V
        axon_params = [axon_dict for i in range(N)]

        init_states = np.zeros(N)

        # synaptic inputs
        mag_stim = 200.0        # 1/s
        synaptic_inputs = np.zeros((int(simulation_time/step_size), N, n_synapses))
        synaptic_inputs[:, 1, 0] = mag_stim

        # loop over different AMPA synaptic efficiencies and simulate network behavior
        ##############################################################################

        ampa_efficiencies = np.linspace(0.1, 1.0, 20) * 1.273 * 3e-13

        final_state = np.zeros(len(ampa_efficiencies))

        for i, efficiency in enumerate(ampa_efficiencies):

            # set ampa parameters
            #####################

            ampa_dict = {'efficacy': float(efficiency),  # A
                         'tau_decay': 0.006,  # s
                         'tau_rise': 0.0006,  # s
                         'conductivity_based': False}

            synapse_params = [ampa_dict, gaba_a_dict]

            # initialize neural mass network
            ################################

            nmm = CircuitFromScratch(connectivity=connections,
                                     step_size=step_size,
                                     synapse_params=synapse_params,
                                     axon_params=axon_params,
                                     max_synaptic_delay=max_synaptic_delay,
                                     init_states=init_states,
                                     population_labels=population_labels,
                                     delays=None)

            # run network simulation
            ########################

            nmm.run(synaptic_inputs=synaptic_inputs,
                    simulation_time=simulation_time)

            final_state[i] = nmm.get_population_states(state_variable_idx=0)[-1, 0]

        # load target data
        ###################

        with open('JR_results_III.pickle', 'rb') as f:
            target_states = pickle.load(f)

        # calculate NMRSE between time-series
        #####################################

        error = NMRSE(final_state, target_states)
        error = np.mean(error)

        # perform unit test
        ###################

        print('VII.3 test response to varying AMPA synapse efficiencies given constant input to EINs.')
        self.assertAlmostEqual(error, 0, places=0)
        print('VII.3 done!')

    def test_9_JR_network_I(self):
        """
        tests whether 2 delay-connected vs unconnected JR circuits behave as expected.
        """

        # set parameters
        ################

        # connectivity matrices
        inter_circuit_conns = np.array([[0, 1], [0, 0]]) * 100.
        C1 = np.zeros((2, 2, 2))
        C2 = np.zeros((2, 2, 2))
        C2[:, :, 0] = inter_circuit_conns

        # delay matrix
        D = np.zeros((2, 2))
        D[0, 1] = 0.001
        D[1, 0] = 0.001

        # neural mass circuits
        nmm1 = JansenRitCircuit()
        nmm2 = JansenRitCircuit()
        nmm3 = JansenRitCircuit()
        nmm4 = JansenRitCircuit()

        # simulation step-size
        step_size = 5e-4

        # simulation time
        simulation_time = 1.
        timesteps = np.int(simulation_time / step_size)

        # synaptic input
        stim_time = 0.3
        stim_timesteps = np.int(stim_time / step_size)
        synaptic_input = np.zeros((timesteps, 6, 2))
        synaptic_input[0:stim_timesteps, 1, 0] = 300.

        # initialize nmm network
        ########################
        circuit1 = CircuitFromCircuit(circuits=[nmm1, nmm2],
                                      connectivity=C1,
                                      delays=D,
                                      circuit_labels=['NMM1', 'NMM2'])
        circuit2 = CircuitFromCircuit(circuits=[nmm3, nmm4],
                                      connectivity=C2,
                                      delays=D,
                                      circuit_labels=['NMM1', 'NMM2'])

        # run network simulations
        #########################

        circuit1.run(synaptic_inputs=synaptic_input, simulation_time=simulation_time)
        circuit2.run(synaptic_inputs=synaptic_input, simulation_time=simulation_time)

        # perform unit tests
        ####################

        states1 = circuit1.get_population_states(state_variable_idx=0)
        states2 = circuit2.get_population_states(state_variable_idx=0)

        error = NMRSE(states1, states2)
        error = np.mean(error)

        print('VII.4 test information transfer between two delay-connected JR circuits...')
        self.assertNotAlmostEqual(error, 0, places=0)
        print('VII.4 done!')


##################
# run unit tests #
##################

if __name__ == '__main__':
    unittest.main()
