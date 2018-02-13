"""Contains overall BGTC network class plus microcircuit classes for the network nodes
"""

__author__ = 'Richard Gast'

import numpy as np
from core.circuit import Circuit, CircuitFromScratch, CircuitFromCircuit
from core.population import SecondOrderPopulation, PlasticPopulation, JansenRitPyramidalCells, JansenRitInterneurons


#####################
# class definitions #
#####################


class BGTC(CircuitFromCircuit):
    """Model of the basal ganglia-thalamocortical network.
    """

    def __init__(self, step_size, max_synaptic_delay):

        # initialize each sub-circuit
        #############################

        pass


class M1(CircuitFromScratch):
    """Model of the primary motor cortex (M1/BA4).
    """

    def __init__(self, step_size=1e-3, max_synaptic_delay=None, connectivity=None, connectivity_scaling=135.,
                 feedback=None, delays=None, synapse_params=None, conductance_based=False, axon_params=None,
                 tau_leak=0.016, resting_potential=-0.075, init_states=None
                 ):

        # set parameters
        ################

        n_populations = 3
        n_synapses = 2
        population_labels = ['L23_PCs', 'L5_PCs', 'IINs']

        # connection strengths
        if connectivity is None:
            fb = np.zeros(n_populations) if feedback is None else feedback
            c = connectivity_scaling
            connectivity = np.zeros((n_populations, n_populations, n_synapses))
            connectivity[:, :, 0] = [[fb[0] * c, 0.5 * c, 0.],
                                     [1.0 * c, fb[1] * c, 0.],
                                     [0.2 * c, 0.05 * c, fb[2] * c]]
            connectivity[:, :, 1] = [[0., 0., 0.2 * c],
                                     [0., 0., 0.2 * c],
                                     [0., 0., fb[2] * c]]

        # synapses
        if conductance_based:
            synapse_types = ['AMPAConductanceSynapse', 'GABAAConductanceSynapse']
        else:
            synapse_types = ['AMPACurrentSynapse', 'GABAACurrentSynapse']

        # axons
        axon_types = ['KnoescheAxon' for _ in range(n_populations)]

        # initial condition
        if init_states is None:
            init_states = np.zeros(n_populations)

        # call super init
        #################

        super().__init__(connectivity=connectivity,
                         delays=delays,
                         step_size=step_size,
                         synapses=synapse_types,
                         synapse_class='DoubleExponentialSynapse',
                         synapse_params=synapse_params,
                         axons=axon_types,
                         axon_params=axon_params,
                         max_synaptic_delay=max_synaptic_delay,
                         population_class='Population',
                         membrane_capacitance=1e-12,
                         tau_leak=tau_leak,
                         resting_potential=resting_potential,
                         init_states=init_states,
                         population_labels=population_labels
                         )


class Thalamus(Circuit):
    """Model of a single thalamus channel based on [1]_.
    """

    def __init__(self, step_size=1e-3, max_synaptic_delay=None, epsilon=1e-10, connectivity=None,
                 connectivity_scaling=20., feedback=None, delays=None, synapse_params=None,
                 axon_params=None, init_states=None):
        """Instantiates thalamus model.
        """

        # set parameters
        ################

        n_populations = 2
        n_synapses = 3
        population_labels = ['TCR', 'RE']

        # connection strengths
        if connectivity is None:
            fb = np.zeros(n_populations) if feedback is None else feedback
            c = connectivity_scaling
            connectivity = np.zeros((n_populations, n_populations, n_synapses))
            connectivity[:, :, 0] = [[fb[0] * c, 1.],      # AMPA
                                     [1.0 * c, 0.]]
            connectivity[:, :, 1] = [[0., 0.4 * c],        # GABAA
                                     [0., fb[1] * c]]
            connectivity[:, :, 2] = [[0., 0.4 * c],        # GABAB
                                     [0., fb[1] * c]]

        # synapses
        synapse_types = ['AMPACurrentSynapse', 'GABAACurrentSynapse', 'GABABCurrentSynapse']
        synapse_classes = ['DoubleExponentialSynapse', 'DoubleExponentialSynapse', 'TransformedInputSynapse']
        if not synapse_params:
            ampa_params = {'efficacy': 0.006,
                           'tau_rise': 0.0077,
                           'tau_decay': 0.017}
            gabaa_params = {'efficacy': -0.001,
                            'tau_rise': 0.0077,
                            'tau_decay': 0.025}
            gabab_params = {'efficacy': -0.018,
                            'tau_rise': 0.07,
                            'tau_decay': 0.125}
            synapse_params = [ampa_params, gabaa_params, gabab_params]

        # axons
        axon_types = ['SuffczynskiAxon', 'SuffczynskiAxon']
        axon_class = 'BurstingAxon'
        if not axon_params:
            tcr_params = {'bin_size': step_size,
                          'epsilon': epsilon,
                          'max_delay': max_synaptic_delay,
                          'tau_rise': 0.05,
                          'tau_decay': 0.1,
                          'max_firing_rate': 800.,
                          'activation_threshold': 0.006,
                          'activation_steepness': 670.,
                          'lts_threshold': -0.016,
                          'lts_steepness': 170.}
            re_params = {'bin_size': step_size,
                         'epsilon': epsilon,
                         'max_delay': max_synaptic_delay,
                         'tau_rise': 0.05,
                         'tau_decay': 0.1,
                         'max_firing_rate': 800.,
                         'activation_threshold': 0.016,
                         'activation_steepness': 670.,
                         'lts_threshold': -0.006,
                         'lts_steepness': 170.}
            axon_params = [tcr_params, re_params]

        # initial condition
        if init_states is None:
            init_states = np.zeros(n_populations)

        # delays
        if delays is None:
            delays = np.zeros((n_populations, n_populations))
        else:
            delays = np.array(delays / step_size, dtype=int)

        # populations
        TCR = SecondOrderPopulation(synapses=synapse_types,
                                    axon=axon_types[0],
                                    init_state=init_states[0],
                                    step_size=step_size,
                                    max_synaptic_delay=max_synaptic_delay,
                                    synapse_params=synapse_params,
                                    axon_params=axon_params[0],
                                    synapse_class=synapse_classes,
                                    axon_class=axon_class,
                                    label=population_labels[0])

        RE = SecondOrderPopulation(synapses=synapse_types[0:2],
                                   axon=axon_types[1],
                                   init_state=init_states[1],
                                   step_size=step_size,
                                   max_synaptic_delay=max_synaptic_delay,
                                   synapse_params=synapse_params[0:2],
                                   axon_params=axon_params[1],
                                   synapse_class=synapse_classes[0:2],
                                   axon_class=axon_class,
                                   label=population_labels[1])

        # call super init
        #################

        super().__init__(populations=[TCR, RE],
                         connectivity=connectivity,
                         delays=delays,
                         step_size=step_size)


###############
# tryout area #
###############

th = Thalamus(max_synaptic_delay=0.5)
synaptic_input = np.zeros((1000, 2, 3))
synaptic_input[:, 0, 0] = 10 * np.random.randn(1000) + 100.
th.run(synaptic_input, 1.)
th.plot_population_states()


