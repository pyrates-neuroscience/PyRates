"""Contains overall BGTC network class plus microcircuit classes for the network nodes
"""

__author__ = 'Richard Gast'

import numpy as np
from core.circuit import Circuit, CircuitFromScratch, CircuitFromCircuit
from core.population import SecondOrderPopulation, PlasticPopulation, JansenRitPyramidalCells, JansenRitInterneurons
from core.utility import update_param


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
                 axon_params=None, init_states=None, resting_potentials=None):
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
            connectivity[:, :, 0] = [[fb[0] * c, 0.],      # AMPA
                                     [1.0 * c, 0.]]
            connectivity[:, :, 1] = [[0., 0.4 * c],        # GABAA
                                     [0., fb[1] * c]]
            connectivity[:, :, 2] = [[0., 0.4 * c],        # GABAB
                                     [0., fb[1] * c]]
        # resting potentials
        if not resting_potentials:
            resting_potentials = [-0.065, -0.070]

        # synapses
        synapse_types = ['AMPACurrentSynapse', 'GABAACurrentSynapse', 'GABABCurrentSynapse']
        synapse_classes = ['DoubleExponentialSynapse', 'DoubleExponentialSynapse', 'TransformedInputSynapse']

        ampa_params = {'efficacy': 0.006,
                       'tau_rise': 1./130,
                       'tau_decay': 1./60}
        gabaa_params = {'efficacy': -0.001,
                        'tau_rise': 1./130,
                        'tau_decay': 1./40}
        gabab_params = {'efficacy': -0.018,
                        'tau_rise': 1./15,
                        'tau_decay': 1./8,
                        'threshold': 11.,
                        'steepness': -0.01}
        synapse_params_tmp = [ampa_params, gabaa_params, gabab_params]

        if synapse_params:
            for i, syn in enumerate(synapse_params):
                for key, val in syn.items():
                    synapse_params_tmp[i][key] = val

        # axons
        axon_types = ['SuffczynskiAxon', 'SuffczynskiAxon']
        axon_class = 'BurstingAxon'

        tcr_params = {'bin_size': step_size,
                      'epsilon': epsilon,
                      'max_delay': max_synaptic_delay,
                      'resting_potential': resting_potentials[0],
                      'tau_rise': 1./20,
                      'tau_decay': 1/10.,
                      'max_firing_rate': 800.,
                      'activation_threshold': resting_potentials[0]+0.006,
                      'activation_steepness': -0.0015,
                      'inactivation_threshold': resting_potentials[0]-0.016,
                      'inactivation_steepness': 0.006}
        re_params = {'bin_size': step_size,
                     'epsilon': epsilon,
                     'max_delay': max_synaptic_delay,
                     'resting_potential': resting_potentials[1],
                     'tau_rise': 1./20,
                     'tau_decay': 1/10.,
                     'max_firing_rate': 800.,
                     'activation_threshold': resting_potentials[1]+0.016,
                     'activation_steepness': -0.0015,
                     'inactivation_threshold': resting_potentials[1]-0.006,
                     'inactivation_steepness': 0.006}
        axon_params_tmp = [tcr_params, re_params]

        if axon_params:
            for i, ax in enumerate(axon_params):
                for key, val in ax.items():
                    axon_params_tmp[i][key] = val

        # initial condition
        if init_states is None:
            init_states = np.array(resting_potentials)

        # delays
        if delays is None:
            delays = np.zeros((n_populations, n_populations))

        # populations
        TCR = SecondOrderPopulation(synapses=synapse_types,
                                    axon=axon_types[0],
                                    init_state=init_states[0],
                                    step_size=step_size,
                                    max_synaptic_delay=max_synaptic_delay,
                                    synapse_params=synapse_params_tmp,
                                    axon_params=axon_params_tmp[0],
                                    synapse_class=synapse_classes,
                                    axon_class=axon_class,
                                    label=population_labels[0],
                                    resting_potential=resting_potentials[0])

        RE = SecondOrderPopulation(synapses=synapse_types[0:2],
                                   axon=axon_types[1],
                                   init_state=init_states[1],
                                   step_size=step_size,
                                   max_synaptic_delay=max_synaptic_delay,
                                   synapse_params=synapse_params_tmp[0:2],
                                   axon_params=axon_params_tmp[1],
                                   synapse_class=synapse_classes[0:2],
                                   axon_class=axon_class,
                                   label=population_labels[1],
                                   resting_potential=resting_potentials[1])

        # call super init
        #################

        super().__init__(populations=[TCR, RE],
                         connectivity=connectivity,
                         delays=delays,
                         step_size=step_size,
                         synapse_types=['AMPA', 'GABAA', 'GABAB'])
