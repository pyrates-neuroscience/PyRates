"""Contains overall BGTC network class plus microcircuit classes for the network nodes
"""

__author__ = 'Richard Gast'

import numpy as np
from core.circuit import Circuit, CircuitFromScratch, CircuitFromCircuit
from core.population import Population, PlasticPopulation, JansenRitPyramidalCells, JansenRitInterneurons


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
            fb = np.zeros(3) if feedback is None else feedback
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


###############
# tryout area #
###############

fb = np.zeros(3)
fb[0] = 0.2
m1 = M1(feedback=fb)
synaptic_input = np.zeros((1000, 3, 2))
synaptic_input[:, 0, 0] = 22 * np.random.randn(1000) + 220.
m1.run(synaptic_input, 1.)
m1.plot_population_states()
