"""Contains overall BGTC network class plus microcircuit classes for the network nodes
"""

__author__ = 'Richard Gast'

import numpy as np
from core.circuit import Circuit, CircuitFromScratch, CircuitFromCircuit
from core.population import SecondOrderPopulation, PlasticPopulation, JansenRitPyramidalCells, JansenRitInterneurons
from core.utility import update_param
from typing import Optional, List


#####################
# class definitions #
#####################


class BGTC(CircuitFromCircuit):
    """Model of the basal ganglia-thalamocortical network.
    """

    def __init__(self,
                 connection_strengths: Optional[np.ndarray] = None,
                 delays: Optional[np.ndarray] = None,
                 source_populations: Optional[list] = None,
                 target_populations: Optional[list] = None,
                 target_synapses: Optional[list] = None,
                 step_size: float = 1e-3,
                 max_synaptic_delay: float = 0.5
                 ):
        """Instantiates basal-ganglia-thalamocortical circuit.
        """

        # set parameters
        ################

        # connectivity
        if connection_strengths is None:
            connection_strengths = [110., 588., 672., 127.]
        if source_populations is None:
            source_populations = [8, 8, 2, 2]
        if target_populations is None:
            target_populations = [2, 1, 4, 6]
        if target_synapses is None:
            target_synapses = [0, 0, 0, 0]

        # delays
        if delays is None:
            delays = [8e-3, 8e-3, 8e-3, 8e-3]

        # circuits
        m1 = M1(step_size=step_size, max_synaptic_delay=max_synaptic_delay)
        bg_th = BasalGanglia(step_size=step_size, max_synaptic_delay=max_synaptic_delay)

        # call super init
        #################

        super().__init__(circuits=[m1, bg_th],
                         connection_strengths=connection_strengths,
                         source_populations=source_populations,
                         target_populations=target_populations,
                         target_synapses=target_synapses,
                         delays=delays,
                         circuit_labels=['M1', 'BG_TH'])


class M1(Circuit):
    """Model of the primary motor cortex (M1/BA4).
    """

    def __init__(self,
                 step_size=1e-3,
                 max_synaptic_delay=1.0,
                 connectivity=None,
                 delays=None,
                 synapse_params=None,
                 synapse_types=None,
                 synapse_class='ExponentialSynapse',
                 axon_params=None,
                 axon_types=None,
                 axon_class='SigmoidAxon',
                 init_states=None
                 ):

        # set parameters
        ################

        n_populations = 4
        n_synapses = 2
        population_labels = ['S_PCs', 'M_PCs', 'D_PCs', 'IINs']

        # connection strengths
        if connectivity is None:
            connection_strengths = np.array([357., 872., 387., 340., 311., 405., 377., 429., 331., 403., 753.,
                                             376., 382., 414.])
            connectivity = np.zeros((n_populations, n_populations, n_synapses))
            connectivity[:, :, 0] = [[0., connection_strengths[1], connection_strengths[13], 0.],
                                     [connection_strengths[7], 0., 0., 0.],
                                     [connection_strengths[10], 0., 0., 0.],
                                     [connection_strengths[12], connection_strengths[4], connection_strengths[5], 0.]]
            connectivity[:, :, 1] = [[connection_strengths[6], 0., 0., connection_strengths[11]],
                                     [0., connection_strengths[0], 0., connection_strengths[2]],
                                     [0., 0., connection_strengths[9], connection_strengths[8]],
                                     [0., 0., 0., connection_strengths[3]]]

        # delays
        if delays is None:
            delays = np.ones((n_populations, n_populations)) * 1e-3

        # synapses
        if not synapse_types:
            synapse_types = ['excitatory', 'inhibitory']
        if not synapse_class:
            synapse_class = 'ExponentialSynapse'
        if not synapse_params:
            efficacy = 1e-3
            synapse_params_sp = [{'efficacy': efficacy, 'tau': 3.2e-3}, {'efficacy': -efficacy, 'tau': 3.2e-3}]
            synapse_params_mp = [{'efficacy': efficacy, 'tau': 3.7e-3}, {'efficacy': -efficacy, 'tau': 3.7e-3}]
            synapse_params_dp = [{'efficacy': efficacy, 'tau': 10.6e-3}, {'efficacy': -efficacy, 'tau': 10.6e-3}]
            synapse_params_iin = [{'efficacy': efficacy, 'tau': 14.1e-3}, {'efficacy': -efficacy, 'tau': 14.1e-3}]
            synapse_params = [synapse_params_sp, synapse_params_mp, synapse_params_dp, synapse_params_iin]
        elif type(synapse_params[0]) is dict:
            synapse_params = [synapse_params for _ in range(n_populations)]

        # axons
        if not axon_types:
            axon_types = [None for _ in range(n_populations)]
        if not axon_params:
            threshold = 0.
            normalize = True
            max_firing_rate = 1.
            steepness = 2/3 * 1e3
            axon_params = [{'membrane_potential_threshold': threshold,
                            'normalize': normalize,
                            'max_firing_rate': max_firing_rate,
                            'sigmoid_steepness': steepness} for _ in range(n_populations)]

        # initial condition
        if init_states is None:
            init_states = np.zeros(n_populations)

        # instantiate populations
        #########################

        SPs = SecondOrderPopulation(synapses=None, axon=axon_types[0], init_state=init_states[0],
                                    step_size=step_size, max_synaptic_delay=max_synaptic_delay,
                                    synapse_params=synapse_params[0], axon_params=axon_params[0],
                                    synapse_class=synapse_class, axon_class=axon_class, label=population_labels[0])
        MPs = SecondOrderPopulation(synapses=None, axon=axon_types[1], init_state=init_states[1],
                                    step_size=step_size, max_synaptic_delay=max_synaptic_delay,
                                    synapse_params=synapse_params[1], axon_params=axon_params[1],
                                    synapse_class=synapse_class, axon_class=axon_class, label=population_labels[1])
        DPs = SecondOrderPopulation(synapses=None, axon=axon_types[2], init_state=init_states[2],
                                    step_size=step_size, max_synaptic_delay=max_synaptic_delay,
                                    synapse_params=synapse_params[2], axon_params=axon_params[2],
                                    synapse_class=synapse_class, axon_class=axon_class, label=population_labels[2])
        IINs = SecondOrderPopulation(synapses=None, axon=axon_types[3], init_state=init_states[3],
                                     step_size=step_size, max_synaptic_delay=max_synaptic_delay,
                                     synapse_params=synapse_params[3], axon_params=axon_params[3],
                                     synapse_class=synapse_class, axon_class=axon_class, label=population_labels[3])

        # call super init
        #################

        super().__init__(populations=[SPs, MPs, DPs, IINs],
                         connectivity=connectivity,
                         delays=delays,
                         step_size=step_size,
                         synapse_types=synapse_types
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


class BasalGanglia(Circuit):
    """Basal Ganglia circuit consisting of STR, GPe, STN, GPi/SNr and Thalamus.
    """

    def __init__(self,
                 step_size: float = 1e-3,
                 max_synaptic_delay: float = 0.5,
                 connectivity: Optional[np.ndarray] = None,
                 delays: Optional[np.ndarray] = None,
                 synapse_types: Optional[List[str]] = None,
                 synapse_params: Optional[list] = None,
                 synapse_class: str = 'ExponentialSynapse',
                 axon_types: Optional[List[str]] = None,
                 axon_params: Optional[List[dict]] = None,
                 axon_class: str = 'SigmoidAxon',
                 init_states: Optional[np.ndarray] = None
                 ):
        """Instantiates BG circuit.
        """

        # set parameters
        ################

        n_populations = 5
        n_synapses = 2
        population_labels = ['STR', 'GPe', 'STN', 'GPi/SNr', 'Tha']

        # connection strengths
        if connectivity is None:
            connection_strengths = np.array([962., 828., 1403., 719., 526., 568., 345., 780., 301.])
            connectivity = np.zeros((n_populations, n_populations, n_synapses))
            connectivity[:, :, 0] = [[0., 0., 0., 0., 0.],
                                     [0., 0., connection_strengths[4], 0., 0.],
                                     [0., 0., 0., 0., 0.],
                                     [0., 0., connection_strengths[6], 0., 0.],
                                     [0., 0., 0., 0., 0.]]
            connectivity[:, :, 1] = [[connection_strengths[0], 0., 0., 0., 0.],
                                     [connection_strengths[1], connection_strengths[2], 0., 0., 0.],
                                     [0., connection_strengths[3], 0., 0., 0.],
                                     [connection_strengths[5], 0., 0., connection_strengths[7], 0.],
                                     [0., 0., 0., connection_strengths[8], 0.]]

        # delays
        if delays is None:
            delays = np.ones((n_populations, n_populations)) * 4e-3

        # synapses
        if not synapse_types:
            synapse_types = ['excitatory', 'inhibitory']
        if not synapse_class:
            synapse_class = 'ExponentialSynapse'
        if not synapse_params:
            efficacy = 1e-3
            synapse_params_str = [{'efficacy': efficacy, 'tau': 9.3e-3}, {'efficacy': -efficacy, 'tau': 9.3e-3}]
            synapse_params_gpe = [{'efficacy': efficacy, 'tau': 12.2e-3}, {'efficacy': -efficacy, 'tau': 12.2e-3}]
            synapse_params_stn = [{'efficacy': efficacy, 'tau': 3.5e-3}, {'efficacy': -efficacy, 'tau': 3.5e-3}]
            synapse_params_gpi = [{'efficacy': efficacy, 'tau': 12.1e-3}, {'efficacy': -efficacy, 'tau': 12.1e-3}]
            synapse_params_tha = [{'efficacy': efficacy, 'tau': 10.1e-3}, {'efficacy': -efficacy, 'tau': 10.1e-3}]
            synapse_params = [synapse_params_str, synapse_params_gpe, synapse_params_stn, synapse_params_gpi,
                              synapse_params_tha]
        elif type(synapse_params[0]) is dict:
            synapse_params = [synapse_params for _ in range(n_populations)]

        # axons
        if not axon_types:
            axon_types = [None for _ in range(n_populations)]
        if not axon_params:
            threshold = 0.
            normalize = True
            max_firing_rate = 1.
            steepness = 2 / 3 * 1e3
            axon_params = [{'membrane_potential_threshold': threshold,
                            'normalize': normalize,
                            'max_firing_rate': max_firing_rate,
                            'sigmoid_steepness': steepness} for _ in range(n_populations)]

        # initial condition
        if init_states is None:
            init_states = np.zeros(n_populations)

        # instantiate populations
        #########################

        STR = SecondOrderPopulation(synapses=None, axon=axon_types[0], init_state=init_states[0],
                                    step_size=step_size, max_synaptic_delay=max_synaptic_delay,
                                    synapse_params=synapse_params[0], axon_params=axon_params[0],
                                    synapse_class=synapse_class, axon_class=axon_class, label=population_labels[0])
        GPe = SecondOrderPopulation(synapses=None, axon=axon_types[1], init_state=init_states[1],
                                    step_size=step_size, max_synaptic_delay=max_synaptic_delay,
                                    synapse_params=synapse_params[1], axon_params=axon_params[1],
                                    synapse_class=synapse_class, axon_class=axon_class, label=population_labels[1])
        STN = SecondOrderPopulation(synapses=None, axon=axon_types[2], init_state=init_states[2],
                                    step_size=step_size, max_synaptic_delay=max_synaptic_delay,
                                    synapse_params=synapse_params[2], axon_params=axon_params[2],
                                    synapse_class=synapse_class, axon_class=axon_class, label=population_labels[2])
        GPi = SecondOrderPopulation(synapses=None, axon=axon_types[3], init_state=init_states[3],
                                    step_size=step_size, max_synaptic_delay=max_synaptic_delay,
                                    synapse_params=synapse_params[3], axon_params=axon_params[3],
                                    synapse_class=synapse_class, axon_class=axon_class, label=population_labels[3])
        Tha = SecondOrderPopulation(synapses=None, axon=axon_types[4], init_state=init_states[4],
                                    step_size=step_size, max_synaptic_delay=max_synaptic_delay,
                                    synapse_params=synapse_params[4], axon_params=axon_params[4],
                                    synapse_class=synapse_class, axon_class=axon_class, label=population_labels[4])

        # call super init
        #################

        super().__init__(populations=[STR, GPe, STN, GPi, Tha],
                         connectivity=connectivity,
                         delays=delays,
                         step_size=step_size,
                         synapse_types=synapse_types
                         )
