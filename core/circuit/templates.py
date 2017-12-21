"""Templates for specific ciricuit parametrizations.
"""

import numpy as np

from core.circuit import CircuitFromPopulations, CircuitFromScratch
from typing import Optional

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


class JansenRitCircuit(CircuitFromPopulations):
    """Basic Jansen-Rit circuit as defined in [1]_.

    Parameters
    ----------
    resting_potential
        Default = -0.075 V.
    tau_leak
        Default = 0.016 s.
    membrane_capacitance
        Default = 1e-12 q/S.
    step_size
        Default = 5e-4 s.
    max_synaptic_delay
        Default = 0.05 s.
    delays
        Default = None
    init_states
        Default = np.zeros(3)

    See Also
    --------
    :class:`CircuitFromPopulations`: Detailed description of parameters.
    :class:`Circuit`: Detailed description of attributes and methods.

    References
    ----------
    .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
       of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

    """

    def __init__(self, resting_potential: float=-0.075, tau_leak: float=0.016, membrane_capacitance: float=1e-12,
                 step_size: float=5e-4, max_synaptic_delay: float=0.05, delays: np.ndarray=None,
                 init_states: np.ndarray=np.zeros(3)) -> None:
        """Initializes a basic Jansen-Rit circuit of pyramidal cells, excitatory interneurons and inhibitory interneurons.
        For detailed parameter description see super class.
        """

        ##################
        # set parameters #
        ##################

        populations = ['JansenRitPyramidalCells',
                       'JansenRitExcitatoryInterneurons',
                       'JansenRitInhibitoryInterneurons']

        population_labels = ['JR_PCs',
                             'JR_EINs',
                             'JR_IINs']

        N = 3                                               # PCs, EINs, IIns
        n_synapses = 2                                      # AMPA and GABAA

        ###################
        # set connections #
        ###################

        connections = np.zeros((N, N, n_synapses))
        C = 135

        # AMPA connections (excitatory)
        connections[:, :, 0] = [[0, 0.8 * C, 0], [1.0 * C, 0, 0], [0.25 * C, 0, 0]]

        # GABA-A connections (inhibitory)
        connections[:, :, 1] = [[0, 0, 0.25 * C], [0, 0, 0], [0, 0, 0]]

        ###################
        # call super init #
        ###################

        super().__init__(population_types=populations,
                         connectivity=connections,
                         delays=delays,
                         population_labels=population_labels,
                         resting_potential=resting_potential,
                         tau_leak=tau_leak,
                         membrane_capacitance=membrane_capacitance,
                         step_size=step_size,
                         max_synaptic_delay=max_synaptic_delay,
                         init_states=init_states)


class WangKnoescheCircuit(CircuitFromScratch):
    """Basic cortical column circuit as defined in [1]_, using the axon defined in [2]_.

    Parameters
    ----------
    resting_potential
        Default = -0.075 V.
    tau_leak
        Default = 0.016 s.
    membrane_capacitance
        Default = 1e-12 q/S.
    step_size
        Default = 5e-4 s.
    max_synaptic_delay
        Default = 0.05 s.
    delays
        Default = None
    init_states
        Default = np.zeros(3)

    See Also
    --------
    :class:`CircuitFromPopulations`: Detailed description of parameters.
    :class:`Circuit`: Detailed description of attributes and methods.

    References
    ----------
    .. [1] P. Wang & T.R. Knoesche, "A realistic neural mass model of the cortex with laminar-specific connections and
       synaptic plasticity-evaluation with auditory habituation." PloS one, vol. 8(10): e77876, 2013.
    .. [2] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
       of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

    """

    def __init__(self, resting_potential: float=-0.075, tau_leak: float=0.016, membrane_capacitance: float=1e-12,
                 step_size: float=5e-4, max_synaptic_delay: float=0.05, delays: np.ndarray=None,
                 init_states: Optional[np.ndarray]=None) -> None:
        """Initializes a basic Wang-Knoesche circuit of pyramidal cells and inhibitory interneurons in layer 2/3 as well
        as layer 5/6 plus a layer 4 excitatory interneuron population.
        """

        ##################
        # set parameters #
        ##################

        # population information
        population_labels = ['L23_PCs',
                             'L23_IINs',
                             'L4_EINs',
                             'L56_PCs',
                             'L56_IINs']
        N = 5                                               # L23 PCs+IINs, L4 EINs, L56 PCs+IIns

        # synapse information
        n_synapses = 2                                      # AMPA and GABAA
        tau_depression = 0.05                               # synaptic depression time constant [s]
        tau_recycle = 0.5                                   # synaptic neurotransmitter recycling time constant [s]
        max_synaptic_delay = max_synaptic_delay             # see parameter documentation
        synapses = ['AMPACurrentSynapse',                   # synapse types
                    'GABAACurrentSynapse']
        synapse_params = {'tau_depression': tau_depression,
                          'tau_recycle': tau_recycle}
        synapse_params = [synapse_params, None]          # one dictionary per synapse type

        # connectivity matrix
        connections = np.zeros((N, N, n_synapses))

        connections[:, :, 0] = [[0., 0., 108., 0., 0.],     # AMPA connections (excitatory)
                                [33.75, 0., 0., 0., 0.],
                                [0., 0., 0., 135., 0.],
                                [135., 0., 0., 0., 0.],
                                [0., 0., 0., 33.75, 0.]]

        connections[:, :, 1] = [[0., 33.75, 0., 0., 0.],    # GABA-A connections (inhibitory)
                                [0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 33.75],
                                [0., 0., 0., 0., 0.]]

        # axon information
        axons = ['JansenRitAxon' for i in range(N)]         # use Jansen-Rit parametrization of axon ([2]_)

        # initial states
        if init_states is None:
            init_states = np.zeros(N) - 0.075

        ###################
        # call super init #
        ###################

        super().__init__(connectivity=connections,
                         delays=delays,
                         step_size=step_size,
                         synapses=synapses,
                         synapse_params=synapse_params,
                         axons=axons,
                         max_synaptic_delay=max_synaptic_delay,
                         membrane_capacitance=membrane_capacitance,
                         tau_leak=tau_leak,
                         resting_potential=resting_potential,
                         init_states=init_states,
                         population_labels=population_labels,
                         plastic_populations=True)


class MoranCircuit(CircuitFromScratch):
    """Basic cortical column circuit as defined in [1]_.

    Parameters
    ----------
    resting_potential
        Default = -0.075 V.
    tau_leak
        Default = 0.016 s.
    membrane_capacitance
        Default = 1e-12 q/S.
    step_size
        Default = 5e-4 s.
    delays
        Default = None
    init_states
        Default = np.zeros(3)

    See Also
    --------
    :class:`CircuitFromPopulations`: Detailed description of parameters.
    :class:`Circuit`: Detailed description of attributes and methods.

    References
    ----------
    .. [1] P. Wang & T.R. Knoesche, "A realistic neural mass model of the cortex with laminar-specific connections and
       synaptic plasticity-evaluation with auditory habituation." PloS one, vol. 8(10): e77876, 2013.

    """

    def __init__(self, resting_potential: float=-0.075, tau_leak: float=0.016, membrane_capacitance: float=1e-12,
                 step_size: float=5e-4, delays: np.ndarray=None, epsilon: float=1e-14,
                 init_states: Optional[np.ndarray]=None) -> None:
        """Initializes a basic Moran circuit of pyramidal cells, excitatory and inhibitory interneurons.
        """

        ##################
        # set parameters #
        ##################

        # population information
        population_labels = ['PCs',
                             'EINs',
                             'IINs']
        N = 3

        # synapse information
        n_synapses = 2                                      # AMPA and GABAA
        synapses = ['AMPACurrentSynapse',                   # synapse types
                    'GABAACurrentSynapse']
        synapse_params = {'epsilon': epsilon}
        synapse_params = [synapse_params, synapse_params]   # one dictionary per synapse type

        # connectivity matrix
        connections = np.zeros((N, N, n_synapses))

        connections[:, :, 0] = [[0., 128., 0.],     # AMPA connections (excitatory)
                                [128., 0.,  0.],
                                [64., 0., 0.]]

        connections[:, :, 1] = [[0., 0., 64.],    # GABA-A connections (inhibitory)
                                [0., 0., 0.],
                                [0., 0., 16.]]

        # axon information
        axons = ['JansenRitAxon' for i in range(N)]    # use axon as defined in ([2]_)

        axon_params = {'tau': 0.512,
                       'firing_rate_target': 2.5}
        axon_params = [None, axon_params, None]

        # initial states
        if init_states is None:
            init_states = np.zeros(N)

        ###################
        # call super init #
        ###################

        super().__init__(connectivity=connections,
                         delays=delays,
                         step_size=step_size,
                         synapses=synapses,
                         synapse_params=synapse_params,
                         axons=axons,
                         axon_params=axon_params,
                         membrane_capacitance=membrane_capacitance,
                         tau_leak=tau_leak,
                         resting_potential=resting_potential,
                         init_states=init_states,
                         population_labels=population_labels,
                         plastic_populations=True,
                         axon_class='SigmoidAxon')
