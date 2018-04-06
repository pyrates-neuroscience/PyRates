"""Templates for specific circuit parametrizations.
"""

import numpy as np
from typing import Optional, List

from pyrates.circuit import CircuitFromScratch, Circuit
from pyrates.population import WangKnoescheCells
from pyrates.population import MoranPyramidalCells, MoranExcitatoryInterneurons, MoranInhibitoryInterneurons

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


#######################
# jansen-rit circuits #
#######################


class JansenRitLeakyCapacitorCircuit(CircuitFromScratch):
    """Jansen-Rit circuit as defined in [1]_, implemented as a leaky capacitor.

    Parameters
    ----------
    step_size
        Default = 5e-4 s.
    max_synaptic_delay
        Default = None.
    delays
        Default = None
    connectivity_scaling
        Default = 135.
    feedback_strength
        Default = np.zeros(3)
    init_states
        Default = np.zeros(3) - 0.075
    synapse_params
        Default = None.
    axon_params
        Default = None.
    resting_potential
        Default = -0.075 V.
    tau_leak
        Default = 0.016 s.
    membrane_capacitance
        Default = 1e-12 q/V.
    conductance_based
        Default = False
        
    See Also
    --------
    :class:`CircuitFromScratch`: Detailed description of parameters.
    :class:`Circuit`: Detailed description of attributes and methods.

    References
    ----------
    .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
       of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

    """

    def __init__(self,
                 step_size: float = 1e-4,
                 max_synaptic_delay: Optional[float] = None,
                 delays: Optional[np.ndarray] = None,
                 connectivity_scaling: float = 135.,
                 feedback_strength: np.ndarray = np.zeros(3),
                 init_states: np.ndarray = np.zeros(3) - 0.075,
                 synapse_params: Optional[List[dict]] = None,
                 axon_params: Optional[List[dict]] = None,
                 resting_potential: float = -0.075,
                 tau_leak: float = 0.016,
                 membrane_capacitance: float = 1e-12,
                 conductance_based: bool = False,
                 ) -> None:
        """Initializes a leaky capacitor version of a Jansen-Rit circuit.
        """

        # set parameters
        ################

        # general
        population_labels = ['LC_PCs',
                             'LC_EINs',
                             'LC_IINs']

        N = 3                                               # PCs, EINs, IIns
        n_synapses = 2                                      # excitatory and inhibitory

        # connectivity
        connections = np.zeros((N, N, n_synapses))
        c = connectivity_scaling
        fb = feedback_strength

        connections[:, :, 0] = [[fb[0] * c, 0.8 * c, 0],  # excitatory connections
                                [1.0 * c, fb[1] * c, 0],
                                [0.25 * c, 0, 0]]

        connections[:, :, 1] = [[0, 0, 0.25 * c],         # inhibitory connections
                                [0, 0, 0],
                                [0, 0, fb[2] * c]]

        # delays
        if delays is None:
            delays = np.zeros((N, N))

        # synapses
        if conductance_based:
            synapse_types = ['AMPAConductanceSynapse', 'GABAAConductanceSynapse']
        else:
            synapse_types = ['AMPACurrentSynapse', 'GABAACurrentSynapse']
        synapse_class = 'DoubleExponentialSynapse'

        # axon
        axon_types = ['KnoescheAxon', 'KnoescheAxon', 'KnoescheAxon']
        axon_class = 'SigmoidAxon'

        # call super init
        #################

        super().__init__(connectivity=connections,
                         delays=delays,
                         synapses=synapse_types,
                         synapse_params=synapse_params,
                         synapse_class=synapse_class,
                         synapse_types=['AMPA', 'GABAA'],
                         axons=axon_types,
                         axon_params=axon_params,
                         axon_class=axon_class,
                         population_labels=population_labels,
                         step_size=step_size,
                         max_synaptic_delay=max_synaptic_delay,
                         init_states=init_states,
                         resting_potential=resting_potential,
                         tau_leak=tau_leak,
                         membrane_capacitance=membrane_capacitance
                         )


class JansenRitCircuit(CircuitFromScratch):
    """Jansen-Rit circuit as defined in [1]_ with optional self-feedback loops at each population (motivated by [2]_).

    Parameters
    ----------
    step_size
        Default = 5e-4 s.
    max_synaptic_delay
        Default = None.
    delays
        Default = None
    connectivity_scaling
        Default = 135.
    feedback_strength
        Default = np.zeros(3)
    init_states
        Default = np.zeros(3)
    synapse_params
        Default = None.
    axon_params
        Default = None.

    See Also
    --------
    :class:`CircuitFromPopulations`: Detailed description of parameters.
    :class:`Circuit`: Detailed description of attributes and methods.

    References
    ----------
    .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
       of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.
    .. [2] V. Youssofzadeh, G. Prasad & K.F. Wong-Lin, "On self-feedback connectivity in neural mass models applied to
       event-related potentials." NeuroImage, vol. 108, pp. 364-376, 2015.

    """

    def __init__(self,
                 step_size: float = 5e-4,
                 max_synaptic_delay: Optional[float] = None,
                 delays: Optional[np.ndarray] = None,
                 connectivity_scaling: float = 135.,
                 feedback_strength: np.ndarray = np.zeros(3),
                 init_states: np.ndarray = np.zeros(3),
                 synapse_params: Optional[List[dict]] = None,
                 axon_params: Optional[List[dict]] = None,
                 ) -> None:
        """Initializes a basic Jansen-Rit circuit of pyramidal cells, excitatory interneurons and inhibitory
        interneurons.
        """

        # set parameters
        ################

        # general
        population_labels = ['JR_PCs',
                             'JR_EINs',
                             'JR_IINs']

        N = 3                                               # PCs, EINs, IIns
        n_synapses = 2                                      # excitatory and inhibitory

        # connectivity
        connections = np.zeros((N, N, n_synapses))
        c = connectivity_scaling
        fb = feedback_strength

        connections[:, :, 0] = [[fb[0] * c, 0.8 * c, 0],    # excitatory connections
                                [1.0 * c, fb[1] * c, 0],
                                [0.25 * c, 0, 0]]

        connections[:, :, 1] = [[0, 0, 0.25 * c],           # inhibitory connections
                                [0, 0, 0],
                                [0, 0, fb[2] * c]]

        # synapses
        synapse_types = ['JansenRitExcitatorySynapse', 'JansenRitInhibitorySynapse']
        synapse_class = 'ExponentialSynapse'

        # axon
        axon_types = ['JansenRitAxon', 'JansenRitAxon', 'JansenRitAxon']
        axon_class = 'SigmoidAxon'

        # call super init
        #################

        super().__init__(connectivity=connections,
                         delays=delays,
                         synapses=synapse_types,
                         synapse_params=synapse_params,
                         synapse_class=synapse_class,
                         synapse_types=['excitatory', 'inhibitory'],
                         axons=axon_types,
                         axon_params=axon_params,
                         axon_class=axon_class,
                         population_labels=population_labels,
                         step_size=step_size,
                         max_synaptic_delay=max_synaptic_delay,
                         init_states=init_states
                         )


class GeneralizedJansenRitCircuit(Circuit):
    """Jansen-Rit circuit as described in [1]_, generalized to multiple sub-circuits as described in [2]_.

    Parameters
    ----------
    step_size
        Default = 5e-4 s.
    max_synaptic_delay
        Default = None.
    delays
        Default = None
    connectivity_scalings
        Default = 135.
    feedback_strengths
        Default = None
    init_states
        Default = None
    synapse_params
        Default = None.
    axon_params
        Default = None.

    See Also
    --------
    :class:`CircuitFromPopulations`: Detailed description of parameters.
    :class:`Circuit`: Detailed description of attributes and methods.

    References
    ----------
    .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
       of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

    """
    def __init__(self,
                 n_circuits: int,
                 synapse_params: Optional[List[List[dict]]] = None,
                 axon_params: Optional[List[List[dict]]] = None,
                 connectivity_scalings: Optional[List[float]] = None,
                 feedback_strengths: Optional[List[np.ndarray]] = None,
                 weights: Optional[List[float]] = None,
                 delays: Optional[np.ndarray] = None,
                 delay_distributions: Optional[np.ndarray] = None,
                 step_size: float = 5e-4,
                 max_synaptic_delay: Optional[float] = None,
                 init_states: Optional[List[np.ndarray]] = None
                 ):
        """Instantiates multi-circuit Jansen-Rit NMM.
        """

        # single-circuit information
        ############################

        n_populations = 3
        n_synapses = 2

        # check input arguments
        #######################

        if not synapse_params:
            synapse_params = [None for _ in range(n_circuits)]
        if not axon_params:
            axon_params = [None for _ in range(n_circuits)]
        if not connectivity_scalings:
            connectivity_scalings = [135 for _ in range(n_circuits)]
        if not weights:
            weights = [1/n_circuits for _ in range(n_circuits)]
        if not init_states:
            init_states = [np.zeros(n_populations) for _ in range(n_circuits)]
        if not feedback_strengths:
            feedback_strengths = [np.zeros(n_populations) for _ in range(n_circuits)]
        if delays is None:
            delays = np.zeros((n_circuits * n_populations, n_circuits * n_populations))

        connectivity = np.zeros((n_circuits * n_populations, n_circuits * n_populations, n_synapses))
        populations = list()

        for i in range(n_circuits):

            circuit_tmp = JansenRitCircuit(step_size=step_size,
                                           max_synaptic_delay=max_synaptic_delay,
                                           connectivity_scaling=connectivity_scalings[i],
                                           feedback_strength=feedback_strengths[i],
                                           synapse_params=synapse_params[i],
                                           axon_params=axon_params[i],
                                           init_states=init_states[i])

            connectivity[:, i*circuit_tmp.n_populations:(i + 1) * circuit_tmp.n_populations, :] = \
                np.tile(weights[i] * circuit_tmp.C, (n_circuits, 1, 1))
            for pop in circuit_tmp.populations:
                pop.targets = list()
            populations += circuit_tmp.populations

        super().__init__(populations=populations,
                         connectivity=connectivity,
                         delays=delays,
                         delay_distributions=delay_distributions,
                         step_size=step_size)


#########################################################################
# wang knoesche circuit with synaptic plasticity on excitatory synapses #
#########################################################################


class WangKnoescheCircuit(Circuit):
    """Basic 5-population cortical column circuit as defined in [1]_.

    Parameters
    ----------
    step_size
        Default = 5e-4 s.
    max_synaptic_delay
        Default = None.
    delays
        Default = None
    init_states
        Default = np.zeros(5)
    tau_depression
        Default = 0.05 s.
    tau_recycle
        Default = 0.5 s.
    plastic_synapses
        Optional list that indicates whether plasticity should be enabled at 

    See Also
    --------
    :class:`CircuitFromPopulations`: Detailed description of parameters.
    :class:`Circuit`: Detailed description of attributes and methods.

    References
    ----------
    .. [1] P. Wang & T.R. Knoesche, "A realistic neural mass model of the cortex with laminar-specific connections and
       synaptic plasticity-evaluation with auditory habituation." PloS one, vol. 8(10): e77876, 2013.

    """

    def __init__(self,
                 step_size: float = 5e-4,
                 max_synaptic_delay: Optional[float] = None,
                 delays: Optional[np.ndarray] = None,
                 init_states: np.ndarray = np.zeros(5),
                 tau_depression: float = 0.05,
                 tau_recycle: float = 0.5,
                 plastic_synapses: Optional[List[bool]] = None
                 ) -> None:
        """Initializes a basic Wang-Knoesche circuit of pyramidal cells and inhibitory interneurons in layer 2/3 as well
        as layer 5/6 plus a layer 4 excitatory interneuron population.
        """

        if not plastic_synapses:
            plastic_synapses = [True, False]

        # set parameters
        ################

        # synapse information
        n_synapses = 2

        # populations
        l23_pcs = WangKnoescheCells(step_size=step_size,
                                    init_state=init_states[0],
                                    label='L23_PCs',
                                    tau_depression=tau_depression,
                                    tau_recycle=tau_recycle,
                                    plastic_synapses=plastic_synapses)
        l23_iins = WangKnoescheCells(step_size=step_size,
                                     init_state=init_states[1],
                                     label='L23_IIns',
                                     synapses=['JansenRitExcitatorySynapse'],
                                     plastic_synapses=[plastic_synapses[0]],
                                     tau_depression=tau_depression,
                                     tau_recycle=tau_recycle)
        l4_eins = WangKnoescheCells(step_size=step_size,
                                    init_state=init_states[2],
                                    label='L4_EINs',
                                    synapses=['JansenRitExcitatorySynapse'],
                                    plastic_synapses=[plastic_synapses[0]],
                                    tau_depression=tau_depression,
                                    tau_recycle=tau_recycle)
        l56_pcs = WangKnoescheCells(step_size=step_size,
                                    init_state=init_states[3],
                                    label='L56_PCs',
                                    plastic_synapses=plastic_synapses,
                                    tau_depression=tau_depression,
                                    tau_recycle=tau_recycle)
        l56_iins = WangKnoescheCells(step_size=step_size,
                                     init_state=init_states[4],
                                     label='L56_IINs',
                                     synapses=['JansenRitExcitatorySynapse'],
                                     plastic_synapses=[plastic_synapses[0]],
                                     tau_depression=tau_depression,
                                     tau_recycle=tau_recycle)

        N = 5

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

        # delay matrix
        if delays is None:
            delays = np.zeros((N, N))

        # call super init
        #################

        super().__init__(populations=[l23_pcs, l23_iins, l4_eins, l56_pcs, l56_iins],
                         connectivity=connections,
                         delays=delays,
                         step_size=step_size)


#################################################
# moran circuit with spike-frequency adaptation #
#################################################


class MoranCircuit(Circuit):
    """Basic 3-population cortical column circuit as defined in [1]_.

    Parameters
    ----------
    step_size
        Default = 5e-4 s.
    delays
        Default = None
    tau
        Time-scale of the spike-frequency adaptation [unit = s] (default = 0.512).
    init_states
        Default = np.zeros(3)

    See Also
    --------
    :class:`CircuitFromPopulations`: Detailed description of parameters.
    :class:`Circuit`: Detailed description of attributes and methods.

    References
    ----------
    .. [1] R.J. Moran, S.J. Kiebel, K.E. Stephan, R.B. Reilly, J. Daunizeau & K.J. Friston, "A Neural Mass Model of
       Spectral Responses in Electrophysiology" NeuroImage, vol. 37, pp. 706-720, 2007.

    """

    def __init__(self,
                 step_size: float = 5e-4,
                 delays: Optional[np.ndarray] = None,
                 tau: Optional[float] = 0.512,
                 init_states: np.ndarray = np.zeros(3)
                 ) -> None:
        """Initializes a basic Moran circuit of pyramidal cells (plastic + non-plastic), excitatory and inhibitory
        interneurons.
        """

        # set parameters
        ################

        population_labels = ['PCs_plastic',
                             'PCs_nonplastic',
                             'EINs',
                             'IINs']
        N = 4

        # synapse information
        n_synapses = 2                                  # excitatory and inhibitory

        # connectivity matrix
        connections = np.zeros((N, N, n_synapses))

        connections[:, :, 0] = [[0., 0., 128., 0.],     # excitatory connections
                                [0., 0., 128., 0.],
                                [128., 0., 0.,  0.],
                                [0., 64., 0., 0.]]

        connections[:, :, 1] = [[0., 0., 0., 64.],      # inhibitory connections
                                [0., 0., 0., 64.],
                                [0., 0., 0., 0.],
                                [0., 0., 0., 16.]]

        # delays
        if delays is None:
            delays = np.zeros((N, N)) + 2e-3

        # instantiate populations
        #########################

        pcs_plastic = MoranPyramidalCells(step_size=step_size,
                                          init_state=init_states[0],
                                          tau=tau,
                                          label=population_labels[0]
                                          )
        pcs_nonplastic = MoranPyramidalCells(step_size=step_size,
                                             init_state=init_states[0],
                                             label=population_labels[1]
                                             )
        eins = MoranExcitatoryInterneurons(step_size=step_size,
                                           init_state=init_states[1],
                                           label=population_labels[2])
        iins = MoranInhibitoryInterneurons(step_size=step_size,
                                           init_state=init_states[2],
                                           label=population_labels[3])

        # call super init
        #################

        super().__init__(populations=[pcs_plastic, pcs_nonplastic, eins, iins],
                         connectivity=connections,
                         delays=delays,
                         step_size=step_size)


jrc = JansenRitCircuit(step_size=1e-3)
inp = np.zeros((1000, 3, 2))
inp[:, 0, 0] = np.random.uniform(120, 320, 1000)
jrc.run(inp, 1.)
from pyrates.observer import ExternalObserver
obs = ExternalObserver(observer=jrc.observer)