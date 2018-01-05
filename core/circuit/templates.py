"""Templates for specific circuit parametrizations.
"""

import numpy as np

from core.circuit import CircuitFromPopulations, CircuitFromScratch, Circuit
from core.population import WangKnoescheCells
from typing import Optional

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


class JansenRitCircuit(CircuitFromPopulations):
    """Basic Jansen-Rit circuit as defined in [1]_.

    Parameters
    ----------
    resting_potential
        Default = 0.0 V.
    step_size
        Default = 5e-4 s.
    max_synaptic_delay
        Default = None.
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

    def __init__(self,
                 resting_potential: float = 0.0,
                 step_size: float = 5e-4,
                 max_synaptic_delay: Optional[float] = None,
                 delays: Optional[np.ndarray] = None,
                 init_states: np.ndarray=np.zeros(3),
                 ) -> None:
        """Initializes a basic Jansen-Rit circuit of pyramidal cells, excitatory interneurons and inhibitory
        interneurons.
        """

        ##################
        # set parameters #
        ##################

        populations = ['JansenRitPyramidalCells',
                       'JansenRitInterneurons',
                       'JansenRitInterneurons']

        population_labels = ['JR_PCs',
                             'JR_EINs',
                             'JR_IINs']

        N = 3                                               # PCs, EINs, IIns
        n_synapses = 2                                      # excitatory and inhibitory

        ###################
        # set connections #
        ###################

        connections = np.zeros((N, N, n_synapses))
        c = 135.

        # excitatory connections
        connections[:, :, 0] = [[0, 0.8 * c, 0],
                                [1.0 * c, 0, 0],
                                [0.25 * c, 0, 0]]

        # inhibitory connections
        connections[:, :, 1] = [[0, 0, 0.25 * c],
                                [0, 0, 0],
                                [0, 0, 0]]

        ###################
        # call super init #
        ###################

        super().__init__(population_types=populations,
                         connectivity=connections,
                         delays=delays,
                         population_class='SecondOrderPopulation',
                         population_labels=population_labels,
                         resting_potential=resting_potential,
                         step_size=step_size,
                         max_synaptic_delay=max_synaptic_delay,
                         init_states=init_states)


class WangKnoescheCircuit(Circuit):
    """Basic 5-population cortical column circuit as defined in [1]_.

    Parameters
    ----------
    resting_potential
        Default = 0.0 V.
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
                 resting_potential: float = 0.0,
                 step_size: float = 5e-4,
                 max_synaptic_delay: Optional[float] = None,
                 delays: Optional[np.ndarray] = None,
                 init_states: np.ndarray = np.zeros(5),
                 tau_depression: float = 0.05,
                 tau_recycle: float = 0.5
                 ) -> None:
        """Initializes a basic Wang-Knoesche circuit of pyramidal cells and inhibitory interneurons in layer 2/3 as well
        as layer 5/6 plus a layer 4 excitatory interneuron population.
        """

        ##################
        # set parameters #
        ##################

        # synapse information
        n_synapses = 2

        # populations
        l23_pcs = WangKnoescheCells(step_size=step_size, resting_potential=resting_potential,
                                    max_synaptic_delay=max_synaptic_delay, init_state=init_states[0], label='L23_PCs',
                                    tau_depression=tau_depression, tau_recycle=tau_recycle)
        l23_iins = WangKnoescheCells(step_size=step_size, resting_potential=resting_potential,
                                     max_synaptic_delay=max_synaptic_delay, init_state=init_states[1], label='L23_IIns',
                                     synapses=['JansenRitExcitatorySynapse'], plastic_synapses=[True],
                                     tau_depression=tau_depression, tau_recycle=tau_recycle)
        l4_eins = WangKnoescheCells(step_size=step_size, resting_potential=resting_potential,
                                    max_synaptic_delay=max_synaptic_delay, init_state=init_states[2], label='L4_EINs',
                                    synapses=['JansenRitExcitatorySynapse'], plastic_synapses=[True],
                                    tau_depression=tau_depression, tau_recycle=tau_recycle)
        l56_pcs = WangKnoescheCells(step_size=step_size, resting_potential=resting_potential,
                                    max_synaptic_delay=max_synaptic_delay, init_state=init_states[3], label='L56_PCs',
                                    tau_depression=tau_depression, tau_recycle=tau_recycle)
        l56_iins = WangKnoescheCells(step_size=step_size, resting_potential=resting_potential,
                                     max_synaptic_delay=max_synaptic_delay, init_state=init_states[4], label='L56_IINs',
                                     synapses=['JansenRitExcitatorySynapse'], plastic_synapses=[True],
                                     tau_depression=tau_depression, tau_recycle=tau_recycle)

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

        ###################
        # call super init #
        ###################

        super().__init__(populations=[l23_pcs, l23_iins, l4_eins, l56_pcs, l56_iins],
                         connectivity=connections,
                         delays=delays,
                         step_size=step_size)


class MoranCircuit(CircuitFromPopulations):
    """Basic 3-population cortical column circuit as defined in [1]_.

    Parameters
    ----------
    resting_potential
        Default = 0.0 V.
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
    .. [1] R.J. Moran, S.J. Kiebel, K.E. Stephan, R.B. Reilly, J. Daunizeau & K.J. Friston, "A Neural Mass Model of
       Spectral Responses in Electrophysiology" NeuroImage, vol. 37, pp. 706-720, 2007.

    """

    def __init__(self,
                 resting_potential: float = 0.0,
                 step_size: float = 5e-4,
                 delays: Optional[np.ndarray] = None,
                 init_states: np.ndarray = np.zeros(3)
                 ) -> None:
        """Initializes a basic Moran circuit of pyramidal cells, excitatory and inhibitory interneurons.
        """

        ##################
        # set parameters #
        ##################

        # population information
        population_classes = ['SecondOrderPopulation',
                              'SecondOrderPlasticPopulation',
                              'SecondOrderPopulation']

        population_labels = ['PCs',
                             'EINs',
                             'IINs']
        N = 3

        # synapse information
        n_synapses = 2                                       # excitatory and inhibitory

        # connectivity matrix
        connections = np.zeros((N, N, n_synapses))

        connections[:, :, 0] = [[0., 128., 0.],     # excitatory connections
                                [128., 0.,  0.],
                                [64., 0., 0.]]

        connections[:, :, 1] = [[0., 0., 64.],    # inhibitory connections
                                [0., 0., 0.],
                                [0., 0., 16.]]

        ###################
        # call super init #
        ###################

        super().__init__(connectivity=connections,
                         population_types=['MoranPyramidalCells',
                                           'MoranExcitatoryInterneurons',
                                           'MoranInhibitoryInterneurons'],
                         delays=delays,
                         step_size=step_size,
                         resting_potential=resting_potential,
                         init_states=init_states,
                         population_class=population_classes,
                         population_labels=population_labels)
