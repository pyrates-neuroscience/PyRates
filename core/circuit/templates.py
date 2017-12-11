"""Templates for specific ciricuit parametrizations.
"""

import numpy as np

from core.circuit import CircuitFromPopulations

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
    variable_step_size
        Default = False.
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
                 step_size: float=5e-4, variable_step_size: bool=False, max_synaptic_delay: float=0.05,
                 delays: np.ndarray=None, init_states: np.ndarray=np.zeros(3)) -> None:
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
                         variable_step_size=variable_step_size,
                         max_synaptic_delay=max_synaptic_delay,
                         init_states=init_states)
