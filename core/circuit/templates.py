"""
"""
import numpy as np

from core.circuit import CircuitFromPopulations

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


class JansenRitCircuit(CircuitFromPopulations):
    """
    Basic Jansen-Rit circuit as defined in Jansen & Rit (1995).

    :var N: integer that indicates number of populations in network.
    :var n_synapses: number of different synapse types expressed in the network
    :var synapse_types: list with names of the different synapse types in the network.
    :var C: N x N x n_synapses connectivity matrix.
    :var D: N x N x n_velocities delay matrix, where n_velocities is the number of possible velocities for each
         connection. If n_velocities > 1, there will be a field velocity_distributions on self including multiple
         probability distributions over the n velocities and a field velocity_distribution_indices with an N x N
         matrix including an index to a single distribution for each connection.
    :var neural_mass_labels: list with the names of the different populations.
    :var neural_masses: list with the population object instances.
    :var neural_mass_states: N x n_time_steps matrix, including the average membrane potential of each population at
         each time-step.
    :var step_size: scalar, indicating the size of the time-steps made during the simulation.

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
