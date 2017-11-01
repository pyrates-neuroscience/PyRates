"""
"""

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"

from core.network import NeuralMassModel


class JansenRitCircuit(NeuralMassModel):
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

    def __init__(self, population_resting_potentials=-0.075, population_leak_taus=0.016, population_capacitance=1e-12,
                 step_size=0.001, synaptic_kernel_length=100, distances=None, positions=None, velocities=None,
                 synapse_params=None, axon_params=None, init_states=None):
        """
        Initializes a basic Jansen-Rit circuit of pyramidal cells, excitatory interneurons and inhibitory interneurons.
        For detailed parameter description see super class.
        """

        ##################
        # set parameters #
        ##################

        populations = ['JansenRitPyramidalCells',
                       'JansenRitExcitatoryInterneurons',
                       'JansenRitInhibitoryInterneurons']
        population_labels = ['PCs', 'EINs', 'IINs']
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

        super(JansenRitCircuit, self).__init__(connections=connections,
                                               population_types=populations,
                                               population_labels=population_labels,
                                               population_resting_potentials=population_resting_potentials,
                                               population_leak_taus=population_leak_taus,
                                               population_capacitance=population_capacitance,
                                               step_size=step_size,
                                               synaptic_kernel_length=synaptic_kernel_length,
                                               distances=distances,
                                               positions=positions,
                                               velocities=velocities,
                                               synapse_params=synapse_params,
                                               axon_params=axon_params,
                                               init_states=init_states)