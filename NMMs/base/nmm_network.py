"""
Includes a basic neural mass model class.
"""

import numpy as np
import NMMs.base.populations as pop

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


class NeuralMassModel:
    """
    Basic neural mass model class. Initializes a number of delay-coupled neural masses that are characterized by a
    number of synapses and an axon.

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

    def __init__(self, connections, population_labels=None, synapses=None, axons=None, step_size=0.001,
                 synaptic_kernel_length=100, distances=None, positions=None, velocities=None, synapse_params=None,
                 axon_params=None, init_states=None):
        """
        Initializes a network of neural masses.

        :param connections: N x N x n_synapses array resembling the number of synaptic contacts between every pair
               of the N populations for each of the n_synapses synapses.
               example: 3 populations, 2 synapses
               [ 1<-1, 1<-2, 1<-3 ]             [ 1<-1, 1<-2, 1<-3 ]
               [ 2<-1, 2<-2, 2<-3 ] [ 0 ]       [ 2<-1, 2<-2, 2<-3 ] [ 1 ]
               [ 3<-1, 3<-2, 3<-3 ]             [ 3<-1, 3<-2, 3<-3 ]
        :param population_labels: Can be list of character strings indicating the type of each neural mass in network
               (default = None).
        :param synapses: Can be list of n_synapses strings, indicating which pre-defined synapse types the third axis of the
               connections matrix resembles. If None, connections.shape[2] has to be 2, where the first entry is the
               default excitatory synapse as defined in Jansen & Rit (1995) and the second is the default inhibitory one
               (default = None).
        :param axons: Can be list of N strings, indicating which pre-defined axon type to use for each population in the
               network. Set either single list entry to None or parameter to None to not use custom axons
               (default = None).
        :param step_size: scalar, determining the time step-size with which the network simulation will progress
               [unit = s] (default = 0.001).
        :param synaptic_kernel_length: integer, determining the length of the synaptic kernel in terms of bins, where
               the distance between two bins is always equal to the step-size [unit = time-steps] (default = 100).
        :param distances: N x N array resembling the distance between every pair of neural masses. Can be None if
               network should not have delays or if population positions are used instead [unit = mm] (default = None).
        :param positions: N x 3 array, including the (x,y,z) coordinates of each neural mass. Will be used to compute
               euclidean distances if distances were not passed (default = None).
        :param velocities: Can be
                           1) a scalar determining the global velocity
                           2) an N x N array with a velocity for each connection in the network
                           3) a dictionary with the following fields:
                                'indices': N x N array of indices
                                'distributions': velocity distributions the indices refer to
                                'values': vector of velocity values for each bin of the velocity distributions
                           4) None if network is supposed to be delay-free
               [unit = m/s] (default = None).
        :param synapse_params: Can be
                               1) A list of length n_synapses with each entry being a dictionary with synapse parameters
                                  Default synapse parameters will be replaced by parameters in the dictionary (if set).
                               2) None, if pre-defined synapse parametrizations are to be used.
               (default = None)
        :param axon_params: Can be
                            1) A list of length N with each entry being a dictionary with axon parameters. Default axon
                               parameters will be replaced by the parameters in the dictionary (if set)
                            2) None, if pre-defined synapse parametrizations are to be used.
               (default = None)
        :param init_states: array including the initial states of each neural mass in network (membrane potential,
               firing rate). Each row is a neural mass.

        """

        ##########################
        # check input parameters #
        ##########################
        # TODO: exceptions
        assert type(connections) is np.ndarray
        assert connections.shape[0] == connections.shape[1]
        assert len(connections.shape) == 3
        assert type(population_labels) is list or population_labels is None
        assert step_size >= 0.
        assert type(step_size) is float
        assert synaptic_kernel_length > 0
        assert type(synaptic_kernel_length) is int
        if distances is not None:
            assert type(distances) is np.ndarray
        if (distances is not None or positions is not None) and (velocities is None):
            raise ValueError('Velocities need to be specified to calculate information propagation delays.')

        ##########################
        # set network parameters #
        ##########################

        self.N = int(connections.shape[0])
        self.neural_mass_labels = population_labels if population_labels else [str(i) for i in range(self.N)]
        self.neural_masses = list()
        self.C = connections
        self.n_synapses = int(connections.shape[2])
        self.step_size = step_size
        self.active_synapses = np.zeros((self.N, self.n_synapses), dtype=bool)

        # set up synapse labels
        if synapses:
            self.synapse_types = synapses
        elif synapse_params:
            self.synapse_types = ['custom' for i in range(self.n_synapses)]
        else:
            self.synapse_types = ['JansenRit_excitatory', 'JansenRit_inhibitory']

        # check whether connectivity matrix and synapse types match
        if self.n_synapses != len(self.synapse_types):
            raise ValueError('Third dimension of connectivity matrix needs to be the same size as the number of '
                             'synapses in the network. If no synapses are passed, this is 2 (for default excitatory '
                             'and inhibitory synapse).')

        # check whether synapse params were set correctly
        if synapse_params and len(synapse_params) != self.n_synapses:
            raise ValueError('If used, synapse_params has to be a list with a dictionary for each synapse type.')

        ############################################################
        # make population specific parameters set to none iterable #
        ############################################################

        axons = check_nones(axons, self.N)
        synapse_params = check_nones(synapse_params, self.n_synapses)
        axon_params = check_nones(axon_params, self.N)

        ##########################
        # initialize populations #
        ##########################

        if init_states is None:
            init_states = np.zeros((self.N, 2))

        for i in range(self.N):

            # check and extract  synapses that exist at respective population
            self.active_synapses[i, :] = (np.sum(connections[i, :, :], axis=0) != 0).squeeze()
            idx = np.asarray(self.active_synapses[i, :].nonzero(), dtype=int)
            if len(idx) == 1:
                idx = idx[0]
            synapses_tmp = [self.synapse_types[j] for j in idx]
            synapse_params_tmp = [synapse_params[j] for j in idx]

            # pass only those synapses next to other parameters to population class
            self.neural_masses.append(pop.Population(synapses=synapses_tmp,
                                                     axon=axons[i], init_state=init_states[i], step_size=step_size,
                                                     synaptic_kernel_length=synaptic_kernel_length,
                                                     axon_params=axon_params[i],
                                                     synapse_params=synapse_params_tmp))

        ##################################
        # set inter-population distances #
        ##################################

        if distances is not None:

            # use passed distance matrix
            self.D = distances

        elif positions is not None:

            # use positions to calculate distance matrix
            self.D = get_euclidean_distances(positions)

        else:

            # set distances to zero
            self.D = np.zeros((self.N, self.N))

        ###########################################################
        # transform distances into information propagation delays #
        ###########################################################

        if type(velocities) is float or np.ndarray:

            # divide distances by single or pair-wise velocity
            self.D = self.D / velocities

        elif type(velocities) is dict:

            # make D 3-dimensional
            self.D = np.tile(self.D.reshape(self.N, self.N, 1), (1, 1, len(velocities['values'])))

            # divide D by each possible velocity value on third dimension
            for i in range(self.D.shape[2]):
                self.D[:, :, i] = self.D[:, :, i] / velocities['values'][i]

            # store velocity distribution weights and indices on object instance
            self.velocity_distributions = velocities['distributions']
            self.velocity_distribution_indices = velocities['indices']

        elif not velocities and not all(self.D) == 0:

            raise ValueError('Velocities need to be determined to realize information propagation delays.')

        else:

            raise ValueError('Wrong input type for velocities')

    def run(self, synaptic_inputs, simulation_time, extrinsic_current=None, cutoff_time=0., store_step=1, verbose=False):
        """
        Simulates neural mass network.

        :param synaptic_inputs: n_timesteps x N x n_synapses array including the extrinsic synaptic input to each neural
               mass over the time course of the simulation [unit = firing rate].
        :param simulation_time: scalar, indicating the total time the network behavior is to be simulated [unit = s].
        :param extrinsic_current: n_timesteps x N array including all extrinsic currents applied to each neural
               mass over the time course of the simulation [unit = mA] (default = None).
        :param cutoff_time: scalar, indicating the initial time period of the simulation that will not be stored
               [unit = s] (default = 0).
        :param store_step: integer, indicating which simulated time steps will be stored. If store_step = n, every n'th
               step will be stored [unit = time-step] (default = 1).
        :param verbose: if true, relative progress of simulation will be displayed (default = False).

        """

        ##############################################################################
        # transform simulation times and network delays from seconds into time-steps #
        ##############################################################################

        time_steps = np.int(simulation_time / self.step_size)
        cutoff = np.int(cutoff_time / self.step_size)
        D = np.array(self.D / self.step_size, dtype=int)

        ####################
        # check parameters #
        ####################

        assert synaptic_inputs.shape[0] >= time_steps
        assert synaptic_inputs.shape[1] == self.N
        assert synaptic_inputs.shape[2] == self.n_synapses
        if any([np.sum(synaptic_inputs[i][self.active_synapses == 0]) for i in range(time_steps)]) > 0:
            raise ValueError('Cannot pass synaptic input to non-existent synapses!')
        if extrinsic_current:
            assert extrinsic_current.shape[0] >= time_steps
            assert extrinsic_current.shape[1] == self.N
        assert simulation_time >= 0
        assert cutoff_time >= 0
        assert store_step >= 1
        assert type(store_step) is int

        #####################################################################
        # initialize previous firing rate look-up matrix for network delays #
        #####################################################################

        firing_rates_lookup = np.zeros((self.N, np.max(D) + 1))
        firing_rates_lookup[:, 0] = np.array([self.neural_masses[i].output_firing_rate[-1] for i in range(self.N)])

        ####################
        # simulate network #
        ####################

        self.neural_mass_states = np.zeros((self.N, (time_steps - cutoff) // store_step + 1))
        n_store = 0

        for n in range(time_steps):

            # check at which position in firing rate look-up we are
            firing_rates_lookup_idx = np.mod(n, firing_rates_lookup.shape[1])

            # get delayed input arriving at each neural mass
            network_input = self.get_delayed_input(D, firing_rates_lookup)

            # update state of each neural mass according to input and store relevant state variables
            for i in range(self.N):

                # update all state variables
                if extrinsic_current:
                    self.neural_masses[i].state_update(synaptic_inputs[n, i, self.active_synapses[i, :]] +
                                                       network_input[i, self.active_synapses[i, :]],
                                                       extrinsic_current[n, i])
                else:
                    self.neural_masses[i].state_update(synaptic_inputs[n, i, self.active_synapses[i, :]] +
                                                       network_input[i, self.active_synapses[i, :]])

                # update firing-rate look-up
                firing_rates_lookup[i, firing_rates_lookup_idx] = self.neural_masses[i].output_firing_rate[-1]

                # store state-variables
                if n > cutoff and np.mod(n, store_step) == 0:
                    self.neural_mass_states[i, n_store] = np.asarray(self.neural_masses[i].state_variables[-1])
                    if i == self.N-1:
                        n_store += 1

            # display simulation progress
            if verbose and np.mod(n, time_steps//100) == 0:
                print('simulation process: ', (np.float(n)/np.float(time_steps)) * 100., ' %')

    def get_delayed_input(self, D, firing_rate_lookup):
        """
        Applies network delays to receive inputs arriving at each neural mass

        :param D: N x N x n_velocities delay matrix, where n_velocities is the number of possible
               velocities for each connection pair [unit = s].
        :param firing_rate_lookup: N x buffer_size matrix with collected firing rates from previous time-steps
               [unit = firing rate].

        :return: network input: N x n_synapses matrix. Delayed, weighted network input to each synapse of each neural
                 mass [unit: firing rate].

        """

        assert D.shape[0] == self.N
        assert D.shape[0] == D.shape[1]
        assert firing_rate_lookup.shape[0] == D.shape[0]

        #####################################
        # collect input to each neural mass #
        #####################################

        network_input = np.zeros((self.N, self.n_synapses))

        for i in range(self.N):

            # get delayed firing rates from all populations at each possible velocity
            firing_rate_delayed = np.array([firing_rate_lookup[i, D[i, j]] for j in range(self.N)])

            if hasattr(self, 'velocity_distributions'):

                # get velocity distribution for each connection to population i
                velocities = np.array(self.velocity_distributions[self.velocity_distribution_indices[i, :]]).squeeze()

                # apply velocity distribution weights to the delayed firing rates
                firing_rate_delayed = np.sum(firing_rate_delayed * velocities, axis=1)

            # apply connection weights to delayed firing rates
            network_input[i, :] = np.dot(self.C[i, :].T, firing_rate_delayed)

        return network_input


def check_nones(param, n):
    """
    Checks whether param is None. If yes, it returns a list of n Nones. If not, the param is returned.

    :param param: Parameter, to be tested for None.
    :param n: size of the list to return.

    :return: param or list of Nones

    """

    assert type(n) is int
    assert n > 0

    return [None for i in range(n)] if param is None else param


def get_euclidean_distances(positions):
    """
    Calculates the euclidean distances for every pair of positions.

    :param positions: N x 3 matrix, where N is the number of positions

    :return: N x N matrix with euclidean distances

    """

    assert type(positions) is np.ndarray
    assert positions.shape[1] == 3

    n = positions.shape[0]
    D = np.zeros((n, n))

    for i in range(n):

        # calculate coordinate difference between position i and all others
        differences = np.tile(positions[i, :], (n, 1)) - positions

        # calculate the square root of the sum of the squared differences for each pair of positions
        D[i, :] = np.sqrt(np.sum(differences**2, axis=1))

    return D
