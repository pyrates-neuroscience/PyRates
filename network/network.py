"""
Includes a basic neural mass model class.
"""

from matplotlib.pyplot import *
from scipy.interpolate import interp1d

import population.population as pop
import population.templates
from network import JansenRitCircuit

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"

# TODO: Try implementing adaptive simulation step-sizes
# TODO: Implement neuromodulatory mechanisms (see neuromodulatory synapses at population level)


class NeuralMassModel(object):
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

    def __init__(self, connections, population_labels=None, population_types=None, synapses=None, axons=None,
                 population_resting_potentials=-0.075, population_leak_taus=0.016, population_capacitance=1e-12,
                 step_size=0.001, synaptic_kernel_length=100, distances=None, positions=None, velocities=None,
                 synapse_params=None, axon_params=None, neuromodulatory_effect=None, init_states=None):
        """
        Initializes a network of neural masses.

        :param connections: N x N x n_synapses array representing the number of synaptic contacts between every pair
               of the N populations for each of the n_synapses synapses.
               example: 3 populations, 2 synapses
               [ 1<-1, 1<-2, 1<-3 ]             [ 1<-1, 1<-2, 1<-3 ]
               [ 2<-1, 2<-2, 2<-3 ] [ 0 ]       [ 2<-1, 2<-2, 2<-3 ] [ 1 ]
               [ 3<-1, 3<-2, 3<-3 ]             [ 3<-1, 3<-2, 3<-3 ]
        :param population_labels: Can be list of character strings indicating the name of each neural mass in network
               (default = None).
        :param population_types: Can be list of character strings indicating the type of each neural mass in network.
               Has to be one of the pre-implemented population types (default = None).
        :param synapses: Can be list of n_synapses strings, indicating which pre-defined synapse types the third axis of
               the connections matrix represents. If None, connections.shape[2] has to be 2, where the first entry is
               the default excitatory synapse as defined in Jansen & Rit (1995) and the second is the default inhibitory
               one (default = None).
        :param axons: Can be list of N strings, indicating which pre-defined axon type to use for each population in the
               network. Set either single list entry to None or parameter to None to not use custom axons
               (default = None).
        :param population_resting_potentials: Can be scalar, indicating the resting membrane potential of all
               populations or vector with a single resting potential for each population [unit = V] (default = -0.075).
        :param population_leak_taus: Can be scalar, indicating the leak current time-scale of all populations or
               vector with a single leak tau for each population [unit = s] (default = 0.016).
        :param population_capacitance: Can be scalar, indicating the membrane capacitance of all populations or vector
               with a single membrane capacitance for each population [unit = ] (default = 1e-12).
        :param step_size: scalar, determining the time step-size with which the network simulation will progress
               [unit = s] (default = 0.001).
        :param synaptic_kernel_length: integer, determining the length of the synaptic kernel in terms of bins, where
               the distance between two bins is always equal to the step-size [unit = time-steps] (default = 100).
        :param distances: N x N array representing the distance between every pair of neural masses. Can be None if
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
        :param neuromodulatory_effect: Can be list of scalars or vectors, indicating the direction in which the effect
               of neuromodulatory synapses is applied to all other synapses (default = None) [unit = 1].
        :param init_states: array including the initial states of each neural mass in network (membrane potential,
               firing rate). Each row is a neural mass.

        """

        ##########################
        # check input parameters #
        ##########################

        assert type(connections) is np.ndarray
        assert connections.shape[0] == connections.shape[1]
        assert len(connections.shape) == 3
        assert type(population_labels) is list or population_labels is None
        assert type(population_types) is list or population_types is None
        assert type(population_resting_potentials) is float or np.ndarray
        assert type(population_leak_taus) is float or np.ndarray
        assert type(population_capacitance) is float or np.ndarray
        assert step_size > 0.
        assert synaptic_kernel_length > 0
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
        self.neural_mass_states = list()
        self.C = connections
        self.n_synapses = int(connections.shape[2])
        self.step_size = step_size
        self.active_synapses = np.zeros((self.N, self.n_synapses), dtype=bool)
        self.time_steps_old = 0
        self.neuromodulation = neuromodulatory_effect

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

        ################################################
        # make population specific parameters iterable #
        ################################################

        population_types = check_nones(population_types, self.N)
        axons = check_nones(axons, self.N)
        synapse_params = check_nones(synapse_params, self.n_synapses)
        axon_params = check_nones(axon_params, self.N)

        if type(population_resting_potentials) is float:
            population_resting_potentials = [population_resting_potentials for i in range(self.N)]
        if type(population_leak_taus) is float:
            population_leak_taus = [population_leak_taus for i in range(self.N)]
        if type(population_capacitance) is float:
            population_capacitance = [population_capacitance for i in range(self.N)]

        ##########################
        # initialize populations #
        ##########################

        if init_states is None:
            init_states = np.zeros((self.N, 2))

        for i in range(self.N):

            # check and extract synapses that exist at respective population
            self.active_synapses[i, :] = (np.sum(connections[i, :, :], axis=0) != 0).squeeze()
            idx = np.asarray(self.active_synapses[i, :].nonzero(), dtype=int)
            if len(idx) == 1:
                idx = idx[0]
            synapses_tmp = [self.synapse_types[j] for j in idx]
            synapse_params_tmp = [synapse_params[j] for j in idx]

            # pass parameters to function that initializes the population
            self.neural_masses.append(set_population(population_type=population_types[i],
                                                     synapses=synapses_tmp,
                                                     axon=axons[i],
                                                     init_state=init_states[i],
                                                     step_size=step_size,
                                                     synaptic_kernel_length=synaptic_kernel_length,
                                                     resting_potential=population_resting_potentials[i],
                                                     tau_leak=population_leak_taus[i],
                                                     membrane_capacitance=population_capacitance[i],
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

        if type(velocities) is float or type(velocities) is np.ndarray:

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

        elif not velocities and np.sum(self.D) > 0:

            raise ValueError('Velocities need to be determined to realize information propagation delays.')

        elif velocities:

            raise ValueError('Wrong input type for velocities')

        ###########################
        # initialize delay buffer #
        ###########################

        max_delay = np.max(np.array(self.D / self.step_size, dtype=int)) + 1

        self.firing_rates_lookup = np.zeros((self.N, max_delay))
        self.firing_rates_lookup[:, 0] = np.array([self.neural_masses[i].output_firing_rate[-1] for i in range(self.N)])

    def run(self, synaptic_inputs, simulation_time, extrinsic_current=None, extrinsic_modulation=None, cutoff_time=0.,
            store_step=1, verbose=False, continue_run=False):
        """
        Simulates neural mass network.

        :param synaptic_inputs: n_timesteps x N x n_synapses array including the extrinsic synaptic input to each neural
               mass over the time course of the simulation [unit = 1/s].
        :param simulation_time: scalar, indicating the total time the network behavior is to be simulated [unit = s].
        :param extrinsic_current: n_timesteps x N array including all extrinsic currents applied to each neural
               mass over the time course of the simulation [unit = A] (default = None).
        :param extrinsic_modulation: Can be list of lists of scalars or vectors of scalars, indicating the extrinsic
               modulation of all/each synapse(s) of each population. First list dimension is timesteps, second is number
               of populations (default = None) [unit = 1].
        :param cutoff_time: scalar, indicating the initial time period of the simulation that will not be stored
               [unit = s] (default = 0).
        :param store_step: integer, indicating which simulated time steps will be stored. If store_step = n, every n'th
               step will be stored [unit = 1] (default = 1).
        :param verbose: if true, relative progress of simulation will be displayed (default = False).
        :param continue_run: if true, old run will be continued, if false, new simulation will be started
               (default = False).

        """

        ##############################################################################
        # transform simulation times and network delays from seconds into time-steps #
        ##############################################################################

        time_steps = np.int(simulation_time / self.step_size)
        cutoff = np.int(cutoff_time / self.step_size)

        ####################
        # check parameters #
        ####################

        assert synaptic_inputs.shape[0] >= time_steps
        assert synaptic_inputs.shape[1] == self.N
        assert synaptic_inputs.shape[2] == self.n_synapses
        if any([np.sum(synaptic_inputs[i][self.active_synapses == 0]) for i in range(time_steps)]) > 0:
            raise ValueError('Cannot pass synaptic input to non-existent synapses!')
        if extrinsic_current is not None:
            assert extrinsic_current.shape[0] >= time_steps
            assert extrinsic_current.shape[1] == self.N
        assert simulation_time >= 0
        assert cutoff_time >= 0
        assert store_step >= 1
        assert type(store_step) is int

        ################################################
        # check whether to start new simulation or not #
        ################################################

        if continue_run:
            time_steps += self.time_steps_old
        else:
            self.firing_rates_lookup[:] = 0

        ##################################
        # set neuromodulatory parameters #
        ##################################

        if extrinsic_modulation is None:
            extrinsic_modulation = [[1.0 for i in range(self.N)] for j in range(self.time_steps_old, time_steps)]

        if self.neuromodulation is None:
            self.neuromodulation = [1.0 for i in range(self.N)]

        ####################
        # simulate network #
        ####################

        for n in range(self.time_steps_old, time_steps):

            # get delayed input arriving at each neural mass
            network_input = self.get_delayed_input()

            neural_mass_states = np.zeros(self.N)
            firing_rates = np.zeros((self.N, 1))

            # update state of each neural mass according to input and store relevant state variables
            for i in range(self.N):

                # calculate synaptic input
                synaptic_input = synaptic_inputs[n - self.time_steps_old, i, self.active_synapses[i, :]] + \
                                 network_input[i, self.active_synapses[i, :]]

                # update all state variables
                if extrinsic_current is not None:
                    self.neural_masses[i].state_update(synaptic_input=synaptic_input,
                                                       extrinsic_current=extrinsic_current[n - self.time_steps_old, i],
                                                       extrinsic_synaptic_modulation=extrinsic_modulation[n][i],
                                                       synaptic_modulation_direction=self.neuromodulation[i])
                else:
                    self.neural_masses[i].state_update(synaptic_input=synaptic_input,
                                                       extrinsic_synaptic_modulation=extrinsic_modulation[n][i],
                                                       synaptic_modulation_direction=self.neuromodulation[i]
                                                       )

                # update firing-rate look-up
                firing_rates[i] = self.neural_masses[i].output_firing_rate[-1]

                # store membrane potential
                neural_mass_states[i] = np.asarray(self.neural_masses[i].state_variables[-1])

            # store state-variables
            if n >= cutoff and np.mod(n, store_step) == 0:
                self.neural_mass_states.append(neural_mass_states)

            # update firing-rate look-up
            self.firing_rates_lookup = np.append(firing_rates, self.firing_rates_lookup, axis=1)
            self.firing_rates_lookup = self.firing_rates_lookup[:, 0:-1]

            # display simulation progress
            if verbose and np.mod(n, time_steps//100) == 0:
                print('simulation process: ', (np.float(n)/np.float(time_steps)) * 100., ' %')

        # save simulation time
        self.time_steps_old = time_steps

    def get_delayed_input(self):
        """
        Applies network delays to receive inputs arriving at each neural mass

        :return: network input: N x n_synapses matrix. Delayed, weighted network input to each synapse of each neural
                 mass [unit: 1/s].

        """

        # transform propagation delays from seconds into time-steps
        D = np.array(self.D / self.step_size, dtype=int)

        #####################################
        # collect input to each neural mass #
        #####################################

        network_input = np.zeros((self.N, self.n_synapses))

        for i in range(self.N):

            # get delayed firing rates from all populations at each possible velocity
            firing_rate_delayed = np.array([self.firing_rates_lookup[j, D[i, j]] for j in range(self.N)])

            if hasattr(self, 'velocity_distributions'):

                # get velocity distribution for each connection to population i
                velocities = np.array(self.velocity_distributions[self.velocity_distribution_indices[i, :]]).squeeze()

                # apply velocity distribution weights to the delayed firing rates
                firing_rate_delayed = np.sum(firing_rate_delayed * velocities, axis=1)

            # apply connection weights to delayed firing rates
            network_input[i, :] = np.dot(self.C[i, :].T, firing_rate_delayed)

        return network_input

    def update_step_size(self, factor, interpolation_type='cubic'):
        """
        Updates the time-step size with which the network is simulated.

        :param factor: Scalar, indicates by which factor the current step-size is supposed to be scaled.
        :param interpolation_type: character string, indicates the type of interpolation used for up-/down-sampling the
               firing-rate look-up matrix (default = 'cubic').

        """

        assert factor >= 0

        ##########################
        # update step-size field #
        ##########################

        step_size_old = self.step_size
        self.step_size *= factor

        ###################################
        # update populations and synapses #
        ###################################

        for i in range(self.N):

            self.neural_masses[i].step_size = self.step_size

            for j in range(len(self.neural_masses[i].synapses)):

                self.neural_masses[i].synapses[j].step_size = self.step_size
                self.neural_masses[i].synapses[j].evaluate_kernel(build_kernel=True)

        ##############################
        # update firing rate loop-up #
        ##############################

        # get old and new maximum delay
        new_delay = np.max(np.array(self.D / self.step_size, dtype=int)) + 1
        old_delay = self.firing_rates_lookup.shape[1]

        # initialize new loop-up matrix
        firing_rates_lookup_new = np.zeros((self.N, new_delay))

        # get step-size values of old look-up matrix
        x = np.arange(old_delay) * step_size_old

        # for each neural mass, get firing rates in old look-up matrix, create linear interpolation function and use
        # interpolation function to get firing rates for step-sizes of new look-up matrix
        for i in range(self.N):
            y = self.firing_rates_lookup[i, :]
            f = interp1d(x, y, kind=interpolation_type)
            firing_rates_lookup_new[i, :] = f(np.arange(new_delay) * self.step_size)

        self.firing_rates_lookup = firing_rates_lookup_new

    def plot_neural_mass_states(self, neural_mass_idx=None, time_window=None, create_plot=True):
        """
        Creates figure with neural mass states over time.

        :param neural_mass_idx: Can be list of neural mass indices, indicating for which populations to plot the states.
               (default = None).
        :param time_window: Can be array with start and end time of plotting window [unit = s] (default = None).
        :param create_plot: If false, plot will not be shown

        :return: figure handle

        """

        neural_mass_states = np.array(self.neural_mass_states).T

        ##########################
        # check input parameters #
        ##########################

        assert neural_mass_idx is None or type(neural_mass_idx) is list
        assert time_window is None or type(time_window) is np.ndarray

        ##############################
        # check positional arguments #
        ##############################

        if neural_mass_idx is None:
            neural_mass_idx = range(self.N)

        if time_window is None:
            time_window = np.array([0, neural_mass_states.shape[1]])
        time_window = np.array(time_window / self.step_size, dtype=int)

        #####################################
        # plot neural mass states over time #
        #####################################

        fig = figure('Neural Mass States')
        hold('on')

        legend_labels = []
        for i in neural_mass_idx:
            plot(neural_mass_states[i, time_window[0]:time_window[1]])
            legend_labels.append(self.neural_mass_labels[i])

        hold('off')
        legend(legend_labels)
        ylabel('membrane potential [V]')
        xlabel('timesteps')

        if create_plot:
            fig.show()

        return fig


class NeuralMassNetwork(NeuralMassModel):
    """
    Large-scale neural mass network, consisting of multiple local neural mass circuits.
    """

    def __init__(self, connections, input_populations=None, output_populations=None, input_synapses=None,
                 nmm_types=None, nmm_labels=None, distances=None, positions=None, velocities=None, nmm_parameters=None,
                 step_size=5e-4):
        """
        Initializes a network of neural mass models.

        :param connections:
        :param input_populations:
        :param output_populations:
        :param input_synapses:
        :param nmm_types:
        :param nmm_labels:
        :param distances:
        :param positions:
        :param velocities:
        :param nmm_parameters:
        :param step_size:

        """

        ####################
        # check parameters #
        ####################

        assert connections.shape[0] == connections.shape[1]
        assert input_populations is None or len(input_populations) == connections.shape[0]
        assert output_populations is None or len(output_populations) == connections.shape[0]
        assert input_synapses is None or len(input_synapses) == connections.shape[0]
        assert nmm_types is None or len(nmm_types) == connections.shape[0]
        assert nmm_labels is None or len(nmm_labels) == connections.shape[0]
        assert distances is None or distances.shape == connections.shape
        assert positions is None or (positions.shape[0] == connections.shape[0] and positions.shape[1] == 3)
        assert velocities is None or (type(velocities) is float or list or dict)
        assert nmm_parameters is None or type(nmm_parameters) is list
        assert step_size >= 0
        if nmm_types is None and nmm_parameters is None:
            raise ValueError('Either NMM types or custom parameters have to be passed!')

        ##########################
        # set network parameters #
        ##########################

        self.N = connections.shape[0]
        self.neural_mass_labels = nmm_labels if nmm_labels else [str(i) for i in range(self.N)]
        self.C = connections
        self.neural_masses = list()
        self.neural_mass_states = list()
        self.step_size = step_size
        self.time_steps_old = 0
        self.n_synapses = 1

        if input_populations:
            self.input_populations = input_populations
        else:
            self.input_populations = np.zeros(self.N, dtype=int)

        if output_populations:
            self.output_populations = output_populations
        else:
            self.output_populations = np.zeros(self.N, dtype=int)

        if input_synapses:
            self.input_synapses = input_synapses
        else:
            self.input_synapses = np.zeros(self.N, dtype=int)

        #########################################
        # make nmm specific parameters iterable #
        #########################################

        nmm_types = check_nones(nmm_types, self.N)

        ###################
        # initialize NMMs #
        ###################

        if nmm_parameters is None:

            for i in range(self.N):
                self.neural_masses.append(set_nmm(nmm_type=nmm_types[i],
                                                  step_size=step_size))

        else:

            for i in range(self.N):
                self.neural_masses.append(set_nmm(nmm_type=nmm_types[i],
                                                  nmm_parameters=nmm_parameters[i],
                                                  step_size=step_size))

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

        # transform propagation delays from seconds into time-steps
        max_delay = np.max(np.array(self.D / self.step_size, dtype=int)) + 1

        ###########################
        # initialize delay buffer #
        ###########################

        self.firing_rates_lookup = np.zeros((self.N, max_delay))
        self.firing_rates_lookup[:, 0] = np.array([self.neural_masses[i].neural_masses[self.output_populations[i]].
                                                  output_firing_rate[-1] for i in range(self.N)])

    def run(self, synaptic_inputs, simulation_time, extrinsic_current=None, cutoff_time=0, store_step=1, verbose=False,
            continue_run=False):
        """
        Simulates network behavior over time.

        :param synaptic_inputs:
        :param simulation_time:
        :param extrinsic_current:
        :param cutoff_time:
        :param store_step:
        :param verbose:
        :param continue_run:

        """

        ##############################################################################
        # transform simulation times and network delays from seconds into time-steps #
        ##############################################################################

        time_steps = np.int(simulation_time / self.step_size)
        cutoff = np.int(cutoff_time / self.step_size)

        ####################
        # check parameters #
        ####################

        assert synaptic_inputs.shape[0] >= time_steps
        assert synaptic_inputs.shape[1] == self.N
        assert len(synaptic_inputs.shape) == 4
        if extrinsic_current:
            assert extrinsic_current.shape[0] >= time_steps
            assert extrinsic_current.shape[1] == self.N
            assert len(extrinsic_current.shape) == 3
        assert simulation_time >= 0
        assert cutoff_time >= 0
        assert store_step >= 1
        assert type(store_step) is int

        ################################################
        # check whether to start new simulation or not #
        ################################################

        if continue_run:
            time_steps += self.time_steps_old
        else:
            self.firing_rates_lookup[:] = 0

        ####################
        # simulate network #
        ####################

        for n in range(self.time_steps_old, time_steps):

            # check at which position in firing rate look-up we are
            firing_rates_lookup_idx = np.mod(n, self.firing_rates_lookup.shape[1])

            # get delayed input arriving at each neural mass
            network_input = self.get_delayed_input()

            neural_mass_states = np.zeros(self.N)

            ###########################
            # TODO: CPU Parallelization
            ###########################
            # EXAMPLE CODE
            """
            import multiprocessing
            N_CPU_total = multiprocessing.cpu_count()
            if N_CPU_total > N_MAX_CPU:
                N_CPU = N_MAX_CPU
            else:
                N_CPU = N_CPU_total

            print(
                "Starting multiprocessed index version with {:d}/{:d} processors ...").format(N_CPU, N_CPU_total)

            for j in range(len(points_out)):
                print("Starting run " + str(j + 1) + " of " + str(range(len(points_out))))
                # n_points[j] are all points in roi
                # call multiprocessed version here -------------------------------

                #  range(N_points_out_max) aufteilen nach cpu count
                avg = N_points[j] / float(N_CPU)
                chunks = []
                last = 0.0
                points = range(N_points[j])
                while last < N_points[j]:
                    chunks.append(np.array(points[int(last):int(last + avg)]))
                    last += avg

                # call pool
                pool = multiprocessing.Pool(N_CPU)
                partialized = partial(elem_workhorse,
                                      points_out=points_out[j],
                                      P1_all=P1_all,
                                      P2_all=P2_all,
                                      P3_all=P3_all,
                                      P4_all=P4_all,
                                      N_points_total=N_points[j],
                                      N_CPU=N_CPU)

                results = pool.map(partialized, chunks)

                # concatenate results
                tet_idx[:, j] = np.concatenate(results, axis=0)
                pool.close()
                pool.join()
                
            """

            # update state of each neural mass according to input and store relevant state variables
            for i in range(self.N):

                # calculate synaptic input
                synaptic_input = synaptic_inputs[n, i, 0:self.neural_masses[i].N, 0:self.neural_masses[i].n_synapses]
                synaptic_input = np.reshape(synaptic_input, (1, self.neural_masses[i].N,
                                                             self.neural_masses[i].n_synapses))
                synaptic_input[0, self.input_populations[i], self.input_synapses[i]] = network_input[i]

                # get extrinsic current
                if extrinsic_current is None:
                    extrinsic_current_tmp = np.zeros((1, self.neural_masses[i].N))
                else:
                    extrinsic_current_tmp = np.reshape(extrinsic_current[n, i, 0:self.neural_masses[i].N],
                                                       (1, self.neural_masses[i].N))
                # update nmm state
                self.neural_masses[i].run(synaptic_inputs=synaptic_input,
                                          simulation_time=self.step_size,
                                          extrinsic_current=extrinsic_current_tmp,
                                          continue_run=True)

                # update firing-rate look-up
                self.firing_rates_lookup[i, firing_rates_lookup_idx] = \
                    self.neural_masses[i].neural_masses[self.output_populations[i]].output_firing_rate[-1]

                # store membrane potential
                neural_mass_states_tmp = np.asarray(self.neural_masses[i].neural_mass_states[-1])
                neural_mass_states[i] = neural_mass_states_tmp[self.output_populations[i]]

            # store state-variables
            if n > cutoff and np.mod(n, store_step) == 0:
                self.neural_mass_states.append(neural_mass_states)

            # display simulation progress
            if verbose and np.mod(n, time_steps // 100) == 0:
                print('simulation process: ', (np.float(n) / np.float(time_steps)) * 100., ' %')

            # save simulation time
            self.time_steps_old = time_steps

    def update_step_size(self, factor, interpolation_type='cubic'):
        """
        Updates the time-step size with which the network is simulated.

        :param factor: Scalar, indicates by which factor the current step-size is supposed to be scaled.
        :param interpolation_type: character string, indicates the type of interpolation used for up-/down-sampling the
               firing-rate look-up matrix (default = 'cubic').

        """

        assert factor >= 0

        ##########################
        # update step-size field #
        ##########################

        step_size_old = self.step_size
        self.step_size *= factor

        ###################################
        # update populations and synapses #
        ###################################

        for i in range(self.N):

            self.neural_masses[i].update_step_size(factor=factor)

        ##############################
        # update firing rate loop-up #
        ##############################

        # get old and new maximum delay
        new_delay = np.max(np.array(self.D / self.step_size, dtype=int)) + 1
        old_delay = self.firing_rates_lookup.shape[1]

        # initialize new loop-up matrix
        firing_rates_lookup_new = np.zeros((self.N, new_delay))

        # get step-size values of old look-up matrix
        x = np.arange(old_delay) * step_size_old

        # for each neural mass, get firing rates in old look-up matrix, create linear interpolation function and use
        # interpolation function to get firing rates for step-sizes of new look-up matrix
        for i in range(self.N):
            y = self.firing_rates_lookup[i, :]
            f = interp1d(x, y, kind=interpolation_type)
            firing_rates_lookup_new[i, :] = f(np.arange(new_delay) * self.step_size)

        self.firing_rates_lookup = firing_rates_lookup_new


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


def set_population(population_type, synapses, axon, init_state, step_size, synaptic_kernel_length, resting_potential,
                   tau_leak, membrane_capacitance, axon_params, synapse_params):
    """
    Instantiates a population. For detailed parameter description, see population class.

    :param population_type: Can be character string, indicating which pre-implemented population sub-class to use or None,
           if custom population is to be initialized.

    :return population instance

    """

    if population_type == 'JansenRitPyramidalCells':

        pop_instance = population.templates.JansenRitPyramidalCells(init_state=init_state,
                                                                    step_size=step_size,
                                                                    synaptic_kernel_length=synaptic_kernel_length,
                                                                    resting_potential=resting_potential,
                                                                    tau_leak=tau_leak,
                                                                    membrane_capacitance=membrane_capacitance,
                                                                    axon_params=axon_params,
                                                                    synapse_params=synapse_params)

    elif population_type == 'JansenRitExcitatoryInterneurons':

        pop_instance = population.templates.JansenRitExcitatoryInterneurons(init_state=init_state,
                                                                            step_size=step_size,
                                                                            synaptic_kernel_length=synaptic_kernel_length,
                                                                            resting_potential=resting_potential,
                                                                            tau_leak=tau_leak,
                                                                            membrane_capacitance=membrane_capacitance,
                                                                            axon_params=axon_params,
                                                                            synapse_params=synapse_params)

    elif population_type == 'JansenRitInhibitoryInterneurons':

        pop_instance = population.templates.JansenRitInhibitoryInterneurons(init_state=init_state,
                                                                            step_size=step_size,
                                                                            synaptic_kernel_length=synaptic_kernel_length,
                                                                            resting_potential=resting_potential,
                                                                            tau_leak=tau_leak,
                                                                            membrane_capacitance=membrane_capacitance,
                                                                            axon_params=axon_params,
                                                                            synapse_params=synapse_params)

    elif population_type is None:

        pop_instance = pop.Population(synapses=synapses,
                                      axon=axon,
                                      init_state=init_state,
                                      step_size=step_size,
                                      synaptic_kernel_length=synaptic_kernel_length,
                                      resting_potential=resting_potential,
                                      tau_leak=tau_leak,
                                      membrane_capacitance=membrane_capacitance,
                                      axon_params=axon_params,
                                      synapse_params=synapse_params)

    else:

        raise ValueError('Invalid population type!')

    return pop_instance


def set_nmm(nmm_type, nmm_parameters=None, step_size=5e-4):
    """
    Initializes nmm instance.

    :param nmm_type: character string indicating which pre-implemented nmm to use.
    :param nmm_parameters: dictionary, can be used to initialize custom nmms.
    :param step_size:

    :return: nmm_instance
    """

    ####################
    # check parameters #
    ####################

    assert type(nmm_type) is str or nmm_type is None
    assert type(nmm_parameters) is dict or nmm_parameters is None
    assert step_size >= 0

    #####################################
    # check all possible nmm parameters #
    #####################################

    if nmm_parameters:

        param_list = ['connections', 'population_labels', 'population_types', 'synapses', 'axons',
                      'population_resting_potentials', 'population_leak_taus', 'population_capacitance',
                      'synaptic_kernel_length', 'distances', 'positions', 'velocities', 'synapse_params',
                      'axon_params', 'init_states']

        for param in param_list:

            if param not in nmm_parameters:
                nmm_parameters[param] = None

    ##################
    # initialize nmm #
    ##################

    if nmm_parameters is None:

        if nmm_type == 'JansenRitCircuit':

            nmm_instance = JansenRitCircuit(step_size=step_size)

        else:

            raise ValueError('Wrong NMM type!')

    else:

        if nmm_type == 'JansenRitCircuit':

            nmm_instance = JansenRitCircuit(
                            population_resting_potentials=nmm_parameters['population_resting_potentials'],
                            population_leak_taus=nmm_parameters['population_leak_taus'],
                            population_capacitance=nmm_parameters['population_capacitance'],
                            step_size=step_size,
                            synaptic_kernel_length=nmm_parameters['synaptic_kernel_length'],
                            distances=nmm_parameters['distances'],
                            positions=nmm_parameters['positions'],
                            velocities=nmm_parameters['velocities'],
                            synapse_params=nmm_parameters['synapse_params'],
                            axon_params=nmm_parameters['axon_params'],
                            init_states=nmm_parameters['init_states'])

        else:

            nmm_instance = NeuralMassModel(
                            connections=nmm_parameters['connections'],
                            population_labels=nmm_parameters['population_labels'],
                            population_types=nmm_parameters['population_types'],
                            synapses=nmm_parameters['synapses'],
                            axons=nmm_parameters['axons'],
                            population_resting_potentials=nmm_parameters['population_resting_potentials'],
                            population_leak_taus=nmm_parameters['population_leak_taus'],
                            population_capacitance=nmm_parameters['population_capacitance'],
                            step_size=step_size,
                            synaptic_kernel_length=nmm_parameters['synaptic_kernel_length'],
                            distances=nmm_parameters['distances'],
                            positions=nmm_parameters['positions'],
                            velocities=nmm_parameters['velocities'],
                            synapse_params=nmm_parameters['synapse_params'],
                            axon_params=nmm_parameters['axon_params'],
                            init_states=nmm_parameters['init_states'])

    return nmm_instance


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
