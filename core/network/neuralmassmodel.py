"""
Includes a basic neural mass model class.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from core.population import JansenRitPyramidalCells, JansenRitExcitatoryInterneurons, \
    JansenRitInhibitoryInterneurons, Population
from core.population.population import interpolate_array

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
                           Note: delay-free is equivalent to saying the expected delay is smaller than the defined
                                 time step
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
        self.time_steps = list()
        self.active_synapses = np.zeros((self.N, self.n_synapses), dtype=bool)
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
            store_step=1, variable_step_size=False, verbose=False, continue_run=False):
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
        :param variable_step_size: if true, variable step-size solver will be used to simulate the model
               (Runge-Kutta 4/5). Else Euler (default = False).
        :param verbose: if true, relative progress of simulation will be displayed (default = False).
        :param continue_run: if true, old run will be continued, if false, new simulation will be started
               (default = False).

        """

        ####################
        # check parameters #
        ####################

        assert synaptic_inputs.shape[1] == self.N
        assert synaptic_inputs.shape[2] == self.n_synapses
        if extrinsic_current is not None:
            assert extrinsic_current.shape[1] == self.N
        assert simulation_time >= 0
        assert cutoff_time >= 0
        assert store_step >= 1
        assert type(store_step) is int

        #########################################
        # enable continuation of old simulation #
        #########################################

        if not continue_run:
            self.time_steps.append(0.)

        simulation_time += self.time_steps[-1]
        t = self.time_steps[-1]

        ##################################
        # set neuromodulatory parameters #
        ##################################

        if extrinsic_modulation is None:
            extrinsic_modulation = np.ones((synaptic_inputs.shape[0], self.N)).tolist()

        if self.neuromodulation is None:
            self.neuromodulation = np.ones(self.N).tolist()

        ####################
        # simulate network #
        ####################

        n = 0
        self.time_steps.pop()

        while t < simulation_time-self.step_size:

            # get delayed input arriving at each neural mass
            network_input = self.get_delayed_input()

            neural_mass_states = np.zeros(self.N)
            firing_rates = np.zeros((self.N, 1))
            if variable_step_size:
                step_sizes = np.zeros(self.N)

            # update state of each neural mass according to input and store relevant state variables
            for i in range(self.N):

                # check whether neural mass needs to be updated
                if self.neural_masses[i].t <= t:

                    # calculate synaptic input
                    synaptic_input = synaptic_inputs[n, i, self.active_synapses[i, :]] + \
                                     network_input[i, self.active_synapses[i, :]]

                    # update all state variables
                    if extrinsic_current is not None:
                        self.neural_masses[i].state_update(synaptic_input=synaptic_input,
                                                           extrinsic_current=extrinsic_current[n, i],
                                                           extrinsic_synaptic_modulation=extrinsic_modulation[n][i],
                                                           synaptic_modulation_direction=self.neuromodulation[i],
                                                           variable_step_size=variable_step_size)
                    else:
                        self.neural_masses[i].state_update(synaptic_input=synaptic_input,
                                                           extrinsic_synaptic_modulation=extrinsic_modulation[n][i],
                                                           synaptic_modulation_direction=self.neuromodulation[i],
                                                           variable_step_size=variable_step_size)

                # update firing-rate look-up
                firing_rates[i] = self.neural_masses[i].output_firing_rate[-1]

                # store membrane potential
                neural_mass_states[i] = np.asarray(self.neural_masses[i].state_variables[-1])

                # store updated step-size
                if variable_step_size:
                    step_sizes[i] = self.neural_masses[i].step_size

            # store state-variables
            if t >= cutoff_time and np.mod(n, store_step) == 0:
                self.neural_mass_states.append(neural_mass_states)

            # update firing-rate look-up
            self.firing_rates_lookup = np.append(firing_rates, self.firing_rates_lookup, axis=1)
            self.firing_rates_lookup = self.firing_rates_lookup[:, 0:-1]

            # display simulation progress
            if verbose and (t == 0 or ((t / simulation_time) * 100. % 10) < 2*self.step_size):
                print('simulation process: ', (t / simulation_time) * 100., ' %')

            # update run variables
            t += self.step_size
            n += 1
            self.time_steps.append(t)

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

    def update_step_size(self, new_step_size, synaptic_inputs, update_threshold=1e-2, extrinsic_current=None,
                         extrinsic_synaptic_modulation=None, idx=0, interpolation_type='linear'):
        """
        Updates the time-step size with which the network is simulated.

        :param new_step_size: Scalar, indicates the new simulation step-size [unit = s].
        :param synaptic_inputs: synaptic input array that needs to be interpolated.
        :param update_threshold: If step-size ratio (old vs new) is larger than threshold, interpolations are initiated.
        :param extrinsic_current: extrinsic current array that needs to be interpolated.
        :param extrinsic_synaptic_modulation: synaptic modulation that needs to be interpolated.
        :param idx: Can be used to interpolate arrays from this point on (int) (default = 0).
        :param interpolation_type: character string, indicates the type of interpolation used for up-/down-sampling the
               firing-rate look-up matrix (default = 'cubic').

        :return interpolated synaptic input array.
        """

        ############################################
        # check whether update has to be performed #
        ############################################

        step_size_ratio = np.max((self.step_size, new_step_size)) / np.min((self.step_size, new_step_size))

        if np.abs(step_size_ratio - 1) > update_threshold and self.time_steps:

            ##########################
            # update step-size field #
            ##########################

            step_size_old = self.step_size
            self.step_size = new_step_size

            ##############################
            # update firing rate loop-up #
            ##############################

            # get maximum delay in seconds
            delay = np.max(np.array(self.D, dtype=int)) + self.step_size

            # check whether update is necessary
            step_diff = abs(int(delay/self.step_size) - int(delay/step_size_old))

            if step_diff >= 1 and delay > self.step_size:

                # perform interpolation of old firing rate look-up
                self.firing_rates_lookup = interpolate_array(old_step_size=step_size_old,
                                                             new_step_size=self.step_size,
                                                             y=self.firing_rates_lookup,
                                                             interpolation_type=interpolation_type,
                                                             axis=1)

            ###############################
            # update all extrinsic inputs #
            ###############################

            # check whether update is necessary
            net_input_time = synaptic_inputs[idx:, :, :].shape[0] * step_size_old

            step_diff = abs(int(net_input_time/self.step_size) - int(net_input_time/step_size_old))

            if step_diff >= 1:

                # perform updates
                synaptic_inputs = interpolate_array(old_step_size=step_size_old,
                                                    new_step_size=self.step_size,
                                                    y=synaptic_inputs[idx:, :, :],
                                                    axis=0,
                                                    interpolation_type=interpolation_type)

                if extrinsic_current:

                    extrinsic_current = interpolate_array(old_step_size=step_size_old,
                                                          new_step_size=self.step_size,
                                                          y=extrinsic_current[idx:, :],
                                                          axis=0,
                                                          interpolation_type=interpolation_type)

                if extrinsic_synaptic_modulation:

                    extrinsic_synaptic_modulation = interpolate_array(old_step_size=step_size_old,
                                                                      new_step_size=self.step_size,
                                                                      y=np.array(extrinsic_synaptic_modulation)[idx:, :],
                                                                      axis=0,
                                                                      interpolation_type=interpolation_type).tolist()

        return synaptic_inputs, extrinsic_current, extrinsic_synaptic_modulation, 0

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

        fig = plt.figure('Neural Mass States')
        plt.hold('on')

        legend_labels = []
        for i in neural_mass_idx:
            plt.plot(neural_mass_states[i, time_window[0]:time_window[1]])
            legend_labels.append(self.neural_mass_labels[i])

        plt.hold('off')
        plt.legend(legend_labels)
        plt.ylabel('membrane potential [V]')
        plt.xlabel('timesteps')

        if create_plot:
            fig.show()

        return fig


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

        pop_instance = JansenRitPyramidalCells(init_state=init_state,
                                               step_size=step_size,
                                               synaptic_kernel_length=synaptic_kernel_length,
                                               resting_potential=resting_potential,
                                               tau_leak=tau_leak,
                                               membrane_capacitance=membrane_capacitance,
                                               axon_params=axon_params,
                                               synapse_params=synapse_params)

    elif population_type == 'JansenRitExcitatoryInterneurons':

        pop_instance = JansenRitExcitatoryInterneurons(init_state=init_state,
                                                       step_size=step_size,
                                                       synaptic_kernel_length=synaptic_kernel_length,
                                                       resting_potential=resting_potential,
                                                       tau_leak=tau_leak,
                                                       membrane_capacitance=membrane_capacitance,
                                                       axon_params=axon_params,
                                                       synapse_params=synapse_params)

    elif population_type == 'JansenRitInhibitoryInterneurons':

        pop_instance = JansenRitInhibitoryInterneurons(init_state=init_state,
                                                       step_size=step_size,
                                                       synaptic_kernel_length=synaptic_kernel_length,
                                                       resting_potential=resting_potential,
                                                       tau_leak=tau_leak,
                                                       membrane_capacitance=membrane_capacitance,
                                                       axon_params=axon_params,
                                                       synapse_params=synapse_params)

    elif population_type is None:

        pop_instance = Population(synapses=synapses,
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
        D[i, :] = np.sqrt(np.sum(differences ** 2, axis=1))

    return D