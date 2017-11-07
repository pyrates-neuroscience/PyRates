"""
"""
import numpy as np
from scipy.interpolate import interp1d

from core.network import NeuralMassModel, JansenRitCircuit
from core.network.neuralmassmodel import check_nones, get_euclidean_distances

__author__ = "Daniel Rose, Richard Gast"
__status__ = "Development"


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