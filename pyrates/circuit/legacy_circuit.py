"""
"""
from typing import List, Union, Optional, Callable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from networkx import relabel_nodes
from pandas import DataFrame

from pyrates.observer import CircuitObserver
from pyrates.population import Population, SynapticInputPopulation, ExtrinsicCurrentPopulation, \
    ExtrinsicModulationPopulation
from pyrates.utility import make_iterable, set_instance
from pyrates.utility.filestorage import RepresentationBase
from pyrates.utility.networkx_wrapper import WrappedMultiDiGraph

__author__ = "Daniel Rose"
__status__ = "Development"


#########################
# generic circuit class #
#########################

class Circuit(RepresentationBase):
    """Base circuit class.

    Sets up a graph-based network of (delay-) coupled populations. Allows to simulate the network behavior subsequently.

    Parameters
    ----------
    populations
        List of population instances. Need to have same simulation step-size.
    connectivity
        Can either be a 3D array (n_populations x n_populations x n_synapses) with each entry representing the average
        number of synaptic contacts between two populations via a given synapse type. 1. dimension are the receiving
        populations, 2. dimension are the sending populations. Or can be a list with connection weights for each
        non-zero connection in network.
    delays
        Can either be a 2D (n_populations x n_populations) or 3D array (n_populations x n_populations x n_delays)
        with each entry representing the information transfer delay between the respective populations [unit = s].
        Or can be a list of delays, if connectivity is a list of connection weights.
    delay_distributions
        3D array (n_populations x n_populations x n_delays) with each entry representing a weight for the
        corresponding entry in delays [unit = 1].
    source_pops
        Keys of the source populations. Only needed, if connectivity is a list with connection weights.
    target_pops
        Keys of the target populations. Only needed, if connectivity is a list with connection weights.
    target_syns
        Keys of the target synapses on the target populations. Only needed, if connectivity is a list with connection
        weights.
    step_size
        Simulation step-size [unit = s] (default = 5e-4).
    synapse_keys
        Keys of the synapse types, the third dimension of connectivity refers to. Only needed, if connectivity is a
        3D array.

    Attributes
    ----------
    populations
        See description of parameter `populations`
    step_size
        See description of parameter `step_size`.
    n_populations
        Number of populations in circuit.
    t
        Current time the circuit lives at [unit = s].
    network_graph
        Directional multi-graph representing the network structure.

    """

    def __init__(self,
                 populations: List[Population],
                 connectivity: Union[np.ndarray, List[float]],
                 delays: Union[np.ndarray, List[float]],
                 delay_distributions: Optional[np.ndarray] = None,
                 source_pops: Optional[List[str]] = None,
                 target_pops: Optional[List[str]] = None,
                 target_syns: Optional[Union[np.ndarray, List[str]]] = None,
                 step_size: float = 5e-4,
                 synapse_keys: Optional[List[str]] = None,
                 ) -> None:
        """Instantiates population circuit object.
        """

        ##########################
        # check input parameters #
        ##########################

        # attribute dimensions
        if isinstance(connectivity, np.ndarray):
            if len(populations) != connectivity.shape[0] or len(populations) != connectivity.shape[1]:
                raise AttributeError('First and second dimension of connectivity matrix must equal the number of '
                                     'populations. See parameter docstrings for further explanation.')
            if len(populations) != delays.shape[0] or len(populations) != delays.shape[1]:
                raise AttributeError('First and second dimension of delay matrix must equal the number of populations. '
                                     'See parameter docstrings for further explanation.')
            if not synapse_keys and connectivity.shape[2] != 2:
                raise AttributeError('If last dimension of connectivity matrix refers to more or less than two '
                                     'synapses, synapse_keys have to be passed.')
            elif synapse_keys and len(synapse_keys) != connectivity.shape[2]:
                raise AttributeError('Number of passed synapse types has to match the last dimension of connectivity.')

        # attribute values
        # noinspection PyTypeChecker
        if step_size < 0. or (isinstance(delays, np.ndarray) and np.sum(delays < 0) > 0) or \
                (isinstance(delays, list) and sum([1 for d in delays if d < 0.]) > 0):
            raise ValueError('Time constants (delays, step_size) cannot be negative. See parameter docstrings for '
                             'further explanation.')

        # set circuit attributes
        ########################

        # general
        self.step_size = step_size
        self.n_populations = len(populations)
        self.t = 0.
        self.synapse_keys = synapse_keys if synapse_keys else ['excitatory', 'inhibitory']
        self.run_info = dict()

        # circuit populations
        self.populations = dict()
        for i, pop in enumerate(populations):
            if pop.key:
                self.populations[pop.key] = pop
            else:
                self.populations[str(i)] = pop

        # infer circuit structure
        #########################

        # turn connectivity matrices into lists if necessary
        if type(connectivity) is not list:

            # create lists
            pop_labels = list(self.populations.keys())
            source_pops = list()
            target_pops = list()
            target_syns = list()
            delays_tmp = list()
            connectivity_tmp = list()

            # for each source population
            for source in range(connectivity.shape[1]):

                # for each target population
                for target in range(connectivity.shape[0]):

                    # get all synapses at target population
                    syn_labels = list(self.populations[pop_labels[target]].synapses.keys())

                    # for each synapse on target population that the source population should connect to
                    for syn in np.where(connectivity[target, source] > 0.)[0]:

                        # get the key of the respective synapse
                        for key in syn_labels:
                            if self.synapse_keys[syn] in key:
                                break

                        if delay_distributions:

                            # loop over all possible delays, append connectivity information to lists
                            #  and apply the delay probability masses to the connectivity weights
                            for delay, delay_weight in zip(delays[target, source], delay_distributions[target, source]):
                                source_pops.append(pop_labels[source])
                                target_pops.append(pop_labels[target])
                                target_syns.append(key)
                                delays_tmp.append(delay)
                                connectivity_tmp.append(connectivity[target, source, syn] * delay_weight)

                        else:

                            # append connectivity information to lists
                            source_pops.append(pop_labels[source])
                            target_pops.append(pop_labels[target])
                            target_syns.append(key)
                            delays_tmp.append(delays[target, source])
                            connectivity_tmp.append(connectivity[target, source, syn])

            connectivity = connectivity_tmp
            delays = delays_tmp

        # set population specific properties
        ####################################

        for i, pop in enumerate(self.populations.keys()):

            # clear all past information stored on population
            self.populations[pop].clear(disconnect=True)

            # update max_population_delays according to information in D
            target_delays = list()
            for t in range(len(delays)):
                for snippet in target_pops[t].split('_'):
                    if snippet not in pop:
                        break
                target_delays.append(delays[t])
            self.populations[pop].max_population_delay = max(target_delays)

            # set population step size to circuit step size
            self.populations[pop].step_size = step_size

            # update all time-dependent attributes of population
            self.populations[pop].update()

        # create network graph
        ######################

        # build graph from lists
        self.network_graph = self.build_graph(plasticity_on_input=False,
                                              source_pops=source_pops,
                                              target_pops=target_pops,
                                              target_syns=target_syns,
                                              conn_strengths=connectivity,
                                              conn_delays=delays)
        self.n_nodes = len(self.network_graph.nodes)

    # noinspection PyPep8Naming
    @property
    def N(self):
        """Read-only value for number of populations. Just keeping this as a safeguard for legacy code.
        """
        return self.n_populations

    # noinspection PyPep8Naming
    @property
    def C(self):
        """Read-only value for the connectivity matrix.
        """

        connectivity = np.zeros((self.n_populations, self.n_populations))
        nodes = list(self.network_graph.nodes.keys())
        for i, source in enumerate(nodes):
            conns = self.network_graph.adj[source]
            for target in conns.keys():
                val = 0
                for _, conn in conns[target].items():
                    val += conn['weight']
                connectivity[nodes.index(target), i] = val

        return connectivity

    # noinspection PyPep8Naming
    @property
    def D(self):
        """Read-only value for the delay matrix.
        """

        delays = np.zeros((self.n_populations, self.n_populations))
        nodes = list(self.network_graph.nodes.keys())
        for i, source in enumerate(nodes):
            conns = self.network_graph.adj[source]
            for target in conns.keys():
                val = list()
                for _, conn in conns[target].items():
                    val.append(conn['delay'])
                delays[nodes.index(target), i] = np.mean(val)

        return delays

    # noinspection PyPep8Naming
    @property
    def connectivity_lists(self):
        """Read-only value for the connectivity information needed to re-build the circuit graph.
        """

        # initialize lists
        connectivity = list()
        delays = list()
        source_pops = list()
        target_pops = list()
        target_syns = list()

        # collect information
        #####################

        nodes = list(self.network_graph.nodes.keys())
        for i, source in enumerate(nodes):
            conns = self.network_graph.adj[source]
            for target in conns.keys():
                for _, conn in conns[target].items():
                    connectivity.append(conn['weight'])
                    delays.append(conn['delay'])
                    source_pops.append(source)
                    target_pops.append(target)
                    target_syns.append(conn['synapse'].label)

        return connectivity, delays, source_pops, target_pops, target_syns

    def _prepare_run(self,
                     synaptic_inputs: np.ndarray,
                     simulation_time: float,
                     synaptic_input_pops: Optional[List[str]] = None,
                     synaptic_input_syns: Optional[List[str]] = None,
                     extrinsic_current: Optional[np.ndarray] = None,
                     extrinsic_current_pops: Optional[list] = None,
                     extrinsic_modulation: Optional[List[np.ndarray]] = None,
                     extrinsic_modulation_pops: Optional[list] = None,
                     sampling_size: Optional[float] = None,
                     observe_populations: Optional[List[str]] = None,
                     observe_states: Optional[List[str]] = None
                     ) -> int:
        """Helper method to check inputs to run, but keep run function a bit cleaner.

        See Also
        --------
        :method:`run` for a detailed documentation of the arguments.

        """

        # check input parameters
        ########################

        # simulation time
        if simulation_time < 0.:
            raise ValueError('Simulation time cannot be negative.')
        simulation_time_steps = int(simulation_time / self.step_size)

        # if needed, transfer synaptic inputs into new, key-based version
        if len(synaptic_inputs.shape) > 2:

            synaptic_inputs_tmp = list()
            synaptic_input_pops = list()
            synaptic_input_syns = list()

            # loop over populations
            for p in range(synaptic_inputs.shape[1]):

                inp_to_pop = synaptic_inputs[:, p, :]

                # loop over synapses
                for s in range(synaptic_inputs.shape[2]):

                    inp_to_syn = inp_to_pop[:, s].squeeze()

                    # store information of the population-synapse pairs for which input is defined
                    if np.sum(inp_to_syn != 0.) > 0.:
                        pop_key = list(self.populations.keys())[p]
                        syn_key = list(self.populations[pop_key].synapses.keys())[s]
                        synaptic_inputs_tmp.append(inp_to_syn)
                        synaptic_input_pops.append(pop_key)
                        synaptic_input_syns.append(syn_key)

            synaptic_inputs = np.array(synaptic_inputs_tmp).T if synaptic_inputs_tmp else None

        # check synaptic input arguments for correct dimensionalities
        if synaptic_inputs is not None:
            if synaptic_inputs.shape[0] != simulation_time_steps:
                raise ValueError('First dimension of synaptic input has to match the number of simulation time steps!\n'
                                 f'input: {synaptic_inputs.shape[0]}, time_steps: {simulation_time_steps}')
            if synaptic_inputs.shape[1] != len(synaptic_input_pops):
                raise ValueError('For each vector of input over time, a target population has to be passed!')
            if synaptic_inputs.shape[1] != len(synaptic_input_syns):
                raise ValueError('For each vector of input over time, a target synapse has to be passed!')

        # check whether extrinsic currents need to be transformed into new, key-based version
        if extrinsic_current is not None and not extrinsic_current_pops:

            extrinsic_current_tmp = list()
            extrinsic_current_pops = list()

            # loop over populations
            for p in range(extrinsic_current.shape[1]):

                curr_tmp = extrinsic_current[:, p].squeeze()

                # store information of the populations for which input is defined
                if np.sum(curr_tmp != 0.) > 0.:
                    extrinsic_current_tmp.append(curr_tmp)
                    extrinsic_current_pops.append(list(self.populations.keys())[p])

            extrinsic_current = np.array(extrinsic_current_tmp, ndmin=2).T

        # extrinsic currents dimensionality check
        if extrinsic_current is not None:
            if extrinsic_current.shape[0] != simulation_time_steps:
                raise ValueError('First dimension of extrinsic current has to match the number of simulation time '
                                 'steps!')
            if extrinsic_current.shape[1] != len(extrinsic_current_pops):
                raise ValueError('For each vector of input current over time, a target population has to be passed!')

        # check whether extrinsic modulations need to be transformed into new, key-based version
        if extrinsic_modulation is not None and not extrinsic_modulation_pops:

            extrinsic_modulation_pops = list()

            # collect relevant population keys
            for p in range(len(extrinsic_modulation)):
                extrinsic_modulation_pops.append(list(self.populations.keys())[p])

        # extrinsic modulation dimensionality check
        if extrinsic_modulation:
            if extrinsic_modulation[0].shape[0] != simulation_time_steps:
                raise ValueError('First dimension of extrinsic modulation arrays has to match the number of simulation '
                                 'time steps!')
            if len(extrinsic_modulation) != len(extrinsic_modulation_pops):
                raise ValueError('For each synaptic modulation array, a target population has to be passed!')

        # save run parameters to instance variable
        ##########################################

        # time vector
        time = (np.arange(0, simulation_time_steps, 1) * self.step_size + self.step_size).tolist()

        # store synaptic input on run_info
        if synaptic_inputs is not None:
            cols = [synaptic_input_pops[i] + '_' + synaptic_input_syns[i] + '_inp'
                    for i in range(synaptic_inputs.shape[1])]
            self.run_info = DataFrame(data=synaptic_inputs, index=time, columns=cols)
        else:
            self.run_info = DataFrame(data=np.zeros((simulation_time_steps, 1)), index=time,
                                      columns=['synaptic_input_to_all_pops'])
        # store extrinsic currents on run_info
        if extrinsic_current_pops:
            for i, pop in enumerate(extrinsic_current_pops):
                self.run_info[pop + '_curr'] = extrinsic_current[:, i]

        # store extrinsic modulations on run_info
        if extrinsic_modulation_pops:
            for i, pop in enumerate(extrinsic_modulation_pops):
                for syn in range(extrinsic_modulation[i].shape[1]):
                    syn_keys = list(self.populations[pop].synapses.keys())
                    self.run_info[pop + '_' + syn_keys[syn] + '_mod'] = extrinsic_modulation[i][:, syn]

        # add dummy populations to graph that handle extrinsic input passage
        ####################################################################

        # synaptic input dummies
        count = 0
        for i, pop in enumerate(synaptic_input_pops):
            self.add_node(SynapticInputPopulation(synaptic_inputs[:, i].squeeze(),
                                                  key='synaptic_input_dummy_' + str(count)),
                          target_nodes=[pop],
                          conn_weights=[1.],
                          conn_targets=[synaptic_input_syns[i]],
                          add_to_population_list=True,
                          key='synaptic_input_dummy_' + str(count))
            count += 1

        # extrinsic current dummies
        if extrinsic_current is not None:
            count = 0
            for i, pop in enumerate(extrinsic_current_pops):
                self.add_node(ExtrinsicCurrentPopulation(extrinsic_current[:, i].squeeze(),
                                                         key='extrinsic_current_dummy_' + str(count)),
                              target_nodes=[pop],
                              conn_weights=[1.],
                              add_to_population_list=True,
                              key='extrinsic_current_dummy_' + str(count))
                count += 1

        # extrinsic modulation dummies
        if extrinsic_modulation is not None:
            count = 0
            for i, pop in enumerate(extrinsic_modulation_pops):
                self.add_node(ExtrinsicModulationPopulation(extrinsic_modulation[i],
                                                            key='extrinsic_modulation_dummy_' + str(count)),
                              target_nodes=[pop],
                              conn_weights=[1.],
                              add_to_population_list=True,
                              key='extrinsic_modulation_dummy_' + str(count))
                count += 1

        # set up observer system
        ########################

        if hasattr(self, 'observer'):

            self.observer.update(circuit=self,
                                 sampling_step_size=sampling_size,
                                 target_populations=observe_populations,
                                 target_states=observe_states)

        else:

            self.observer = CircuitObserver(network=self,
                                            sampling_step_size=sampling_size,
                                            target_populations=observe_populations,
                                            target_states=observe_states)

        return simulation_time_steps

    def run(self,
            synaptic_inputs: np.ndarray,
            simulation_time: float,
            synaptic_input_pops: Optional[List[str]] = None,
            synaptic_input_syns: Optional[List[str]] = None,
            extrinsic_current: Optional[np.ndarray] = None,
            extrinsic_current_pops: Optional[list] = None,
            extrinsic_modulation: Optional[List[np.ndarray]] = None,
            extrinsic_modulation_pops: Optional[list] = None,
            sampling_size: Optional[float] = None,
            observe_populations: Optional[List[str]] = None,
            observe_states: Optional[List[str]] = None,
            verbose: bool = True
            ) -> None:
        """Simulates circuit behavior over time.

        Parameters
        ----------
        synaptic_inputs
            Either 3D array (n_timesteps x n_populations x n_synapses) of synaptic input [unit = 1/s] or 2D array
            (n_timesteps x n_populations). In the latter case, n_populations refers to the number of populations for
            which input is not zero for all time-steps.
        synaptic_input_pops
            Keys of the populations that receive synaptic inputs. Only needed if `synaptic_input` is a 2D array.
        synaptic_input_syns
            Keys of the synapses that receive synaptic inputs. Only needed if `synaptic_input` is a 2D array.
        simulation_time
            Total simulation time [unit = s].
        extrinsic_current
            2D array (n_timesteps x n_populations) of current extrinsically applied to the populations [unit = A].
        extrinsic_current_pops
            Keys of the populations that receive extrinsic currents. If this is passed, extrinsic_current only needs to
            be defined for the populations referenced by this list.
        extrinsic_modulation
            2D array (n_timesteps x n_synapses) with synapse scalings [unit = 1] for each population.
        extrinsic_modulation_pops
            Keys of the populations that receive extrinsic modulation. If this is passed, extrinsic_modulation only
            needs to be defined for the populations referenced by this list.
        sampling_size
            Can be passed to define the step-size with which simulation results are stored on the circuit observer.
        observe_populations
            Keys of the populations for which simulation results are to be stored on the circuit observer.
        observe_states
            Names of the population states that are to be stored on the circuit observer.
        verbose
            If true, simulation progress will be printed to console.

        """

        simulation_time_steps = self._prepare_run(synaptic_inputs=synaptic_inputs,
                                                  synaptic_input_pops=synaptic_input_pops,
                                                  synaptic_input_syns=synaptic_input_syns,
                                                  simulation_time=simulation_time,
                                                  extrinsic_current=extrinsic_current,
                                                  extrinsic_current_pops=extrinsic_current_pops,
                                                  extrinsic_modulation=extrinsic_modulation,
                                                  extrinsic_modulation_pops=extrinsic_modulation_pops,
                                                  sampling_size=sampling_size,
                                                  observe_populations=observe_populations,
                                                  observe_states=observe_states)

        # simulate network
        ##################

        # remove some of the overhead of reading everything from the graph.
        active_populations = []
        for _, node in self.network_graph.nodes(data=True):
            active_populations.append(node['data'])

        for n in range(simulation_time_steps):  # can't think of a way to remove that loop. ;-)

            # pass information through circuit
            for source_pop in active_populations:
                source_pop.project_to_targets()

            # update all population states
            for i, (_, pop) in enumerate(self.populations.items()):
                pop.state_update()

            # display simulation progress
            if verbose:
                if n == 0 or (n % (simulation_time_steps // 10)) == 0:
                    simulation_progress = (self.t / simulation_time) * 100.
                    print(f'simulation progress: {simulation_progress:.0f} %')

            # update time-variant variables
            self.t += self.step_size

            # observe system variables
            self.observer.store_state_variables(circuit=self)

        self.clean_run()

    def clean_run(self):
        """Cleans up any fields not needed after run anymore.
        """

        # remove input dummy nodes from graph
        nodes = list(self.populations.keys())
        for node in nodes:
            if 'dummy' in node:
                self.network_graph.remove_node(node)
                self.n_nodes -= 1
                if node in self.populations.keys():
                    self.populations.pop(node)
                    self.n_populations -= 1

    def get_population_states(self,
                              state_variable: str = 'membrane_potential',
                              population_keys: Optional[List[str]] = None,
                              time_window: Optional[List[float]] = None
                              ) -> np.ndarray:
        """Extracts specified state variable from populations and puts them into matrix. This serves the purpose of
        collecting data for later analysis.

        Parameters
        ----------
        state_variable
            Name of state variable that is to be extracted.
        population_keys
            List with population keys for which to extract states.
        time_window
            Start and end of time window for which to extract the state variables [unit = s].

        Returns
        -------
        np.ndarray
            2D array (n_timesteps x n_populations) of state variable entries.

        """

        # check input parameters
        ########################

        if time_window and any(time_window) < 0:
            raise ValueError('Time constants cannot be negative.')

        if not population_keys:
            population_keys = [key for key in self.populations.keys() if 'dummy' not in key]

        # extract state variables from populations
        ##########################################

        state_idx = self.observer.target_states.index(state_variable)
        population_idx = [self.observer.population_labels.index(key) for key in population_keys]

        # get states from populations for all time-steps
        states = np.array([self.observer.states[state_idx][pop] for pop in population_idx]).T

        # reduce states to time-window
        if time_window:
            states = states[int(time_window[0] / self.step_size):int(time_window[1] / self.step_size), :]

        return states

    def build_graph(self,
                    source_pops: List[str],
                    target_pops: List[str],
                    target_syns: List[str],
                    conn_strengths: List[float],
                    conn_delays: List[int],
                    plasticity_on_input: bool = True):
        """Builds graph from circuit information.
        """

        # initialize graph with nodes
        #############################

        # create empty directed multi-graph
        network_graph = WrappedMultiDiGraph()

        # add populations as network nodes
        for key in self.populations:
            network_graph.add_node(key, data=self.populations[key])

        # build edges
        #############

        pop_keys = list(self.populations.keys())

        # loop over all connections
        for source, target, weight, delay, syn in zip(source_pops, target_pops, conn_strengths, conn_delays,
                                                      target_syns):

            # turn unit of delays from second into step-size
            delay = int(delay / self.step_size)

            # check for populations matching the passed source & population strings
            source_matches = list()
            target_matches = list()
            for pop_key in pop_keys:

                pop_key_split = pop_key.split('_')

                # check for match in source string
                append = True
                for s in source.split('_'):
                    s_found = False
                    for p in pop_key_split:
                        if s == p:
                            s_found = True
                            break
                    if not s_found:
                        append = False
                        break
                if append:
                    source_matches.append(pop_key)

                # check for match in target string
                append = True
                for t in target.split('_'):
                    t_found = False
                    for p in pop_key_split:
                        if t == p:
                            t_found = True
                            break
                    if not t_found:
                        append = False
                        break
                if append:
                    target_matches.append(pop_key)

            # establish all-to-all connection between the matches
            for s in source_matches:

                for t in target_matches:

                    # check whether target synapse is plastic or not
                    syn_idx = list(self.populations[t].synapses.keys()).index(syn)
                    if self.populations[t].features['synaptic_plasticity'] and \
                            self.populations[t].synapse_efficacy_adaptation[syn_idx]:

                        syn_copy = self.populations[t].copy_synapse(syn,
                                                                    max_firing_rate=self.populations
                                                                    [source].axon.transfer_function_args
                                                                    ['max_firing_rate'] * weight)
                        network_graph.add_edge(s, t, weight=weight, delay=delay, synapse=syn_copy)

                    else:

                        network_graph.add_edge(s, t, weight=weight, delay=delay,
                                               synapse=self.populations[t].synapses[syn])

        # turn of plasticity at original synapse if wished
        ##################################################

        if not plasticity_on_input:
            for i, (key, pop) in enumerate(self.populations.items()):
                for j, syn in enumerate(pop.synapses.keys()):
                    if pop.features['synaptic_plasticity'] and pop.synapse_efficacy_adaptation[j] and \
                            'copy' not in syn:
                        pop.synapse_efficacy_adaptation[j] = None
                        pop.synapse_efficacy_adaptation_args[j] = None

        return network_graph

    def add_node(self,
                 data: Union[Population, SynapticInputPopulation, ExtrinsicCurrentPopulation,
                             ExtrinsicModulationPopulation],
                 target_nodes: List[str],
                 conn_weights: List[float],
                 conn_targets: Optional[List[str]] = None,
                 conn_delays: Optional[List[float]] = None,
                 key: Optional[str] = None,
                 add_to_population_list: bool = False
                 ) -> None:
        """Adds node to network graph (and if wished to population list).

        Parameters
        ----------
        data
            Object/instance to be stored on node. Needs to have a `get_firing_rate` method.
        target_nodes
            List with indices for network nodes to project to.
        conn_weights
            List with weight for each target population.
        conn_targets
            List with labels indicating the target objects on the respective populations. Each target has to have a
            `pass_input` method.
        conn_delays
            Delays for each connection.
        key
            Custom key for graph node. If None, integer index will be used.
        add_to_population_list
            If true, element will be added to the circuit's population list.

        """

        # check attributes
        ##################

        if conn_delays is None:
            conn_delays = [0 for _ in range(len(conn_weights))]

        if not key:
            key = str(len(self.network_graph.nodes))

        conn_targets = make_iterable(conn_targets, len(target_nodes))

        # add to network graph
        ######################

        # add node
        self.network_graph.add_node(key, data=data)

        # add edges
        for target, weight, delay, syn in zip(target_nodes, conn_weights, conn_delays, conn_targets):
            self.add_edge(key, target, weight=weight, delay=int(delay / self.step_size), synapse=syn)

        self.n_nodes += 1

        # add to population list
        ########################

        if add_to_population_list:
            self.populations[key] = data
            self.n_populations += 1

    def add_edge(self,
                 source: str,
                 target: str,
                 weight: float,
                 delay: float,
                 synapse: Optional[str] = None,
                 copy_synapse: bool = False,
                 copy_key: Optional[str] = None
                 ) -> None:
        """Adds an edge to the network graph.

        Parameters
        ----------
        source
            Name of source node.
        target
            Name of target node.
        weight
            Connection strength.
        delay
            Information transfer delay.
        synapse
            Label of synapse on target population.
        copy_synapse
            If true, a copy of the given synapse will be created and used to connect source to target.
        copy_key
            Name of new synapse. If None, 'synapse.key' + '_copy' will be used.

        """

        # check passed synapse type
        ###########################

        synapse_keys = list(self.populations[target].synapses.keys())

        # check for a match between the passed synapse key and the synapse keys on the target population
        if synapse and synapse not in synapse_keys:

            for i, syn in enumerate(synapse_keys):

                # if full synapse key is found in one of the synapse names on the target populations, use this name
                if synapse in syn:
                    synapse = syn
                    break

                # raise error if no match was found
                elif i >= len(synapse_keys):
                    raise ValueError('No matching synapse was found on population ' + target + ' for synapse key '
                                     + synapse)

        # add edge
        ##########

        if synapse:

            if copy_synapse:

                # copy the target synapse
                synapse_copy = self.populations[target].copy_synapse(synapse,
                                                                     copy_key=copy_key,
                                                                     max_firing_rate=self.populations
                                                                     [source].axon.transfer_function_args
                                                                     ['max_firing_rate'] * weight)

                # connect source population to the copy
                self.network_graph.add_edge(source, target, weight=weight, delay=delay, synapse=synapse_copy)

            else:

                # connect source population to the target synapse
                self.network_graph.add_edge(source, target, weight=weight, delay=delay,
                                            synapse=self.populations[target].synapses[synapse])

        else:

            # connect source population to target population
            self.network_graph.add_edge(source, target, weight=weight, delay=delay)

    def clear(self):
        """Clears states of all populations
        """

        for _, pop in self.populations.items():
            pop.clear()

        if hasattr(self, 'observer'):
            self.observer.clear()

    def plot_population_states(self,
                               population_keys: Optional[List[str]] = None,
                               state_variable: str = 'membrane_potential',
                               time_window: Optional[Union[np.ndarray, list]] = None,
                               create_plot: bool = True,
                               axis: Optional[Axes] = None
                               ) -> object:
        """Creates figure with population states over time.

        Parameters
        ----------
        population_keys
            Labels of populations for which to plot states over time.
        state_variable
            State variable name.
        time_window
            Start and end time point for which to plot population states.
        create_plot
            If true, plot will be shown.
        axis
            Optional axis handle.

        Returns
        -------
        object
            Axis handle.

        """

        # get population states
        #######################

        if not population_keys:
            population_keys = [key for key in self.populations.keys() if 'dummy' not in key]

        population_states = self.get_population_states(state_variable, population_keys, time_window)

        # plot population states over time
        ##################################

        if axis is None:
            fig, axis = plt.subplots(num='Population States')
        else:
            fig = axis.get_figure()

        legend_labels = []
        for i in range(population_states.shape[1]):
            axis.plot(population_states[:, i])
            legend_labels.append(self.populations[population_keys[i]])

        axis.legend(legend_labels)
        axis.set_ylabel('membrane potential [V]')
        axis.set_xlabel('time-steps')

        if create_plot:
            fig.show()

        return axis


############################################################
# constructor for circuits from synapse + axon information #
############################################################

class CircuitFromScratch(Circuit):
    """Circuit class that requires information about each synapse/axon in the circuit to be build.

    Parameters
    ----------
    connectivity
        See docstring for parameter `connectivity` of :class:`Circuit`.
    delays
        See docstring for parameter `delays` of :class:`Circuit`.
    delay_distributions
        See docstring for parameter `delay_distributions` of :class:`Circuit`.
    source_pops
        See docstring for parameter `source_pops` of :class:`Circuit`.
    target_pops
        See docstring for parameter `target_pops` of :class:`Circuit`.
    target_syns
        See docstring for parameter `target_syns` of :class:`Circuit`.
    step_size
        See docstring for parameter `step_size` of :class:`Circuit`.
    synapses
        Either list of synapse types that exist in circuit (refers to the third dimension of connectivity) or list of
        list of synapse types for each population.
    axons
        List of axon types that exist on each population (n_populations strings).
    synapse_params
        Lists with dictionaries that include name-value pairs for each synapse parameter. Needs to be specified for each
        synapse in network or each synapse on each population.
    axon_params
        List of dictionaries (with n_populations entries) that include name-value pairs for each axon parameter.
    synapse_class
        Name of the synapse class the synapses belong to.
    axon_class
        Name of the axon class the axons belong to.
    synapse_keys
        Keys that will be used to identify the synapses in the network.
    max_synaptic_delay
        Maximum time after which input can still affect synapses [unit = s].
    membrane_capacitance
        Population membrane capacitance [unit = q/S].
    tau_leak
        Time-constant of population leak current [unit = s].
    resting_potential
        Population resting-state membrane potential [unit = V].
    init_states
        2D array containing the initial values for each state variable (2.dim) of each population (1.dim).
    population_keys
        Keys that will be used to identify the populations in the network.
    spike_frequency_adaptation
        Function that describes the axon's `adaptation` parameter dynamics.
    spike_frequency_adaptation_kwargs
        Keyword  arguments of the spike_frequency_adaptation functions.
    synapse_efficacy_adaptation
        Function that describes the synapse's `depression` parameter dynamics.
    synapse_efficacy_adaptation_kwargs
        Keyword  arguments of the synapse_efficacy_adaptation functions.
    enable_modulation
        If true, extrinsic modulations for the synapses can be defined in the `run()` method.

    See Also
    --------
    :class:`Circuit`: Detailed explanation of attributes and methods on circuit.

    """

    def __init__(self,
                 connectivity: Union[np.ndarray, List[float]],
                 delays: Optional[Union[np.ndarray, List[float]]] = None,
                 delay_distributions: Optional[np.ndarray] = None,
                 source_pops: Optional[list] = None,
                 target_pops: Optional[list] = None,
                 target_syns: Optional[Union[np.ndarray, list]] = None,
                 step_size: float = 5e-4,
                 synapses: Optional[Union[List[str], List[List[str]]]] = None,
                 axons: Optional[List[str]] = None,
                 synapse_params: Optional[Union[List[dict], List[List[dict]]]] = None,
                 axon_params: Optional[List[dict]] = None,
                 synapse_class: Union[str, List[str]] = 'DoubleExponentialSynapse',
                 axon_class: Union[str, List[str]] = 'SigmoidAxon',
                 synapse_keys: Optional[list] = None,
                 max_synaptic_delay: Union[float, List[float]] = 0.05,
                 membrane_capacitance: Optional[Union[float, List[float]]] = None,
                 tau_leak: Optional[Union[float, List[float]]] = None,
                 resting_potential: Optional[Union[float, List[float]]] = None,
                 init_states: Union[float, np.ndarray] = 0.,
                 population_keys: Optional[List[str]] = None,
                 spike_frequency_adaptation: Optional[List[Callable[[float], float]]] = None,
                 spike_frequency_adaptation_kwargs: Optional[List[dict]] = None,
                 synapse_efficacy_adaptation: Optional[List[List[Callable[[float], float]]]] = None,
                 synapse_efficacy_adaptation_kwargs: Optional[List[List[dict]]] = None,
                 enable_modulation: bool = False
                 ) -> None:
        """Instantiates circuit from synapse/axon types and parameters.
        """

        # check input parameters
        ########################

        # check whether synapse information was given
        if synapses:
            n_synapses = len(synapses)
        elif synapse_params:
            n_synapses = len(synapse_params)
        else:
            raise AttributeError('Either synapses or synapse_params have to be passed!')

        # check whether axon information was given
        if axons:
            n_pops = len(axons)
        elif axon_params:
            n_pops = len(axon_params)
        else:
            raise AttributeError('Either axons or axon_params have to be passed!')

        # check attribute dimensions
        if type(connectivity) is np.ndarray:
            if n_synapses != connectivity.shape[2]:
                raise AttributeError('Number of synapses needs to match the third dimension of connectivity if a '
                                     'connectivity matrix is passed. If you wish to define synapses for each population'
                                     'use the list option to set the connectivities.')

            if n_pops != connectivity.shape[0]:
                raise AttributeError('Number of axons needs to match the first dimension of connectivity. '
                                     'See docstrings for further explanation.')
        else:
            if n_synapses != n_pops:
                raise AttributeError('If connectivities are passed via lists, synapse information has to be passed for'
                                     'each synapse of population in network.')

        # prepare arguments for population instantiation
        ################################################

        # check information transmission delay
        if delays is None:
            delays = np.zeros((n_pops, n_pops), dtype=int) if type(connectivity) is np.ndarray else \
                [0. for _ in range(len(connectivity))]

        # ensure that population specific variables are iterable
        init_states = make_iterable(init_states, n_pops)
        max_synaptic_delay = make_iterable(max_synaptic_delay, n_pops)
        tau_leak = make_iterable(tau_leak, n_pops)
        resting_potential = make_iterable(resting_potential, n_pops)
        membrane_capacitance = make_iterable(membrane_capacitance, n_pops)
        synapses = make_iterable(synapses, n_synapses)
        synapse_params = make_iterable(synapse_params, n_synapses)
        axons = make_iterable(axons, n_pops)
        axon_params = make_iterable(axon_params, n_pops)
        spike_frequency_adaptation = make_iterable(spike_frequency_adaptation, n_pops)
        spike_frequency_adaptation_kwargs = make_iterable(spike_frequency_adaptation_kwargs, n_pops)
        synapse_efficacy_adaptation = make_iterable(synapse_efficacy_adaptation, n_pops)
        synapse_efficacy_adaptation_kwargs = make_iterable(synapse_efficacy_adaptation_kwargs, n_pops)
        population_keys = make_iterable(population_keys, n_pops)
        synapse_class = make_iterable(synapse_class, n_pops)
        axon_class = make_iterable(axon_class, n_pops)

        # ensure synapse specific variables exist for each synapse of each population
        for i, syn in enumerate(synapse_efficacy_adaptation_kwargs):
            if n_synapses == n_pops:
                n_syns = len(synapses[i]) if synapses[i] else len(synapse_params[i])
                syn = make_iterable(syn, n_syns)
            else:
                syn = make_iterable(syn, n_synapses)
            synapse_efficacy_adaptation_kwargs[i] = syn

        # instantiate each population
        #############################

        populations = list()  # type: List[Population]
        for i in range(n_pops):

            # check synapses that exist at respective population
            if isinstance(connectivity, np.ndarray):

                # get indices of synapses existing at that population
                idx = (np.sum(connectivity[i, :, :], axis=0) != 0).squeeze()
                idx = np.asarray(idx.nonzero(), dtype=int)
                if len(idx) == 1:
                    idx = idx[0]

                # reduce synapse arguments to the existing synapses
                synapses_tmp = [synapses[j] for j in idx]
                synapse_params_tmp = [synapse_params[j] for j in idx]
                synapse_class_tmp = [synapse_class[j] for j in idx]
                if synapse_efficacy_adaptation[i]:
                    synapse_efficacy_adaptation_args_tmp = [synapse_efficacy_adaptation_kwargs[i][j] for j in idx]
                    synapse_efficacy_adaptation_tmp = [synapse_efficacy_adaptation[i][j] for j in idx]
                else:
                    synapse_efficacy_adaptation_args_tmp = None
                    synapse_efficacy_adaptation_tmp = None

            else:

                # get synapse arguments for the population
                synapses_tmp = synapses[i]
                synapse_params_tmp = synapse_params[i]
                synapse_class_tmp = synapse_class[i]
                synapse_efficacy_adaptation_tmp = synapse_efficacy_adaptation[i]
                synapse_efficacy_adaptation_args_tmp = synapse_efficacy_adaptation_kwargs[i]

            # create dictionary with relevant parameters
            pop_params = {'synapses': synapses_tmp, 'axon': axons[i], 'init_state': init_states[i],
                          'step_size': step_size, 'max_synaptic_delay': max_synaptic_delay[i],
                          'synapse_params': synapse_params_tmp, 'axon_params': axon_params[i],
                          'synapse_class': synapse_class_tmp, 'axon_class': axon_class[i],
                          'key': population_keys[i],
                          'synapse_keys': synapse_keys[0:len(synapses_tmp)] if synapse_keys else None,
                          'spike_frequency_adaptation': spike_frequency_adaptation[i],
                          'spike_frequency_adaptation_kwargs': spike_frequency_adaptation_kwargs[i],
                          'synapse_efficacy_adaptation': synapse_efficacy_adaptation_tmp,
                          'synapse_efficacy_adaptation_kwargs': synapse_efficacy_adaptation_args_tmp,
                          'tau_leak': tau_leak[i], 'resting_potential': resting_potential[i],
                          'membrane_capacitance': membrane_capacitance[i], 'enable_modulation': enable_modulation}

            # instantiate population and add it to list
            populations.append(Population(**pop_params))

        # call super init
        #################

        super().__init__(populations=populations,
                         connectivity=connectivity,
                         source_pops=source_pops,
                         target_pops=target_pops,
                         target_syns=target_syns,
                         synapse_keys=synapse_keys,
                         delays=delays,
                         delay_distributions=delay_distributions,
                         step_size=step_size)


###############################################################
# constructor that builds circuit from population information #
###############################################################

class CircuitFromPopulations(Circuit):
    """Circuit class that requires information about each population in the circuit to be build.

    Parameters
    ----------
    population_types
        List of population types (n_populations strings).
    connectivity
        See docstring for parameter `connectivity` of :class:`Circuit`.
    delays
        See docstring for parameter `delays` of :class:`Circuit`.
    delay_distributions
        See docstring for parameter `delay_distributions` of :class:`Circuit`.
    source_pops
        See docstring for parameter `source_pops` of :class:`Circuit`.
    target_pops
        See docstring for parameter `target_pops` of :class:`Circuit`.
    target_syns
        See docstring for parameter `target_syns` of :class:`Circuit`.
    synapse_keys
        See docstring for parameter `synapse_keys` of :class:`Circuit`.
    step_size
        See docstring for parameter `step_size` of :class:`Circuit`.
    max_synaptic_delay
        Maximum time after which input can still affect synapses [unit = s].
    membrane_capacitance
        Population membrane capacitance [unit = q/S].
    tau_leak
        Time-constant of population leak current [unit = s].
    resting_potential
        Population resting-state membrane potential [unit = V].
    init_states
        2D array containing the initial values for each state variable (2.dim) of each population (1.dim).
    population_keys
        Label of each population.

    See Also
    --------
    :class:`Circuit`: Detailed explanation of attributes and methods on circuit.

    """

    def __init__(self,
                 population_types: List[str],
                 connectivity: Union[np.ndarray, List[float]],
                 delays: Optional[Union[np.ndarray, List[float]]] = None,
                 delay_distributions: Optional[np.ndarray] = None,
                 source_pops: Optional[list] = None,
                 target_pops: Optional[list] = None,
                 target_syns: Optional[Union[np.ndarray, list]] = None,
                 synapse_keys: Optional[list] = None,
                 step_size: float = 5e-4,
                 max_synaptic_delay: Union[float, List[float]] = 0.05,
                 membrane_capacitance: Union[float, List[float]] = 1e-12,
                 tau_leak: Optional[Union[float, List[float]]] = None,
                 resting_potential: Union[float, List[float]] = -0.075,
                 init_states: Union[float, np.ndarray] = 0.,
                 population_keys: Optional[List[str]] = None
                 ) -> None:
        """Instantiates circuit from population types and parameters.
        """

        # check input parameters
        ########################

        # attribute dimensions
        if isinstance(connectivity, np.ndarray) and len(population_types) != connectivity.shape[0]:
            raise AttributeError('One population type per population in circuit has to be passed. See parameter '
                                 'docstring for further information.')

        # prepare arguments for population instantiation
        ################################################

        n_pops = len(population_types)

        # check information transmission delay
        if delays is None:
            delays = np.zeros((n_pops, n_pops), dtype=int) if type(connectivity) is np.ndarray else \
                [0. for _ in range(len(connectivity))]

        # make sure that population specific variables are iterable
        init_states = make_iterable(init_states, n_pops)
        max_synaptic_delay = make_iterable(max_synaptic_delay, n_pops)
        tau_leak = make_iterable(tau_leak, n_pops)
        resting_potential =make_iterable(resting_potential, n_pops)
        membrane_capacitance = make_iterable(membrane_capacitance, n_pops)
        population_keys = make_iterable(population_keys, n_pops)

        # instantiate each population
        #############################

        populations = list()
        for i in range(n_pops):

            # create parameter dict
            pop_params = {'init_state': init_states[i],
                          'step_size': step_size,
                          'max_synaptic_delay': max_synaptic_delay[i],
                          'key': population_keys[i],
                          }

            if tau_leak[i]:
                pop_params['tau_leak'] = tau_leak[i]
                pop_params['resting_potential'] = resting_potential[i]
                pop_params['membrane_capacitance'] = membrane_capacitance[i]

            # instantiate population and add it to list
            populations.append(set_instance(Population, population_types[i], **pop_params))

        # call super init
        #################

        super().__init__(populations=populations,
                         connectivity=connectivity,
                         source_pops=source_pops,
                         target_pops=target_pops,
                         target_syns=target_syns,
                         synapse_keys=synapse_keys,
                         delays=delays,
                         delay_distributions=delay_distributions,
                         step_size=step_size)


#####################################################
# constructor that builds circuit from sub-circuits #
#####################################################

class CircuitFromCircuit(Circuit):
    """Circuit class that builds higher-lvl circuit from multiple lower-lvl circuits.

        Parameters
        ----------
        circuits
            List of circuit instances.
        connectivity
            List with connection strengths for each connection between circuits
        delays
            See docstring for parameter `delays` of :class:`Circuit`.
        delay_distributions
            See docstring for parameter `delay_distributions` of :class:`Circuit`.
        target_populations
            List of indices for the input populations of each pair-wise connection.
        source_populations
            List of indices for the output population of each pair-wise connection.
        target_synapses
            List of synapse indices for each connection.
        circuit_keys
            List of strings with circuit names.
        synapse_keys
            See docstring for parameter `synapse_keys` of :class:`Circuit`.

        See Also
        --------
        :class:`Circuit`: Detailed explanation of attributes and methods on circuit.

        """

    def __init__(self,
                 circuits: List[Circuit],
                 connectivity: Union[List[float], np.ndarray],
                 delays: Optional[Union[np.ndarray, List[float]]] = None,
                 delay_distributions: Optional[np.ndarray] = None,
                 source_populations: Optional[List[str]] = None,
                 target_populations: Optional[List[str]] = None,
                 target_synapses: Optional[List[str]] = None,
                 circuit_keys: Optional[List[str]] = None,
                 synapse_keys: Optional[List[str]] = None
                 ) -> None:
        """Instantiates circuit from population types and parameters.
        """

        # set important circuit attributes
        ##################################

        n_circuits = len(circuits)
        step_size = circuits[0].step_size

        # circuit keys/identifiers
        if not circuit_keys:
            circuit_keys = ['Circ' + str(i) for i in range(n_circuits)]

        # initialize collector lists
        connectivity_coll = list()
        delays_coll = list()
        source_pops = list()
        target_pops = list()
        target_syns = list()
        populations = list()

        # collect information from each circuit
        #######################################

        # loop over circuits
        for i in range(n_circuits):

            # collect populations of circuit and update their key
            for key, pop in circuits[i].populations.items():

                # population key updates (add circuit keys to population keys)
                new_key = circuit_keys[i] + '_' + key
                pop.key = new_key
                circuits[i].network_graph = relabel_nodes(circuits[i].network_graph, mapping={key: new_key})
                circuits[i].populations[new_key] = circuits[i].populations.pop(key)

                # add population
                populations.append(pop)

            # collect connectivity information from each circuit
            for key, pop in circuits[i].populations.items():
                for j, target in enumerate(circuits[i].network_graph.adj[key]):
                    connectivity_coll.append(pop.target_weights[j])
                    delays_coll.append(pop.target_delays[j] * step_size)
                    source_pops.append(key)
                    target_pops.append(target)
                    target_syns.append(pop.targets[j].key)

        # set inter-circuit connectivity
        ################################

        # transform all connectivity information to lists
        if isinstance(connectivity, np.ndarray):

            # check synapse information
            if not synapse_keys:
                synapse_keys = ['excitatory', 'inhibitory']
            if len(synapse_keys) != connectivity.shape[2]:
                raise AttributeError('For each entry in the third dimension of connectivity, a synapse type has to be '
                                     'defined!')

            # if not passed, set source + target population and target synapse
            if not source_populations:
                source_populations = list()
            if not target_populations:
                target_populations = list()
            if not target_synapses:
                target_synapses = list()

            count = 0
            for target in range(connectivity.shape[0]):
                for source in range(connectivity.shape[1]):
                    for syn_idx in np.where(connectivity[target, source] > 0.)[0]:
                        if len(source_populations) <= count:
                            source_populations.append(circuit_keys[source] + '_PCs')
                        if len(target_populations) <= count:
                            target_populations.append(circuit_keys[target] + '_PCs')
                        if len(target_synapses) <= count:
                            target_synapses.append(synapse_keys[syn_idx])
                        count += 1

            # find non-zero connections
            idx = connectivity > 0
            idx_delays = np.sum(idx, axis=2) > 0

            # transform connectivity matrix into vector
            connectivity = connectivity[idx]

            # transform delays matrix into list
            if delays is None:
                delays = [0.] * len(connectivity)
            elif isinstance(delays, np.ndarray):
                if len(delays.shape) == 2:
                    delays = delays[idx_delays]
                else:
                    delays_new = list()
                    for d in range(delays.shape[2]):
                        delays_tmp = delays[:, :, d].squeeze()
                        delays_new.append(delays_tmp[idx_delays])
                    delays = delays_new

            # transform delay_distributions matrix into list
            if delay_distributions is not None:
                delay_dists = list()
                for d in range(delays.shape[2]):
                    delay_dists_tmp = delay_distributions[:, :, d].squeeze()
                    delay_dists.append(delay_dists_tmp[idx_delays])
                delay_distributions = delay_dists

        # loop over all connections and add information to collector lists
        for i, (conn, delay, source, target, syn) in enumerate(zip(connectivity, delays, source_populations,
                                                                   target_populations, target_synapses)):

            if isinstance(delay, np.ndarray):

                # if delay distributions were passed, loop over all delays and apply their pdf
                for d, d_prob in zip(delay, delay_distributions[i]):
                    connectivity_coll.append(conn * d_prob)
                    delays_coll.append(d)
                    source_pops.append(source)
                    target_pops.append(target)
                    target_syns.append(syn)

            else:

                # append connectivity infos to lists
                connectivity_coll.append(conn)
                delays_coll.append(delay)
                source_pops.append(source)
                target_pops.append(target)
                target_syns.append(syn)

        # call super init
        #################

        super().__init__(populations=populations,
                         connectivity=connectivity_coll,
                         source_pops=source_pops,
                         target_pops=target_pops,
                         target_syns=target_syns,
                         delays=delays_coll,
                         step_size=step_size)


# def update_step_size(self, new_step_size, synaptic_inputs, update_threshold=1e-2, extrinsic_current=None,
#                      extrinsic_synaptic_modulation=None, idx=0, interpolation_type='linear'):
#     """
#     Updates the time-step size with which the network is simulated.
#
#     :param new_step_size: Scalar, indicates the new simulation step-size [unit = s].
#     :param synaptic_inputs: synaptic input array that needs to be interpolated.
#     :param update_threshold: If step-size ratio (old vs new) is larger than threshold, interpolations are initiated.
#     :param extrinsic_current: extrinsic current array that needs to be interpolated.
#     :param extrinsic_synaptic_modulation: synaptic modulation that needs to be interpolated.
#     :param idx: Can be used to interpolate arrays from this point on (int) (default = 0).
#     :param interpolation_type: character string, indicates the type of interpolation used for up-/down-sampling the
#            firing-rate look-up matrix (default = 'cubic').
#
#     :return interpolated synaptic input array.
#     """
#
#     ############################################
#     # check whether update has to be performed #
#     ############################################
#
#     step_size_ratio = np.max((self.step_size, new_step_size)) / np.min((self.step_size, new_step_size))
#
#     if np.abs(step_size_ratio - 1) > update_threshold and self.time_steps:
#
#         ##########################
#         # update step-size field #
#         ##########################
#
#         step_size_old = self.step_size
#         self.step_size = new_step_size
#
#         ##############################
#         # update firing rate loop-up #
#         ##############################
#
#         # get maximum delay in seconds
#         delay = np.max(np.array(self.D, dtype=int)) + self.step_size
#
#         # check whether update is necessary
#         step_diff = abs(int(delay/self.step_size) - int(delay/step_size_old))
#
#         if step_diff >= 1 and delay > self.step_size:
#
#             # perform interpolation of old firing rate look-up
#             self.firing_rates_lookup = interpolate_array(old_step_size=step_size_old,
#                                                          new_step_size=self.step_size,
#                                                          y=self.firing_rates_lookup,
#                                                          interpolation_type=interpolation_type,
#                                                          axis=1)
#
#         ###############################
#         # update all extrinsic inputs #
#         ###############################
#
#         # check whether update is necessary
#         net_input_time = synaptic_inputs[idx:, :, :].shape[0] * step_size_old
#
#         step_diff = abs(int(net_input_time/self.step_size) - int(net_input_time/step_size_old))
#
#         if step_diff >= 1:
#
#             # perform updates
#             synaptic_inputs = interpolate_array(old_step_size=step_size_old,
#                                                 new_step_size=self.step_size,
#                                                 y=synaptic_inputs[idx:, :, :],
#                                                 axis=0,
#                                                 interpolation_type=interpolation_type)
#
#             if extrinsic_current:
#
#                 extrinsic_current = interpolate_array(old_step_size=step_size_old,
#                                                       new_step_size=self.step_size,
#                                                       y=extrinsic_current[idx:, :],
#                                                       axis=0,
#                                                       interpolation_type=interpolation_type)
#
#             if extrinsic_synaptic_modulation:
#
#                 extrinsic_synaptic_modulation = interpolate_array(old_step_size=step_size_old,
#                                                                   new_step_size=self.step_size,
#                                                                   y=np.array(extrinsic_synaptic_modulation)[idx:, :],
#                                                                   axis=0,
#                                                                   interpolation_type=interpolation_type).tolist()
#
#     return synaptic_inputs, extrinsic_current, extrinsic_synaptic_modulation, 0
