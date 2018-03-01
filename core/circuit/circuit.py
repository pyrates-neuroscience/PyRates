"""Basic neural mass model class plus derivations of it.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
# from networkx import MultiDiGraph
from typing import List, Optional, Union, TypeVar, Callable, Tuple

from core.population import Population, PlasticPopulation, SecondOrderPopulation, SecondOrderPlasticPopulation
from core.population import DummyPopulation
from core.utility import check_nones, set_instance
from core.utility.filestorage import RepresentationBase
from core.utility.networkx_wrapper import WrappedMultiDiGraph

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


#########################
# generic circuit class #
#########################


class Circuit(RepresentationBase):
    """Base circuit class.

    Initializes a number of (delay-) coupled populations (neural masses) that are each characterized by a number of
    synapses and a single axon.

    Parameters
    ----------
    populations
        List of population instances. Need to have same simulation step-size.
    connectivity
        3D array (n_populations x n_populations x n_synapses) with each entry representing the average number of
        synaptic contacts between two populations via a given synapse type. 1. dimension are the receiving populations,
        2. dimension are the sending populations.
    delays
        2D array (n_populations x n_populations) or 3D array (n_populations x n_populations x n_delays) with each entry 
        representing the information transfer delay between the respective populations [unit = s].
    delay_distributions
        3D array (n_populations x n_populations x n_delays) with each entry representing a weight for the 
        corresponding entry in delays [unit = 1].
    step_size
        Simulation step-size [unit = s] (default = 5e-4).

    Attributes
    ----------
    populations
        See description of parameter `populations`
    C
        See description of parameter `connectivity`.
    D
        See description of parameter `delays`.
    step_size
        See description of parameter `step_size`.
    n_populations
        Number of populations in circuit.
    n_synapses
        Number of synapse types in circuit.
    t
        Current time the circuit lives at [unit = s].
    active_synapses
        2D boolean array indicating which synapses (2.dim) exist on which population (1.dim).
    network_graph
        Directional multi-graph representing the network structure.

    """

    def __init__(self,
                 populations: List[Population],
                 connectivity: np.ndarray,
                 delays: np.ndarray,
                 delay_distributions: Optional[np.ndarray] = None,
                 step_size: float = 5e-4,
                 synapse_types: Optional[list] = None,
                 ) -> None:
        """Instantiates population circuit object.
        """

        ##########################
        # check input parameters #
        ##########################

        # attribute dimensions
        if len(populations) != connectivity.shape[0] or len(populations) != connectivity.shape[1]:
            raise AttributeError('First and second dimension of connectivity matrix must equal the number of '
                                 'populations. See parameter docstrings for further explanation.')
        if len(populations) != delays.shape[0] or len(populations) != delays.shape[1]:
            raise AttributeError('First and second dimension of delay matrix must equal the number of populations. '
                                 'See parameter docstrings for further explanation.')
        if not synapse_types and connectivity.shape[2] != 2:
            raise AttributeError('If last dimension of connectivity matrix refers to more or less than two synapses,'
                                 'synapse_types have to be passed.')
        elif synapse_types and len(synapse_types) != connectivity.shape[2]:
            raise AttributeError('Number of passed synapse types has to match the last dimension of connectivity.')

        # attribute values
        # noinspection PyTypeChecker
        if np.sum(delays < 0.) > 0 or step_size < 0.:
            raise ValueError('Time constants (delays, step_size) cannot be negative. See parameter docstrings for '
                             'further explanation.')

        # set circuit attributes
        ########################

        # circuit properties
        self.step_size = step_size
        self.n_populations = len(populations)
        self.n_synapses = connectivity.shape[2]
        self.t = 0.
        self.synapse_mapping = np.zeros((self.n_populations, self.n_synapses), dtype=int) - 999
        self.synapse_types = synapse_types if synapse_types else ['excitatory', 'inhibitory']
        self.run_info = dict()

        # circuit structure
        self.populations = populations
        self.C = connectivity
        self.D = delays
        self.D_pdfs = delay_distributions

        # population specific properties
        for i in range(self.n_populations):

            # make sure state variable history will be saved on population
            self.populations[i].store_state_variables = True

            # create mapping between last dimension of connectivity matrix and synapses existing at each population
            for j in range(self.n_synapses):
                for k in range(self.populations[i].n_synapses):
                    if self.synapse_types[j] in self.populations[i].synapse_labels[k]:
                        self.synapse_mapping[i, j] = k
                        break

            # update max_population_delays according to information in D
            self.populations[i].max_population_delay = np.max(self.D[i, :])
            self.populations[i].update()

            # TODO: make sure that population step-size corresponds to circuit step-size

        # transform delays into steps
        # TODO: check for dtype inside array
        self.D = np.array(self.D / self.step_size, dtype=int)

        # create network graph
        ######################

        self.network_graph = self.build_graph(plasticity_on_input=False)
        self.n_nodes = len(self.network_graph.nodes)

    # noinspection PyPep8Naming
    @property
    def N(self):
        """Read-only value for number of populations. Just keeping this as a safeguard for legacy code."""
        return self.n_populations

    def _prepare_run(self,
                     synaptic_inputs: np.ndarray,
                     simulation_time: float,
                     extrinsic_current: Optional[np.ndarray] = None,
                     extrinsic_modulation: Optional[List[List[np.ndarray]]] = None,
                     ) -> Tuple[int, np.ndarray, List[List[np.ndarray]]]:
        """Helper method to check inputs to run, but keep run function a bit cleaner."""

        # save run parameters to instance variable
        ##########################################

        self.run_info = dict(synaptic_inputs=synaptic_inputs, time_vector=[self.t],
                             extrinsic_current=extrinsic_current, extrinsic_modulation=extrinsic_modulation)

        # check input parameters
        ########################

        # simulation time
        if simulation_time < 0.:
            raise ValueError('Simulation time cannot be negative.')
        simulation_time_steps = int(simulation_time / self.step_size)

        # synaptic inputs
        if synaptic_inputs.shape[0] != simulation_time_steps:
            raise ValueError('First dimension of synaptic input has to match the number of simulation time steps! \n'
                             f'input: {synaptic_inputs.shape[0]}, time_steps: {simulation_time_steps}')
        if synaptic_inputs.shape[1] != self.n_populations:
            raise ValueError('Second dimension of synaptic input has to match the number of populations in the '
                             'circuit!')
        if synaptic_inputs.shape[2] != self.n_synapses:
            raise ValueError('Third dimension of synaptic input has to match the number of synapse types in the '
                             'circuit!')

        # extrinsic currents
        if extrinsic_current is None:
            extrinsic_current = np.zeros((simulation_time_steps, self.n_populations))
        else:
            if extrinsic_current.shape[0] != simulation_time_steps:
                raise ValueError('First dimension of extrinsic current has to match the number of simulation time '
                                 'steps!')
            if extrinsic_current.shape[1] != self.n_populations:
                raise ValueError('Second dimension of extrinsic current has to match the number of populations!')

        # extrinsic modulation
        if not extrinsic_modulation:
            extrinsic_modulation = [[np.ones(self.populations[i].n_synapses) for i in range(self.n_populations)]
                                    for _ in range(simulation_time_steps)]
        else:
            if len(extrinsic_modulation) != simulation_time_steps:
                raise ValueError('First dimension of extrinsic modulation has to match the number of simulation '
                                 'time steps!')
            if len(extrinsic_modulation[0]) != self.n_populations:
                raise ValueError('Second dimension of extrinsic modulation has to match the number of populations!')

        # add dummy population to graph for synaptic input distribution
        ################################################################

        for i in range(self.n_populations):
            for j in range(self.n_synapses):
                if np.any(synaptic_inputs[:, i, j]):
                    self.add_node(DummyPopulation(synaptic_inputs[:, i, j].squeeze()),
                                  target_nodes=[i],
                                  conn_weights=[1.],
                                  conn_targets=[self.populations[i].synapses[self.synapse_mapping[i, j]]],
                                  add_to_population_list=False)

        return simulation_time_steps, extrinsic_current, extrinsic_modulation

    def run(self,
            synaptic_inputs: np.ndarray,
            simulation_time: float,
            extrinsic_current: Optional[np.ndarray] = None,
            extrinsic_modulation: Optional[List[List[np.ndarray]]] = None,
            verbose: bool = False
            ) -> None:
        """Simulates circuit behavior over time.

        Parameters
        ----------
        synaptic_inputs
            3D array (n_timesteps x n_populations x n_synapses) of synaptic input [unit = 1/s].
        simulation_time
            Total simulation time [unit = s].
        extrinsic_current
            2D array (n_timesteps x n_populations) of current extrinsically applied to the populations [unit = A].
        extrinsic_modulation
            List of list of vectors (n_timesteps x n_populations x n_synapses) with synapse scalings [unit = 1].
        verbose
            If true, simulation progress will be printed to console.

        """

        prepared_input = self._prepare_run(synaptic_inputs,
                                           simulation_time,
                                           extrinsic_current,
                                           extrinsic_modulation)
        # just to shorten the line
        simulation_time_steps, extrinsic_current, extrinsic_modulation = prepared_input
        # TODO: Why are extrinsic current and extrinsic modulation defined as lists of lists?

        # simulate network
        ##################

        # remove some of the overhead of reading everything from the graph.
        active_populations = []
        for _, node in self.network_graph.nodes(data=True):
            active_populations.append(node["data"])

        for n in range(simulation_time_steps):  # can't think of a way to remove that loop. ;-)

            # pass information through circuit
            for source_pop in active_populations:
                source_pop.project_to_targets()

            # update all population states
            for i, pop in enumerate(self.populations):
                pop.state_update(extrinsic_current=extrinsic_current[n, i],
                                 extrinsic_synaptic_modulation=extrinsic_modulation[n][i])

            # display simulation progress
            if verbose:
                if n == 0 or (n % (simulation_time_steps // 10)) == 0:
                    simulation_progress = (self.t / simulation_time) * 100.
                    print(f'simulation progress: {simulation_progress:.0f} %')

            # update time-variant variables
            self.t += self.step_size
            self.run_info["time_vector"].append(self.t)

    def get_population_states(self,
                              state_variable_idx: int,
                              population_idx: Optional[Union[list, range]] = None,
                              time_window: Optional[List[float]] = None
                              ) -> np.ndarray:
        """Extracts specified state variable from populations and puts them into matrix. This serves the purpose of
        collecting data for later analysis.

        Parameters
        ----------
        state_variable_idx
            Index of state variable that is to be extracted.
        population_idx
            List with population indices for which to extract states.
        time_window
            Start and end of time window for which to extract the state variables [unit = s].

        Returns
        -------
        np.ndarray
            2D array (n_timesteps x n_populations) of state variable entries.

        """

        # check input parameters
        ########################

        if state_variable_idx < 0:
            raise ValueError('Index cannot be negative.')
        if time_window and any(time_window) < 0:
            raise ValueError('Time constants cannot be negative.')

        if not population_idx:
            population_idx = range(self.n_populations)

        # extract state variables from populations
        ##########################################

        # get states from populations for all time-steps
        states = list()
        for idx in population_idx:
            states.append(np.array(self.populations[idx].state_variables, ndmin=2)[:, state_variable_idx])

        _states = np.array(states, ndmin=2).T

        # reduce states to time-window
        if time_window:
            _states = _states[int(time_window[0] / self.step_size):int(time_window[1] / self.step_size), :]

        return _states

    def build_graph(self,
                    plasticity_on_input: bool = False):
        """Builds graph from circuit information.
        """

        # initialize graph with nodes
        #############################

        # create empty directed multi-graph
        network_graph = WrappedMultiDiGraph()

        # add populations as network nodes
        n_synapses_old = np.zeros(self.n_populations, dtype=int)
        for i in range(self.n_populations):
            n_synapses_old[i] = self.populations[i].n_synapses
            network_graph.add_node(i, data=self.populations[i])

        # build edges
        #############

        # loop over all network nodes
        for source in range(self.n_populations):

            # loop over all potential target nodes
            for target in range(self.n_populations):

                # loop over all synapses
                for idx, syn in enumerate(self.synapse_mapping[target]):

                    # check whether source connects to target via syn
                    if self.C[target, source, idx] > 0.:

                        # add edge with edge information depending on a synapse being plastic or not and multiple
                        # axonal delays being passed or not
                        #########################################################################################

                        # if syn is a plastic synapse, add copies of it to target for each connection on syn
                        if isinstance(self.populations[target], PlasticPopulation) and \
                                self.populations[target].synapse_efficacy_adaptation[syn]:

                            # add synapse copy to population
                            self.populations[target].add_plastic_synapse(synapse_idx=syn,
                                                                         max_firing_rate=self.populations[source].axon.
                                                                         transfer_function_args['max_firing_rate'] *
                                                                                         self.C[target, source, idx])

                            # create one edge for each delay between source and target
                            if self.D_pdfs is None:
                                network_graph.add_edge(source, target,
                                                       weight=float(self.C[target, source, idx]),
                                                       delay=int(self.D[source, target]),
                                                       synapse=self.populations[target].synapses[-1])
                            else:
                                for w, d in zip(self.D_pdfs[target, source, :], self.D[target, source, :]):
                                    network_graph.add_edge(source, target,
                                                           weight=float(self.C[target, source, idx]) * w,
                                                           delay=int(d),
                                                           synapse=self.populations[target].synapses[-1])

                        else:

                            # create one edge for each delay between source and target
                            if self.D_pdfs is None:
                                network_graph.add_edge(source, target,
                                                       weight=float(self.C[target, source, idx]),
                                                       delay=int(self.D[source, target]),
                                                       synapse=self.populations[target].synapses[syn])
                            else:
                                for w, d in zip(self.D_pdfs[target, source, :], self.D[target, source, :]):
                                    network_graph.add_edge(source, target,
                                                           weight=float(self.C[target, source, idx]) * w,
                                                           delay=int(d),
                                                           synapse=self.populations[target].synapses[syn])

        # turn of plasticity at original synapse if wished
        if not plasticity_on_input:
            for i, pop in enumerate(self.populations):
                for syn in range(n_synapses_old[i]):
                    if type(pop) is PlasticPopulation and pop.synapse_efficacy_adaptation[syn]:
                        pop.synapse_efficacy_adaptation[syn] = None
                        pop.state_variables[-1].pop(-1)

        return network_graph

    def add_node(self, data: Union[Population, DummyPopulation],
                 target_nodes: List[int],
                 conn_weights: List[float],
                 conn_targets: List[object],
                 conn_delays: Optional[List[float]] = None,
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
            List with entries indicating the target objects on the respective populations. Each target has to have a 
            `pass_input` method.
        conn_delays
            Delays for each connection.
        add_to_population_list
            If true, element will be added to the circuit's population list.
            
        """

        # check attributes
        ##################

        if conn_delays is None:
            conn_delays = [0 for _ in range(len(conn_weights))]

        # add to network graph
        ######################

        # check current number of nodes
        n_nodes = len(self.network_graph.nodes)

        # add node
        self.network_graph.add_node(n_nodes, data=data)

        # add edges
        for target, weight, delay, syn in zip(target_nodes, conn_weights, conn_delays, conn_targets):
            self.network_graph.add_edge(n_nodes, target,
                                        weight=weight,
                                        delay=delay,
                                        synapse=syn)

        self.n_nodes += 1

        # add to population list
        ########################

        if add_to_population_list:
            self.populations.append(data)

    def clear(self):
        """Clears states of all populations
        """

        for pop in self.populations:
            pop.clear()

    def plot_population_states(self,
                               population_idx: Optional[Union[List[int], range]] = None,
                               state_idx: int = 0,
                               time_window: Optional[Union[np.ndarray, list]] = None,
                               create_plot: bool = True,
                               axes: Optional[Axes] = None
                               ) -> object:
        """Creates figure with population states over time.

        Parameters
        ----------
        population_idx
            Index of populations for which to plot states over time.
        state_idx
            State variable index.
        time_window
            Start and end time point for which to plot population states.
        create_plot
            If true, plot will be shown.
        axes
            Optional axis handle.

        Returns
        -------
        object
            Axis handle.

        """

        # check population idx
        ######################

        if population_idx is None:
            population_idx = range(self.n_populations)

        # get population states
        #######################

        population_states = self.get_population_states(state_idx, population_idx, time_window)

        # plot population states over time
        ##################################

        if axes is None:
            fig, axes = plt.subplots(num='Population States')
        else:
            fig = axes.get_figure()

        legend_labels = []
        for i in range(len(population_idx)):
            axes.plot(population_states[:, i])
            legend_labels.append(self.populations[population_idx[i]].label)

        axes.legend(legend_labels)
        axes.set_ylabel('membrane potential [V]')
        axes.set_xlabel('time-steps')

        if create_plot:
            fig.show()

        return axes


############################################################
# constructor for circuits from synapse + axon information #
############################################################


class CircuitFromScratch(Circuit):
    """Circuit class that requires information about each synapse/axon in the circuit to be build.

    Parameters
    ----------
    synapses
        List of synapse types that exist in circuit (n_synapses strings).
    axons
        List of axon types that exist on each population (n_populations strings).
    connectivity
        See docstring for parameter `connectivity` of :class:`Circuit`.
    delays
        See docstring for parameter `delays` of :class:`Circuit`.
    delay_distributions
        See docstring for parameter `delay_distributions` of :class:`Circuit`.
    step_size
        See docstring for parameter `step_size` of :class:`Circuit`.
    synapse_params
        List of dictionaries (with n_synapses entries) that include name-value pairs for each synapse parameter.
    max_synaptic_delay
        Maximum time after which input can still affect synapses [unit = s].
    axon_params
        List of dictionaries (with n_populations entries) that include name-value pairs for each axon parameter.
    membrane_capacitance
        Population membrane capacitance [unit = q/S].
    tau_leak
        Time-constant of population leak current [unit = s].
    resting_potential
        Population resting-state membrane potential [unit = V].
    init_states
        2D array containing the initial values for each state variable (2.dim) of each population (1.dim).
    population_labels
        Labels of populations.
    axon_plasticity_function
        List with functions defining the axonal plasticity mechanisms to be used on each population.
    axon_plasticity_target_param
        List with target parameters of the population axons to which the axonal plasticity function is applied.
    axon_plasticity_function_params
        List with name-value pairs defining the parameters for the axonal plasticity function.
    synapse_plasticity_function
        List of functions defining the synaptic plasticity mechanisms to be used on each population.
    synapse_plasticity_function_params
        List of name-value pairs defining the parameters for the synaptic plasticity function.

    See Also
    --------
    :class:`Circuit`: Detailed explanation of attributes and methods on circuit.

    """

    def __init__(self,
                 connectivity: np.ndarray,
                 delays: Optional[np.ndarray] = None,
                 delay_distributions: Optional[np.ndarray] = None,
                 step_size: float = 5e-4,
                 synapses: Optional[List[str]] = None,
                 axons: Optional[List[str]] = None,
                 synapse_params: Optional[List[dict]] = None,
                 synapse_types: Optional[list] = None,
                 max_synaptic_delay: Union[float, List[float]] = 0.05,
                 synapse_class: Union[str, List[str]] = 'DoubleExponentialSynapse',
                 axon_params: Optional[List[dict]] = None,
                 axon_class: Union[str, List[str]] = 'SigmoidAxon',
                 population_class: Union[List[str], str] = 'Population',
                 membrane_capacitance: Union[float, List[float]] = 1e-12,
                 tau_leak: Union[float, List[float]] = 0.016,
                 resting_potential: Union[float, List[float]] = -0.075,
                 init_states: Union[float, np.ndarray] = 0.,
                 population_labels: Optional[List[str]] = None,
                 spike_frequency_adaptation: Optional[List[Callable[[float], float]]] = None,
                 spike_frequency_adaptation_args: Optional[List[dict]] = None,
                 synapse_efficacy_adaptation: Optional[List[List[Callable[[float], float]]]] = None,
                 synapse_efficacy_adaptation_args: Optional[List[List[dict]]] = None
                 ) -> None:
        """Instantiates circuit from synapse/axon types and parameters.
        """

        # check input parameters
        ########################

        # check whether axon/synapse information was given
        if synapses:
            n_synapses = len(synapses)
        elif synapse_params:
            n_synapses = len(synapse_params)
        else:
            raise AttributeError('Either synapses or synapse_params have to be passed!')

        if axons:
            n_axons = len(axons)
        elif axon_params:
            n_axons = len(axon_params)
        else:
            raise AttributeError('Either axons or axon_params have to be passed!')

        # check attribute dimensions
        if n_synapses != connectivity.shape[2]:
            raise AttributeError('Number of synapses needs to match the third dimension of connectivity. See docstrings'
                                 'for further explanation.')

        if n_axons != connectivity.shape[0]:
            raise AttributeError('Number of axons needs to match the first dimension of connectivity. See docstrings'
                                 'for further explanation.')

        # instantiate populations
        #########################

        N = connectivity.shape[0]

        # check information transmission delay
        if delays is None:
            delays = np.zeros((N, N), dtype=int)

        # set maximum transfer delay for each population
        max_population_delay = np.max(delays, axis=1)

        # make float variables iterable
        if isinstance(init_states, float):
            init_states = np.zeros(N) + init_states

        if isinstance(max_synaptic_delay, float):
            max_synaptic_delay = np.zeros(N) + max_synaptic_delay
        elif not max_synaptic_delay:
            max_synaptic_delay = check_nones(max_synaptic_delay, N)

        if isinstance(tau_leak, float):
            tau_leak = np.zeros(N) + tau_leak

        if isinstance(resting_potential, float):
            resting_potential = np.zeros(N) + resting_potential

        if isinstance(membrane_capacitance, float):
            membrane_capacitance = np.zeros(N) + membrane_capacitance

        # make None and str variables iterable
        synapses = check_nones(synapses, n_synapses)
        synapse_params = check_nones(synapse_params, n_synapses)
        axons = check_nones(axons, N)
        axon_params = check_nones(axon_params, N)
        spike_frequency_adaptation = check_nones(spike_frequency_adaptation, N)
        spike_frequency_adaptation_args = check_nones(spike_frequency_adaptation_args, N)
        synapse_efficacy_adaptation = check_nones(synapse_efficacy_adaptation, N)
        synapse_efficacy_adaptation_args = check_nones(synapse_efficacy_adaptation_args, N)

        for i, syn in enumerate(synapse_efficacy_adaptation_args):
            syn = check_nones(syn, n_synapses)
            synapse_efficacy_adaptation_args[i] = syn
        if not population_labels:
            population_labels = ['Custom' for _ in range(N)]
        if isinstance(synapse_class, str):
            synapse_class = [synapse_class for _ in range(n_synapses)]
        if isinstance(axon_class, str):
            axon_class = [axon_class for _ in range(N)]
        if isinstance(population_class, str):
            population_class = [population_class for _ in range(N)]

        # instantiate each population
        populations = list()  # type: List[Population]
        for i in range(connectivity.shape[0]):

            # check and extract synapses that exist at respective population
            idx = (np.sum(connectivity[i, :, :], axis=0) != 0).squeeze()
            idx = np.asarray(idx.nonzero(), dtype=int)
            if len(idx) == 1:
                idx = idx[0]
            synapses_tmp = [synapses[j] for j in idx]
            synapse_params_tmp = [synapse_params[j] for j in idx]
            synapse_class_tmp = [synapse_class[j] for j in idx]
            if synapse_efficacy_adaptation[i]:
                synapse_efficacy_adaptation_args_tmp = [synapse_efficacy_adaptation_args[i][j] for j in idx]
                synapse_efficacy_adaptation_tmp = [synapse_efficacy_adaptation[i][j] for j in idx]
            else:
                synapse_efficacy_adaptation_args_tmp = None
                synapse_efficacy_adaptation_tmp = None

            # create dictionary with relevant parameters
            pop_params = {'synapses': synapses_tmp,
                          'axon': axons[i],
                          'init_state': init_states[i],
                          'step_size': step_size,
                          'max_synaptic_delay': max_synaptic_delay[i],
                          'max_population_delay': max_population_delay[i],
                          'synapse_params': synapse_params_tmp,
                          'axon_params': axon_params[i],
                          'synapse_class': synapse_class_tmp,
                          'axon_class': axon_class[i],
                          'label': population_labels[i]}

            # add plasticity parameters if necessary
            if population_class[i] == 'SecondOrderPlasticPopulation' or population_class[i] == 'PlasticPopulation':
                pop_params['spike_frequency_adaptation'] = spike_frequency_adaptation[i]
                pop_params['spike_frequency_adaptation_args'] = spike_frequency_adaptation_args[i]
                pop_params['synapse_efficacy_adaptation'] = synapse_efficacy_adaptation_tmp
                pop_params['synapse_efficacy_adaptation_args'] = synapse_efficacy_adaptation_args_tmp

            # add first-order parameters if necessary
            if population_class[i] == 'Population' or population_class[i] == 'PlasticPopulation':
                pop_params['tau_leak'] = tau_leak[i]
                pop_params['resting_potential'] = resting_potential[i]
                pop_params['membrane_capacitance'] = membrane_capacitance[i]

            # instantiate population
            if population_class[i] == 'SecondOrderPlasticPopulation':
                populations.append(SecondOrderPlasticPopulation(**pop_params))
            elif population_class[i] == 'SecondOrderPopulation':
                populations.append(SecondOrderPopulation(**pop_params))
            elif population_class[i] == 'PlasticPopulation':
                populations.append(PlasticPopulation(**pop_params))
            else:
                populations.append(Population(**pop_params))

        # call super init
        #################

        super().__init__(populations=populations,
                         connectivity=connectivity,
                         synapse_types=synapse_types,
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
    population_class
        Class names the population types refer to.
    population_labels
        Label of each population.

    See Also
    --------
    :class:`Circuit`: Detailed explanation of attributes and methods on circuit.

    """

    def __init__(self,
                 population_types: List[str],
                 connectivity: np.ndarray,
                 synapse_types: Optional[list] = None,
                 delays: Optional[np.ndarray] = None,
                 delay_distributions: Optional[np.ndarray] = None,
                 step_size: float = 5e-4,
                 max_synaptic_delay: Union[float, List[float]] = 0.05,
                 membrane_capacitance: Union[float, List[float]] = 1e-12,
                 tau_leak: Union[float, List[float]] = 0.016,
                 resting_potential: Union[float, List[float]] = -0.075,
                 init_states: Union[float, np.ndarray] = 0.,
                 population_class: Union[List[str], str] = 'Population',
                 population_labels: Optional[List[str]] = None
                 ) -> None:
        """Instantiates circuit from population types and parameters.
        """

        # check input parameters
        ########################

        # attribute dimensions
        if len(population_types) != connectivity.shape[0]:
            raise AttributeError('One population type per population in circuit has to be passed. See parameter '
                                 'docstring for further information.')

        # instantiate populations
        #########################

        N = len(population_types)

        # check information transmission delay
        if delays is None:
            delays = np.zeros((N, N), dtype=int)

        # set maximum transfer delay for each population
        max_population_delay = np.max(delays, axis=1)

        # make float variables iterable
        if isinstance(init_states, float):
            init_states = np.zeros(N) + init_states
        if isinstance(max_synaptic_delay, float):
            max_synaptic_delay = np.zeros(N) + max_synaptic_delay
        elif max_synaptic_delay is None:
            max_synaptic_delay = check_nones(max_synaptic_delay, N)
        if isinstance(tau_leak, float):
            tau_leak = np.zeros(N) + tau_leak
        if isinstance(resting_potential, float):
            resting_potential = np.zeros(N) + resting_potential
        if isinstance(membrane_capacitance, float):
            membrane_capacitance = np.zeros(N) + membrane_capacitance

        # make None/str variables iterable
        if not population_labels:
            population_labels = ['Custom' for _ in range(N)]
        if isinstance(population_class, str):
            population_class = [population_class for _ in range(N)]

        # instantiate each population
        populations = list()
        for i in range(N):

            # create parameter dict
            pop_params = {'init_state': init_states[i],
                          'step_size': step_size,
                          'max_synaptic_delay': max_synaptic_delay[i],
                          'max_population_delay': max_population_delay[i],
                          'label': population_labels[i]}

            # pass parameters to instantiation function
            if population_class[i] == 'SecondOrderPlasticPopulation':
                populations.append(set_instance(SecondOrderPlasticPopulation, population_types[i], **pop_params))

            elif population_class[i] == 'SecondOrderPopulation':
                populations.append(set_instance(SecondOrderPopulation, population_types[i], **pop_params))

            else:

                # add first order population parameters
                pop_params['tau_leak'] = tau_leak
                pop_params['resting_potential'] = resting_potential
                pop_params['membrane_capacitance'] = membrane_capacitance

                if population_class[i] == 'PlasticPopulation':
                    populations.append(set_instance(PlasticPopulation, population_types[i], **pop_params))

                else:
                    populations.append(set_instance(Population, population_types[i], **pop_params))

        # call super init
        #################

        super().__init__(populations=populations,
                         connectivity=connectivity,
                         synapse_types=synapse_types,
                         delays=delays,
                         delay_distributions=delay_distributions,
                         step_size=step_size)


#####################################################
# constructor that builds circuit from sub-circuits #
#####################################################


class CircuitFromCircuit(Circuit):
    """Circuit class that builds higher-level circuit from multiple lower-level circuits.

        Parameters
        ----------
        circuits
            List of circuit instances.
        connection_strengths
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
        circuit_labels
            List of strings with circuit labels.

        See Also
        --------
        :class:`Circuit`: Detailed explanation of attributes and methods on circuit.

        """

    def __init__(self,
                 circuits: List[Circuit],
                 connection_strengths: List[float],
                 source_populations: List[int],
                 target_populations: List[int],
                 target_synapses: Optional[List[int]] = None,
                 delays: Optional[List[Union[np.ndarray, float]]] = None,
                 delay_distributions: Optional[List[np.ndarray]] = None,
                 circuit_labels: Optional[List[str]] = None,
                 synapse_types: Optional[List[str]] = None
                 ) -> None:
        """Instantiates circuit from population types and parameters.
        """

        # check input parameters
        ########################

        synapse_types = synapse_types if synapse_types else ['excitatory', 'inhibitory']

        # fetch important circuit attributes
        ####################################

        n_circuits = len(circuits)

        if not circuit_labels:
            circuit_labels = ['' for i in range(n_circuits)]

        # set delays
        if delays is None:
            delays = [0 for _ in range(len(connection_strengths))]

        # set target synapses
        if target_synapses is None:
            target_synapses = [0 for _ in range(len(connection_strengths))]

        # set maximum transfer delay for each population
        max_population_delay = np.max(delays)

        # initialize stuff
        n_populations = np.zeros(n_circuits + 1, dtype=int)
        connectivity_coll = list()
        delays_coll = list()
        delay_distributions_coll = list()
        populations = list()
        n_synapses = np.zeros(n_circuits, dtype=int)
        synapse_mapping = list()

        # loop over circuits
        for i in range(n_circuits):

            # collect population count
            n_populations[i + 1] = n_populations[i] + circuits[i].n_populations

            # collect connectivity matrix
            connectivity_coll.append(circuits[i].C)

            # collect delay matrix and delay distributions
            delays_coll.append(circuits[i].D * circuits[i].step_size)
            delay_distributions_coll.append(circuits[i].D_pdfs)

            # collect number of synapses
            n_synapses[i] = circuits[i].n_synapses

            # collect populations
            for pop in circuits[i].populations:
                # update population label
                pop.label = circuit_labels[i] + '_' + pop.label

                # update population delay
                pop.synaptic_input = np.zeros((int((len(pop.synapses[0].synaptic_kernel) + max_population_delay)),
                                               len(pop.synapses)))

                # add population
                populations.append(pop)

            # extract synapse types
            synapse_types_tmp = circuits[i].synapse_types
            synapse_mapping_tmp = list()
            for syn in synapse_types_tmp:
                synapse_mapping_tmp.append(synapse_types.index(syn))
            synapse_mapping.append(synapse_mapping_tmp)

        # calculate number of synapse types in circuit
        n_synapses = np.max(n_synapses)
        n_synapses = np.max((n_synapses, np.max(target_synapses)))
        n_synapses = np.max((n_synapses, len(synapse_types)))

        # build connectivity matrix
        ###########################

        connectivity = np.zeros((n_populations[-1], n_populations[-1], n_synapses))

        # loop over all circuits
        for i in range(n_circuits):

            # set intra-circuit connectivity of circuit i
            connectivity[n_populations[i]:n_populations[i + 1], n_populations[i]:n_populations[i + 1],
                         np.array(synapse_mapping[i])] = connectivity_coll[i]

        # loop over new connections
        for i, conn in enumerate(connection_strengths):

            # set inter-circuit connections
            connectivity[target_populations[i], source_populations[i], target_synapses[i]] = conn

        # build delay matrix
        ####################

        if delay_distributions is None:

            delays_new = np.zeros((n_populations[-1], n_populations[-1]))
            delay_distributions_new = None

            # loop over all circuits
            for i in range(n_circuits):

                # set intra-circuit delays of circuit i
                delays_new[n_populations[i]:n_populations[i + 1], n_populations[i]:n_populations[i + 1]] = delays_coll[
                    i]

            # loop over new connections
            for i, delay in enumerate(delays):

                # set inter-circuit delays
                delays_new[target_populations[i], source_populations[i]] = delay

        else:

            # calculate maximum number of delays per connection in network
            n_delays = 1
            for d in delay_distributions:
                n_del = len(d)
                n_delays = n_del if n_del > n_delays else n_delays
            for d in delays_coll:
                n_del = d.shape[2]
                n_delays = n_del if n_del > n_delays else n_delays

            delays_new = np.zeros((n_populations[-1], n_populations[-1]), n_delays)
            delay_distributions_new = np.ones((n_populations[-1], n_populations[-1]), n_delays)

            # loop over all circuits
            for i in range(n_circuits):

                # set intra-circuit delays of circuit i
                if len(delays_coll[i].shape) == 2:
                    delays_new[n_populations[i]:n_populations[i + 1], n_populations[i]:n_populations[i + 1], 0] = \
                        delays_coll[i]
                else:
                    delays_new[n_populations[i]:n_populations[i + 1], n_populations[i]:n_populations[i + 1],
                               0:delays_coll[i].shape[2]] = delays_coll[i]
                    delay_distributions_new[n_populations[i]:n_populations[i + 1],
                                            n_populations[i]:n_populations[i + 1],
                                            0:delays_coll[i].shape[2]] = delay_distributions_coll[i]

            # loop over new connections
            for i, delay in enumerate(delays):

                # set inter-circuit delays
                delays_new[source_populations[i], target_populations[i], 0:len(delay)] = delay
                delay_distributions_new[source_populations[i], target_populations[i], 0:len(delay)] = \
                    delay_distributions[i]

        # call super init
        #################

        super().__init__(populations=populations,
                         connectivity=connectivity,
                         synapse_types=synapse_types,
                         delays=delays_new,
                         delay_distributions=delay_distributions_new,
                         step_size=circuits[0].step_size)

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
