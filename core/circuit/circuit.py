"""Basic neural mass model class plus derivations of it.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from networkx import MultiDiGraph
from typing import List, Optional, Union, TypeVar, Callable

from core.population import Population, PlasticPopulation, SecondOrderPopulation, SecondOrderPlasticPopulation
from core.utility import check_nones, set_instance

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


#########################
# generic circuit class #
#########################


class Circuit(object):
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
    N
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
                 step_size: float = 5e-4
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

        # attribute values
        # noinspection PyTypeChecker
        if np.sum(delays < 0.) > 0 or step_size < 0.:
            raise ValueError('Time constants (delays, step_size) cannot be negative. See parameter docstrings for '
                             'further explanation.')

        # set circuit attributes
        ########################

        # circuit properties
        self.step_size = step_size
        self.N = len(populations)
        self.n_synapses = connectivity.shape[2]
        self.t = 0.
        self.active_synapses = list()

        # circuit structure
        self.populations = populations
        self.C = connectivity
        self.D = delays
        self.delay_distributions = delay_distributions

        # population specific properties
        for i in range(self.N):

            # make sure state variable history will be saved on population
            self.populations[i].store_state_variables = True

            # check and extract synapses that exist at respective population
            idx = np.where(np.sum(connectivity[i, :, :].squeeze(), axis=0) != 0)[0]
            self.active_synapses.append(list(idx))

            # check whether max_population_delay has changed
            if np.max(delays[i, :]) > self.populations[i].max_population_delay:
                self.populations[i].max_population_delay = np.max(delays[i, :])
                self.populations[i].set_synapse_dependencies(update=False)

            # TODO: make sure that population step-size corresponds to circuit step-size

        # create network graph
        ######################

        # create empty directed multi-graph
        self.network_graph = MultiDiGraph()

        # add population as network node
        for i in range(self.N):
            self.network_graph.add_node(i, data=self.populations[i])

        # add connections to other network nodes as edges
        for i in range(self.N):
            current_population = self.populations[i]
            new_synapses = list()

            # loop over synapses existing at current_population
            for k2, k in enumerate(self.active_synapses[i]):

                # loop over all network nodes
                for j in range(self.N):

                    # check whether node j connects to current_population via synapse k
                    if self.C[i, j, k] > 0:

                        # if synapse k is plastic, add copy of it to current_population
                        if isinstance(current_population, PlasticPopulation) and \
                                current_population.synapse_plasticity_function_params[k2]:

                            # add synapse copy to population
                            current_population.add_plastic_synapse(synapse_idx=k2,
                                                                   max_firing_rate=self.populations[j].axon.
                                                                   transfer_function_args['max_firing_rate'] * self.C[
                                                                                       i, j, k])
                            new_synapses.append(self.populations[i].n_synapses - 1)

                            # create one edge for each delay between j and i
                            if delay_distributions is None:
                                self.network_graph.add_edge(j, i,
                                                            weight=float(self.C[i, j, k]),
                                                            delay=int(self.D[i, j]),
                                                            synapse_index=new_synapses[-1])
                            else:
                                for w, d in zip(delay_distributions[i, j, :], self.D[i, j, :]):
                                    self.network_graph.add_edge(j, i,
                                                                weight=float(self.C[i, j, k]) * w,
                                                                delay=int(d),
                                                                synapse_index=new_synapses[-1])

                        else:

                            # create one edge for each delay between j and i
                            if delay_distributions is None:
                                self.network_graph.add_edge(j, i,
                                                            weight=float(self.C[i, j, k]),
                                                            delay=int(self.D[i, j]),
                                                            synapse_index=k2)
                            else:
                                for w, d in zip(delay_distributions[i, j, :], self.D[i, j, :]):
                                    self.network_graph.add_edge(j, i,
                                                                weight=float(self.C[i, j, k]) * w,
                                                                delay=int(d),
                                                                synapse_index=k2)

                # turn of plasticity at original synapse
                if isinstance(current_population, PlasticPopulation):
                    current_population.synapse_plasticity_function_params[k2] = None

            self.active_synapses[i] += new_synapses

    def __repr__(self) -> str:
        """Printable representation of the class that can be evaluated with eval and yields the same"""
        module = self.__class__.__module__
        name = self.__class__.__name__
        rep_str = f"{module}.{name}(populations={self.populations}, connectivity={self.C}, delays={self.D}, " \
                  f"delay_distributions={self.delay_distributions}, step_size={self.step_size})"
        return rep_str

    def run(self,
            synaptic_inputs: np.ndarray,
            simulation_time: float,
            extrinsic_current: Optional[np.ndarray] = None,
            extrinsic_modulation: Optional[np.ndarray] = None,
            verbose: bool = False
            ) -> None:
        """Simulates circuit behavior over time.

        Parameters
        ----------
        synaptic_inputs
            3D array (n_timesteps x n_populations x n_synapses) of synaptic input [unit = 1/s].
            # fixme: why have global 3D arrays for synaptic input?
        simulation_time
            Total simulation time [unit = s].
        extrinsic_current
            2D array (n_timesteps x n_populations) of current extrinsically applied to the populations [unit = A].
        extrinsic_modulation
            List (n_timesteps length) of list (with n_populations entries) of vectors with extrinsic modulatory
            influences on each synapse of a population.
        verbose
            If true, simulation progress will be printed to console.

        """

        # check input parameters
        ########################

        # simulation time
        if simulation_time < 0.:
            raise ValueError('Simulation time cannot be negative.')
        simulation_time_steps = int(simulation_time / self.step_size)

        # synaptic inputs
        if synaptic_inputs.shape[0] != simulation_time_steps:
            raise ValueError('First dimension of synaptic input has to match the number of simulation time steps!')
        if synaptic_inputs.shape[1] != self.N:
            raise ValueError('Second dimension of synaptic input has to match the number of populations in the '
                             'circuit!')
        if synaptic_inputs.shape[2] != self.n_synapses:
            raise ValueError('Third dimension of synaptic input has to match the number of synapse types in the '
                             'circuit!')

        # extrinsic currents
        if extrinsic_current is None:
            extrinsic_current = np.zeros((simulation_time_steps, self.N))
        else:
            if extrinsic_current.shape[0] != simulation_time_steps:
                raise ValueError('First dimension of extrinsic current has to match the number of simulation time '
                                 'steps!')
            if extrinsic_current.shape[1] != self.N:
                raise ValueError('Second dimension of extrinsic current has to match the number of populations!')

        # extrinsic modulation
        if not extrinsic_modulation:
            extrinsic_modulation = np.ones((simulation_time_steps, self.N))
        else:
            if extrinsic_modulation.shape[0] != simulation_time_steps:
                raise ValueError('First dimension of extrinsic modulation has to match the number of simulation '
                                 'time steps!')
            if extrinsic_modulation.shape[1] != self.N:
                raise ValueError('Second dimension of extrinsic modulation has to match the number of populations!')

        # simulate network
        ##################

        for n in range(simulation_time_steps):

            # pass information through circuit
            self.pass_through_circuit()

            # update all population states
            self.update_population_states(synaptic_inputs[n, :, :],
                                          extrinsic_current[n, :],
                                          extrinsic_modulation[n, :])  # type: ignore

            # display simulation progress
            if verbose:
                if n == 0 or (n % (simulation_time_steps // 10)) == 0:
                    simulation_progress = (self.t / simulation_time) * 100.
                    print(f'simulation progress: {simulation_progress:.0f} %')

            # update time-variant variables
            self.t += self.step_size

    def update_population_states(self,
                                 synaptic_inputs: np.ndarray,
                                 extrinsic_current: np.ndarray,
                                 extrinsic_modulation: np.ndarray,
                                 ) -> None:
        """Updates states of all populations

        Parameters
        ----------
        synaptic_inputs
            2D array containing the synaptic inputs for each synapse (2.dim) of each population (1.dim).
        extrinsic_current
            vector containing the extrinsic current input to each population.
        extrinsic_modulation
            list (with n_populations entries) of vectors with extrinsic modulatory influences on each synapse of a
            population.

        """

        for i in range(self.N):
            self.populations[i].state_update(synaptic_input=synaptic_inputs[i, self.active_synapses[i][0:self.
                                             n_synapses]],
                                             extrinsic_current=extrinsic_current[i],
                                             extrinsic_synaptic_modulation=extrinsic_modulation[i])

    def pass_through_circuit(self):
        """Passes current population firing rates through circuit.
        """

        for i in range(self.N):
            self.project_to_populations(i)

    def project_to_populations(self, pop: int) -> None:
        """Projects output of given population to the other circuit populations its connected to.

        Parameters
        ----------
        pop
            Index of population where projection origins.

        """

        # extract network connections
        connected_pops = self.network_graph[pop]

        # loop over connections of node
        for target_pop in connected_pops:

            # loop over existing connections between source node and target node
            for conn_idx in connected_pops[target_pop]:
                # transfer input to target node
                self.network_graph.nodes[target_pop]['data'].synaptic_input[
                    self.network_graph.nodes[target_pop]['data'].current_input_idx[
                        connected_pops[target_pop][conn_idx]['synapse_index']] +
                    connected_pops[target_pop][conn_idx]['delay'],
                    connected_pops[target_pop][conn_idx]['synapse_index']] += \
                    self.network_graph.nodes[pop]['data'].current_firing_rate * \
                    connected_pops[target_pop][conn_idx]['weight']

    def get_population_states(self,
                              state_variable_idx: int,
                              population_idx: Optional[Union[list, range]] = None,
                              time_window: Optional[List[float]] = None
                              ) -> np.ndarray:
        """Extracts specified state variable from populations and puts them into matrix.

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
            population_idx = [i for i in range(self.N)]

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

    def clear(self):
        """Clears states of all populations
        """

        for pop in self.populations:
            pop.clear()

    def plot_population_states(self,
                               population_idx: Optional[Union[List[int], range]] = None,
                               state_idx: int = 0,
                               time_window: Optional[np.ndarray] = None,
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
            population_idx = range(self.N)

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
                 max_synaptic_delay: Union[float, List[float]] = 0.05,
                 synaptic_modulation_direction: Optional[List[List[np.ndarray]]] = None,
                 synapse_class: Union[str, List[str]] = 'DoubleExponentialSynapse',
                 axon_params: Optional[List[dict]] = None,
                 axon_class: Union[str, List[str]] = 'SigmoidAxon',
                 population_class: Union[List[str], str] = 'Population',
                 membrane_capacitance: Union[float, List[float]] = 1e-12,
                 tau_leak: Union[float, List[float]] = 0.016,
                 resting_potential: Union[float, List[float]] = -0.075,
                 init_states: Union[float, np.ndarray] = 0.,
                 population_labels: Optional[List[str]] = None,
                 axon_plasticity_function: Optional[List[Callable[[float], float]]] = None,
                 axon_plasticity_target_param: Optional[List[str]] = None,
                 axon_plasticity_function_params: Optional[List[dict]] = None,
                 synapse_plasticity_function: Optional[List[Callable[[float], float]]] = None,
                 synapse_plasticity_function_params: Optional[List[List[dict]]] = None,
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
        else:
            delays = np.array(delays / step_size, dtype=int)

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
        synaptic_modulation_direction = check_nones(synaptic_modulation_direction, N)
        axon_plasticity_function = check_nones(axon_plasticity_function, N)
        axon_plasticity_target_param = check_nones(axon_plasticity_target_param, N)
        axon_plasticity_function_params = check_nones(axon_plasticity_function_params, N)
        synapse_plasticity_function = check_nones(synapse_plasticity_function, N)
        synapse_plasticity_function_params = check_nones(synapse_plasticity_function_params, N)
        for i, syn in enumerate(synapse_plasticity_function_params):
            syn = check_nones(syn, n_synapses)
            synapse_plasticity_function_params[i] = syn
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
            synapse_plasticity_function_params_tmp = [synapse_plasticity_function_params[i][j] for j in idx]

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
                pop_params['axon_plasticity_function'] = axon_plasticity_function[i]
                pop_params['axon_plasticity_target_param'] = axon_plasticity_target_param[i]
                pop_params['axon_plasticity_function_params'] = axon_plasticity_function_params[i]
                pop_params['synapse_plasticity_function'] = synapse_plasticity_function[i]
                pop_params['synapse_plasticity_function_params'] = synapse_plasticity_function_params_tmp

            # add first-order parameters if necessary
            if population_class[i] == 'Population' or population_class[i] == 'PlasticPopulation':
                pop_params['synaptic_modulation_direction'] = synaptic_modulation_direction[i]
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
                 delays: Optional[np.ndarray] = None,
                 delay_distributions: Optional[np.ndarray] = None,
                 step_size: float = 5e-4,
                 max_synaptic_delay: Union[float, List[float]] = 0.05,
                 synaptic_modulation_direction: Optional[List[List[np.ndarray]]] = None,
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
        else:
            delays = np.array(delays / step_size, dtype=int)

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
        synaptic_modulation_direction = check_nones(synaptic_modulation_direction, N)
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
                pop_params['synaptic_modulation_direction'] = synaptic_modulation_direction

                if population_class[i] == 'PlasticPopulation':
                    populations.append(set_instance(PlasticPopulation, population_types[i], **pop_params))

                else:
                    populations.append(set_instance(Population, population_types[i], **pop_params))

        # call super init
        #################

        super().__init__(populations=populations,
                         connectivity=connectivity,
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
        connectivity
            See docstring for parameter `connectivity` of :class:`Circuit`.
        delays
            See docstring for parameter `delays` of :class:`Circuit`.
        delay_distributions
            See docstring for parameter `delay_distributions` of :class:`Circuit`.
        input_populations
            3D list of indices for the input populations (3.dim) of each pair-wise connection (1. + 2.dim).
        output_populations
            2D array of indices for the output population of each pair-wise connection.
        circuit_labels
            List of strings with circuit labels.

        See Also
        --------
        :class:`Circuit`: Detailed explanation of attributes and methods on circuit.

        """

    def __init__(self,
                 circuits: List[Circuit],
                 connectivity: np.ndarray,
                 delays: Optional[np.ndarray] = None,
                 delay_distributions: Optional[np.ndarray] = None,
                 input_populations: Optional[np.ndarray] = None,
                 output_populations: Optional[np.ndarray] = None,
                 circuit_labels: Optional[List[str]] = None
                 ) -> None:
        """Instantiates circuit from population types and parameters.
        """

        # check input parameters
        ########################

        # attribute dimensions
        if len(circuits) != connectivity.shape[0] or len(circuits) != connectivity.shape[1]:
            raise AttributeError('First and second dimension of connectivity need to match the number of circuits. '
                                 'See parameter docstring for further information.')
        if delays is not None and (len(circuits) != delays.shape[0] or len(circuits) != delays.shape[1]):
            raise AttributeError('First and second dimension of delays need to match the number of circuits. '
                                 'See parameter docstring for further information.')

        # fetch important circuit attributes
        ####################################

        n_circuits = len(circuits)

        if not circuit_labels:
            circuit_labels = ['' for i in range(n_circuits)]

        # set delays
        if delays is None:
            delays = np.zeros((n_circuits, n_circuits), dtype=int)
        else:
            delays = np.array(delays / circuits[0].step_size, dtype=int)

        # set maximum transfer delay for each population
        max_population_delay = np.max(delays, axis=1)

        # initialize stuff
        n_populations = np.zeros(n_circuits + 1, dtype=int)
        connectivity_coll = list()
        delays_coll = list()
        populations = list()
        n_synapses = np.zeros(n_circuits, dtype=int)

        # loop over circuits
        for i in range(n_circuits):

            # collect population count
            n_populations[i + 1] = n_populations[i] + circuits[i].N

            # collect connectivity matrix
            connectivity_tmp = circuits[i].C
            n_synapses_diff = connectivity.shape[2] - connectivity_tmp.shape[2]
            if n_synapses_diff > 0:
                connectivity_tmp = np.append(connectivity_tmp, np.zeros((circuits[i].N, circuits[i].N)), axis=2)
            connectivity_coll.append(connectivity_tmp)

            # collect delay matrix
            delays_coll.append(circuits[i].D)

            # collect number of synapses
            n_synapses[i] = circuits[i].n_synapses

            # collect populations
            for pop in circuits[i].populations:
                # update population label
                pop.label = circuit_labels[i] + '_' + pop.label

                # update population delay
                pop.synaptic_input = np.zeros((int((len(pop.synapses[0].synaptic_kernel) + max_population_delay[i])),
                                               len(pop.synapses)))

                # add population
                populations.append(pop)

        # check whether synapse dimensions match
        if any(n_synapses) > connectivity.shape[2]:
            raise AttributeError('Cannot connect circuits that have more synapses than outer connectivity matrix!')

        # build new connectivity and delay matrix
        #########################################

        # set input populations
        if not input_populations:
            input_populations = np.zeros((n_circuits, n_circuits, 1), dtype=int).tolist()

        # set output populations
        if not output_populations:
            output_populations = np.zeros((n_circuits, n_circuits), dtype=int)

        # initialize stuff
        connectivity_new = np.zeros((n_populations[-1], n_populations[-1], connectivity.shape[2]))
        delays_new = np.zeros((n_populations[-1], n_populations[-1]), dtype=int)

        # loop over all circuits
        for i in range(n_circuits):

            # set intra-circuit connectivity of circuit i
            connectivity_new[n_populations[i]:n_populations[i + 1], n_populations[i]:n_populations[i + 1], :] = \
                connectivity_coll[i]

            # set intra-circuit delays of circuit i
            delays_new[n_populations[i]:n_populations[i + 1], n_populations[i]:n_populations[i + 1]] = delays_coll[i]

            # loop again over all circuits
            for j in range(n_circuits):

                # for other circuits
                if i != j:

                    # loop over each input population in circuit i,j
                    for k in range(len(input_populations[i][j])):
                        # set inter-circuit connectivities
                        connectivity_new[n_populations[j] + input_populations[i][j][k],
                        n_populations[i] + output_populations[i, j], :] = connectivity[i, j, :]

                        # set inter-circuit delays
                        delays_new[n_populations[j] + input_populations[i][j][k],
                                   n_populations[i] + output_populations[i, j]] = delays[i, j]

        # call super init
        #################

        super().__init__(populations=populations,
                         connectivity=connectivity_new,
                         delays=delays_new,
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
