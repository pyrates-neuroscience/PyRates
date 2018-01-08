"""Module that includes basic population class.

A population is supposed to be the basic computational unit in the neural mass model. It contains various synapses
plus an axon hillok.

"""

import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from matplotlib.axes import Axes

from core.axon import Axon, SigmoidAxon
from core.synapse import Synapse, DoubleExponentialSynapse, ExponentialSynapse
from core.utility import set_instance, check_nones

from typing import List, Optional, Union, Dict, Callable, TypeVar
FloatLike = Union[float, np.float64]
AxonLike = TypeVar('AxonLike', bound=Axon, covariant=True)
SynapseLike = TypeVar('SynapseLike', bound=Synapse, covariant=True)

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"

# TODO: Rework set-up of state vector to have fixed positions for certain state variables (i.e. use fixed size vector)


class Population(object):
    """Base neural mass or population class.

        A population is defined via a number of synapses and an axon.

        Parameters
        ----------
        synapses
            Can be set to use default synapse types. These include:
            :class:`core.synapse.templates.AMPACurrentSynapse`,
            :class:`core.synapse.templates.GABAACurrentSynapse`,
            :class:`core.synapse.templates.AMPAConductanceSynapse`,
            :class:`core.synapse.templates.GABAAConductanceSynapse`.
        axon
            Can be set to use default axon types. These include:
            :class:`core.axon.templates.JansenRitAxon`.
        init_state
            Vector defining initial state of the population. Vector entries represent the following state variables:
            1) membrane potential (default = 0.0) [unit = V].
        step_size
            Time step-size of a single state update (default = 0.0001) [unit = s].
        max_synaptic_delay
            Maximum time delay after arrival of synaptic input at synapse for which this input can still affect the
            synapse (default = None) [unit = s].
        synaptic_modulation_direction
            2-dim array with first dimension being the additive synapses and the second dimension being the modulatory
            synapses. Powers used on the synaptic modulation values to define up- or down-modulation.
            Should be either 1.0 (up) or -1.0 (down) (default = None) [unit = 1].
        resting_potential
            Membrane potential at which no synaptic currents flow if no input arrives at population
            (default = -0.075) [unit = V].
        tau_leak
            time-scale with which the membrane potential of the population goes back to resting potential
            (default = 0.001) [unit = s].
        membrane_capacitance
            Average capacitance of the populations cell membrane (default = 1e-12) [unit = q/V].
        max_population_delay
            Maximum number of time-steps that external input is allowed to take to affect population
            (default = 0) [unit = s]
        synapse_params
            List of dictionaries containing parameters for custom synapse type. For parameter explanation see
            documentation of respective synapse class (:class:`DoubleExponentialSynapse`) (default = None).
        axon_params
            Parameters for custom axon type. For parameter explanation see documentation of respective axon type
            (:class:`SigmoidAxon`) (default = False).
        store_state_variables
            If false, old state variables will be erased after each state-update (default = False).
        label
            Can be used to label the population (default = 'Custom').

        Attributes
        ----------
        synapses : :obj:`list` of :class:`Synapse` objects
            Synapse instances. See documentation of :class:`Synapse`.
        axon : :class:`Axon` object
            Axon instance. See documentation of :class:`Axon`.
        state_variables : :obj:`list` of :obj:`np.ndarray`
            Collection of state variable vectors over state updates. Vector entries represent the following state
            variables:
            1) membrane potential [unit = V]
        current_firing_rate : float
            Current average firing rate of population [unit = 1/s].
        synaptic_input : np.ndarray
            Vector containing synaptic input over time for each synapse [unit = 1/s].
        current_input_idx : np.ndarary
            Index referring to position in `synaptic_input` that corresponds to time `t` of the population for each
            synapse [unit = 1].
        additive_synapse_idx : np.ndarray
            Indices of non-modulatory synapses.
        modulatory_synapse_idx : np.ndarray
            Indices of modulatory synapses.
        synaptic_currents : np.ndarray
            Vector with synaptic currents produced by the non-modulatory synapses at time-point `t`.
        synaptic_modulation : np.ndarray
            Vector with synaptic modulatory effects (applied multiplicatively to non-modulatory synapses) produced by
            modulatory synapses at time-point `t`.
        synaptic_modulation_direction : np.ndarray
            2D array containing the modulation direction (up/down) on each non-modulatory synapse (1.dim) of each
            modulatory synapse (2.dim). 1 = up-modulation, -1 = down-modulation.
        extrinsic_current : float
            Extrinsic current arriving at time-point `t`, affecting the membrane potential of the population.
        extrinsic_synaptic_modulation : float
            Extrinsic modulatory influences on the population at time-point `t`.
        n_synapses : int
            Number of different synapse types in network.
        kernel_lengths : np.ndarray
            Lengths of the synaptic kernels.
        max_population_delay : float
            Maximum delay with which other populations project to this population [unit = s] (default = 0.).
        max_synaptic_delay : np.ndarray
            See documentation of parameter `max_synaptic_delay`.
        store_state_variables
            See documentation of parameter `store_state_variables`.
        resting_potential
            See documentation of parameter `resting_potential`.
        tau_leak
            See documentation of parameter `tau_leak`.
        step_size
            See documentation of parameter `step_size`.
        membrane_capacitance
            See documentation of parameter `membrane_capacitance`.
        label
            See documentation of parameter 'label'.

        """

    def __init__(self,
                 synapses: Optional[List[str]] = None,
                 axon: Optional[str] = None,
                 init_state: FloatLike = 0.,
                 step_size: float = 0.0001,
                 max_synaptic_delay: Optional[Union[float, np.ndarray]] = None,
                 synaptic_modulation_direction: Optional[np.ndarray] = None,
                 tau_leak: float = 0.016,
                 resting_potential: float = -0.075,
                 membrane_capacitance: float = 1e-12,
                 max_population_delay: FloatLike = 0.,
                 synapse_params: Optional[List[dict]] = None,
                 axon_params: Optional[Dict[str, float]] = None,
                 synapse_class: Union[str, List[str]] = 'DoubleExponentialSynapse',
                 axon_class: str = 'SigmoidAxon',
                 store_state_variables: bool = False,
                 label: str = 'Custom'
                 ) -> None:
        """Instantiation of base population.
        """

        ##########################
        # check input parameters #
        ##########################

        # check synapse/axon attributes
        if not synapses and not synapse_params:
            raise AttributeError('Either synapses or synapse_params have to be passed!')
        if not axon and not axon_params:
            raise AttributeError('Either axon or axon_params have to be passed!')

        # check attribute values
        if step_size < 0 or tau_leak < 0 or max_population_delay < 0:
            raise ValueError('Time constants (tau, step_size, max_delay) cannot be negative. '
                             'See parameter docstring for further information.')
        if membrane_capacitance < 0:
            raise ValueError('Membrane capacitance cannot be negative. See docstring for further information.')

        #############################
        # set population parameters #
        #############################

        self.synapses: List[Synapse] = []  # instance of an Synapse class
        # self.axon: Optional[Axon] = None  # is set in _set_axon
        self.state_variables: List[List[FloatLike]] = []
        self.store_state_variables = store_state_variables
        self.tau_leak = tau_leak
        self.resting_potential = resting_potential
        self.step_size = step_size
        self.membrane_capacitance = membrane_capacitance
        self.max_population_delay = max_population_delay
        self.synaptic_modulation_direction = synaptic_modulation_direction
        self.label = label

        # set initial states
        if type(init_state) is FloatLike:
            self.state_variables.append([init_state])
        else:
            raise TypeError("If this error is raised, then the signature of 'init_state' in 'Synapse' is wrong")
            # self.state_variables.append(init_state)

        ################
        # set synapses #
        ################

        # initialize synapse parameters
        self.n_synapses = len(synapses) if synapses else len(synapse_params)  # type: ignore
        # the following is redundant, because it is also done in self.set_synapse
        # if type(synapse_class) is str:
        #     synapse_types = [synapse_class for _ in range(self.n_synapses)]
        # else:  # if synapse_class is of type List[str]
        #     synapse_types = synapse_class  # type: ignore
        if max_synaptic_delay is None:
            self.max_synaptic_delay = check_nones(max_synaptic_delay, self.n_synapses)
        elif type(max_synaptic_delay) is np.ndarray:
            self.max_synaptic_delay = max_synaptic_delay  # type: ignore
        else:
            self.max_synaptic_delay = np.zeros(self.n_synapses) + max_synaptic_delay

        # instantiate synapses
        self._set_synapses(synapse_subtypes=synapses,  # type: ignore
                           synapse_params=synapse_params,
                           synapse_types=synapse_class)

        # get relevant information from each synapse instance
        synapse_type = np.ones(self.n_synapses, dtype=bool)
        self.kernel_lengths = np.ones(self.n_synapses, dtype=int)

        for i, syn in enumerate(self.synapses):

            # distinguish between additive and multiplicatory synapses
            if syn.modulatory:
                synapse_type[i] = False

            # get kernel length
            self.kernel_lengths[i] = len(syn.synaptic_kernel)

        # set synaptic input array
        self.synaptic_input = np.zeros((int(self.max_population_delay / self.step_size) +
                                        np.max(self.kernel_lengths), self.n_synapses))
        self.dummy_input = np.zeros((1, self.n_synapses))

        # set input index for each synapse
        self.current_input_idx = np.zeros(self.n_synapses, dtype=int)

        # set synaptic current and modulation collector vectors
        self.additive_synapse_idx = np.where(synapse_type == 1)[0]
        self.modulatory_synapse_idx = np.where(synapse_type == 0)[0]
        self.synaptic_currents = np.zeros(len(self.additive_synapse_idx))
        self.synaptic_modulation = np.zeros((1, len(self.modulatory_synapse_idx)))

        # set modulation direction for modulatory synapses
        if self.synaptic_modulation_direction is None and self.synaptic_modulation:
            self.synaptic_modulation_direction = np.ones(len(self.synaptic_currents), len(self.synaptic_modulation))

        ############
        # set axon #
        ############

        self._set_axon(axon, axon_params=axon_params, axon_type=axon_class)
        self.current_firing_rate = self.get_firing_rate()

        ###################################
        # initialize extrinsic influences #
        ###################################

        self.extrinsic_current = 0.
        self.extrinsic_synaptic_modulation = 1.

    def _set_synapses(self,
                      synapse_subtypes: Optional[List[str]] = None,
                      synapse_types: Union[str, List[str]] = 'DoubleExponentialSynapse',
                      synapse_params: Optional[List[dict]] = None
                      ) -> None:
        """Instantiates synapses.

        Parameters
        ----------
        synapse_subtypes
            Names of pre-parametrized synapse sub-classes.
        synapse_types
            Names of synapse classes to instantiate.
        synapse_params
            Dictionaries with synapse parameter name-value pairs.

        """

        # check synapse parameter formats
        if type(synapse_types) is str:
            synapse_types = [synapse_types for _ in range(self.n_synapses)]  # type: ignore

        synapse_subtypes = check_nones(synapse_subtypes, self.n_synapses)
        synapse_params = check_nones(synapse_params, self.n_synapses)

        # set all given synapses
        for i in range(self.n_synapses):

            if synapse_types[i] == 'DoubleExponentialSynapse':
                self.synapses.append(set_instance(DoubleExponentialSynapse,  # type: ignore
                                                  synapse_subtypes[i],  # type: ignore
                                                  synapse_params[i],  # type: ignore
                                                  bin_size=self.step_size,
                                                  max_delay=self.max_synaptic_delay[i]))
            elif synapse_types[i] == 'ExponentialSynapse':
                self.synapses.append(set_instance(ExponentialSynapse,  # type: ignore
                                                  synapse_subtypes[i],   # type: ignore
                                                  synapse_params[i],  # type: ignore
                                                  bin_size=self.step_size, max_delay=self.max_synaptic_delay[i]))
            elif synapse_types[i] == 'Synapse':
                self.synapses.append(set_instance(Synapse,  # type: ignore
                                                  synapse_subtypes[i],  # type: ignore
                                                  synapse_params[i],  # type: ignore
                                                  bin_size=self.step_size,
                                                  max_delay=self.max_synaptic_delay[i]))
            else:
                raise AttributeError('Invalid synapse type!')

    def _set_axon(self,
                  axon_subtype: Optional[str] = None,
                  axon_type: str = 'SigmoidAxon',
                  axon_params: Optional[dict] = None
                  ) -> None:
        """Instantiates axon.

        Parameters
        ----------
        axon_subtype
            Name of pre-parametrized axon sub-class.
        axon_type
            Name of axon class to instantiate.
        axon_params
            Dictionary with axon parameter name-value pairs.

        """

        if axon_type == 'SigmoidAxon':
            self.axon: Axon = set_instance(SigmoidAxon, axon_subtype, axon_params)  # type: ignore
        elif axon_type == 'Axon':
            self.axon: Axon = set_instance(Axon, axon_subtype, axon_params)  # type: ignore
        else:
            raise AttributeError('Invalid axon type!')

    def get_firing_rate(self) -> FloatLike:
        """Calculate the current average firing rate of the population.

        Returns
        -------
        float
            Average firing rate of population [unit = 1/s].

        """

        return self.axon.compute_firing_rate(self.state_variables[-1][0])

    def get_delta_membrane_potential(self,
                                     membrane_potential: FloatLike
                                     ) -> FloatLike:
        """Calculates change in membrane potential as function of synaptic current, leak current and
        extrinsic current.

        Parameters
        ----------
        membrane_potential
            Current membrane potential of population [unit = V].

        Returns
        -------
        float
            Delta membrane potential [unit = V].

        """

        net_current = self.get_synaptic_currents(membrane_potential) \
                      + self.get_leak_current(membrane_potential) \
                      + self.extrinsic_current

        return net_current / self.membrane_capacitance

    def get_synaptic_currents(self,
                              membrane_potential: FloatLike
                              ) -> Union[FloatLike, np.ndarray]:
        """Calculates the net synaptic current over all synapses.

        Parameters
        ----------
        membrane_potential
            Current membrane potential of population [unit = V].

        Returns
        -------
        float
            Net synaptic current [unit = A].

        """

        #############################################
        # compute synaptic currents and modulations #
        #############################################

        # calculate synaptic currents for each additive synapse
        for i, idx in enumerate(self.additive_synapse_idx):

            self.synaptic_currents[i] = self.synapses[idx].get_synaptic_current(
                self.synaptic_input[0:self.current_input_idx[idx] + 1, idx], membrane_potential)

        # compute synaptic modulation value for each modulatory synapse and apply it to currents
        if self.modulatory_synapse_idx:

            self.get_synaptic_modulation(membrane_potential)

        return np.sum(self.synaptic_currents)

    def get_leak_current(self,
                         membrane_potential: FloatLike
                         ) -> FloatLike:
        """Calculates the leakage current at a given point in time (instantaneous).

        Parameters
        ----------
        membrane_potential
            Current membrane potential of population [unit = V].

        Returns
        -------
        float
            Leak current [unit = A].

        """

        return (self.resting_potential - membrane_potential) * self.membrane_capacitance / self.tau_leak

    def get_synaptic_modulation(self,
                                membrane_potential: FloatLike
                                ) -> None:
        """Calculates modulation weight for each additive synapse.

         Modulation values are applied to the synaptic current stored on the population afterwards.

        Parameters
        ----------
        membrane_potential
            Current membrane potential [unit = V].

        """

        # calculate synaptic modulation value of each modulatory synapse
        for i, idx in enumerate(self.modulatory_synapse_idx):

            self.synaptic_modulation[i] = self.synapses[idx].get_synaptic_current(
                self.synaptic_input[0:self.current_input_idx[idx] + 1, idx], membrane_potential)

        # apply modulation direction and get modulation value for each synapse
        synaptic_modulation_new = np.tile(self.synaptic_modulation, (len(self.synaptic_currents), 1))
        synaptic_modulation_new = synaptic_modulation_new ** self.synaptic_modulation_direction
        synaptic_modulation_new = np.prod(synaptic_modulation_new, axis=1)

        # combine extrinsic and intrinsic synaptic modulation
        if self.extrinsic_synaptic_modulation:
            synaptic_modulation_new *= self.extrinsic_synaptic_modulation

        self.synaptic_currents *= synaptic_modulation_new

    def state_update(self,
                     synaptic_input: np.ndarray,
                     extrinsic_current: FloatLike = 0.,
                     extrinsic_synaptic_modulation: float = 1.0
                     ) -> None:
        """Updates state of population by making a single step forward in time.

        Parameters
        ----------
        synaptic_input
            Vector with additional synaptic input per synapse arriving at the time of the state update [unit = 1/s].
        extrinsic_current
            Extrinsic current arriving at time-point `t`, affecting the membrane potential of the population.
            (default = 0.) [unit = A].
        extrinsic_synaptic_modulation
            modulatory input to each synapse. Can be scalar (applied to all synapses then) or vector with len = number
            of synapses (default = 1.0) [unit = 1].

        """

        ##########################################
        # add inputs to internal state variables #
        ##########################################

        # synaptic inputs
        self.synaptic_input[self.current_input_idx, 0:len(synaptic_input)] += synaptic_input  # type: ignore

        # extrinsic inputs
        self.extrinsic_current = extrinsic_current
        self.extrinsic_synaptic_modulation = extrinsic_synaptic_modulation

        ######################################
        # compute average membrane potential #
        ######################################

        membrane_potential = self.state_variables[-1][0]
        membrane_potential = self.take_step(f=self.get_delta_membrane_potential,
                                            y_old=membrane_potential)

        state_vars = [membrane_potential]

        ##########################
        # update state variables #
        ##########################

        # firing rate of population
        self.current_firing_rate = self.get_firing_rate()

        # state history of population
        self.state_variables.append(state_vars)
        if not self.store_state_variables:
            self.state_variables.pop(0)

        ################################
        # rotate synaptic input vector #
        ################################

        # check which synaptic input vectors need to be rotated
        idx = self.current_input_idx < self.kernel_lengths-1
        not_idx = np.invert(idx)

        # either rotate synaptic input vector or increase input index
        self.current_input_idx[idx] += 1
        self.synaptic_input[0:-1, not_idx] = self.synaptic_input[1:, not_idx]
        self.synaptic_input[-1, not_idx] = self.dummy_input[0, not_idx]

    def take_step(self,
                  f: Callable,
                  y_old: Union[FloatLike, np.ndarray],
                  **kwargs
                  ) -> FloatLike:
        """Takes a step of an ODE with right-hand-side f using Euler formalism.

        Parameters
        ----------
        f
            Function that represents right-hand-side of ODE and takes `t` plus `y_old` as an argument.
        y_old
            Old value of y that needs to be updated according to dy(t)/dt = f(t, y)
        **kwargs
            Name-value pairs to be passed to f.

        Returns
        -------
        float
            Updated value of left-hand-side (y).

        """

        return y_old + self.step_size * f(y_old, **kwargs)

    def copy_synapse(self, synapse_idx: int) -> None:
        """Copies an existing synapse

        Parameters
        ----------
        synapse_idx
            Index of synapse to copy (default = None).
        """

        synapse: Synapse = deepcopy(self.synapses[synapse_idx])

        self.add_synapse(synapse)

    def add_synapse(self,
                    synapse: Synapse,
                    synapse_idx: Optional[int] = None,
                    ) -> None:
        """Adds copy of specified synapse to population.

        Parameters
        ----------
        synapse
            Synapse object to add (default = None)
        synapse_idx
            Index of synapse to copy (default = None).

        """

        ###############
        # add synapse #
        ###############

        self.synapses.append(synapse)

        ###############################
        # update synapse dependencies #
        ###############################

        self.n_synapses += 1
        self.kernel_lengths = np.append(self.kernel_lengths, len(synapse.synaptic_kernel))  # type: ignore

        # check modulation dependencies
        if synapse.modulatory:  # type: ignore
            raise AttributeError('Adding modulatory synapses is currently not implemented. Sorry.')
        else:
            if self.synaptic_modulation_direction is not None:
                raise AttributeError('Add synapse not supported for networks with modulatory synapses yet. Sorry.')
            self.synaptic_currents = np.zeros(len(self.synaptic_currents) + 1)
            self.additive_synapse_idx = np.append(self.additive_synapse_idx, self.n_synapses - 1)

        # check synaptic input dependencies
        self.synaptic_input = np.append(self.synaptic_input, np.zeros((self.synaptic_input.shape[0], 1)), axis=1)
        self.current_input_idx = np.append(self.current_input_idx,
                                           self.current_input_idx[synapse_idx])
        self.dummy_input = np.zeros((1, self.n_synapses))

    def clear(self):
        """Clears states stored on population.
        """

        init_state = self.state_variables[0]
        self.state_variables.clear()
        self.state_variables.append(init_state)

    def plot_synaptic_kernels(self, synapse_idx: Optional[List[int]]=None, create_plot: Optional[bool]=True,
                              axes: Axes=None) -> object:
        """Creates plot of all specified synapses over time.

        Parameters
        ----------
        synapse_idx
            Index of synapses for which to plot kernel.
        create_plot
            If true, plot will be shown.
        axes
            Can be used to pass figure handle of figure to plot into.

        Returns
        -------
        figure handle
            Handle of newly created or updated figure.

        """

        ####################
        # check parameters #
        ####################

        # assert synapse_idx is None or type(synapse_idx) is list

        #############################
        # check positional argument #
        #############################

        if synapse_idx is None:
            synapse_idx = list(range(len(self.synapses)))

        #########################
        # plot synaptic kernels #
        #########################

        if axes is None:
            fig, axes = plt.subplots(num='Synaptic Kernel Functions')
        else:
            fig = axes.get_figure()

        synapse_types = list()
        for i in synapse_idx:
            axes = self.synapses[i].plot_synaptic_kernel(create_plot=False, axes=axes)
            synapse_types.append(self.synapses[i].synapse_type)

        plt.legend(synapse_types)

        if create_plot:
            fig.show()

        return axes


class PlasticPopulation(Population):
    """Neural mass or population class with optional plasticity mechanisms on synapses and axon.

    A population is defined via a number of synapses and an axon.

    Parameters
    ----------
    synapses
        See docstring of :class:`Population`.
    axon
        See docstring of :class:`Population`.
    init_state
        Vector defining initial state of the population. Vector entries represent the following state variables:
        1)      membrane potential (default = 0.0) [unit = V].
        2)      membrane potential threshold (default = -0.07) [unit = V].
        3,...)  efficacy scalings of plastic synapses (default = 1.0) [unit = 1].
    axon_plasticity_function
        Function defining the axonal plasticity mechanisms to be used on population.
    axon_plasticity_target_param
        Target parameter of the axon to which the axonal plasticity function is applied.
    axon_plasticity_function_params
        Name-value pairs defining the parameters for the axonal plasticity function.
    synapse_plasticity_function
        Function defining the synaptic plasticity mechanisms to be used on population.
    synapse_plasticity_function_params
        Name-value pairs defining the parameters for the synaptic plasticity function.
    step_size
        See docstring of :class:`Population`.
    max_synaptic_delay
        See docstring of :class:`Population`.
    synaptic_modulation_direction
        See docstring of :class:`Population`.
    resting_potential
        See docstring of :class:`Population`.
    tau_leak
        See docstring of :class:`Population`.
    membrane_capacitance
        See docstring of :class:`Population`.
    max_population_delay
        See docstring of :class:`Population`.
    synapse_params
        See docstring of :class:`Population`.
    axon_params
        See docstring of :class:`Population`.
    store_state_variables
        See docstring of :class:`Population`.
    label
        See docstring of :class:`Population`.

    See Also
    --------
    :class:`Population`: Detailed description of parameters, attributes and methods.

    """

    def __init__(self,
                 synapses: Optional[List[str]] = None,
                 axon: Optional[str] = None,
                 init_state: FloatLike = 0.,
                 step_size: float = 0.0001,
                 max_synaptic_delay: Optional[Union[float, np.ndarray]] = None,
                 synaptic_modulation_direction: Optional[np.ndarray] = None,
                 tau_leak: float = 0.016,
                 resting_potential: float = -0.075,
                 membrane_capacitance: float = 1e-12,
                 max_population_delay: FloatLike = 0.,
                 synapse_params: Optional[List[dict]] = None,
                 axon_params: Optional[Dict[str, float]] = None,
                 synapse_class: Union[str, List[str]] = 'DoubleExponentialSynapse',
                 axon_class: str = 'SigmoidAxon',
                 store_state_variables: bool = False,
                 label: str = 'Custom',
                 axon_plasticity_function: Optional[Callable[[float], float]] = None,
                 axon_plasticity_target_param: Optional[str] = None,
                 axon_plasticity_function_params: Optional[Dict[str, float]] = None,
                 synapse_plasticity_function: Callable[[float], float] = None,
                 synapse_plasticity_function_params: List[dict] = None,
                 ) -> None:
        """Instantiation of plastic population.
        """

        ###################
        # call super init #
        ###################

        super().__init__(synapses=synapses,
                         axon=axon,
                         init_state=init_state,
                         step_size=step_size,
                         max_synaptic_delay=max_synaptic_delay,
                         synaptic_modulation_direction=synaptic_modulation_direction,
                         tau_leak=tau_leak,
                         resting_potential=resting_potential,
                         membrane_capacitance=membrane_capacitance,
                         max_population_delay=max_population_delay,
                         synapse_params=synapse_params,
                         axon_params=axon_params,
                         synapse_class=synapse_class,
                         axon_class=axon_class,
                         store_state_variables=store_state_variables,
                         label=label)

        #############################
        # set plasticity attributes #
        #############################

        # for axon
        if axon_plasticity_function:
            if not axon_plasticity_target_param or not axon_plasticity_function_params:
                raise ValueError("If an axon_plasticity_function was given, then also axon_plasticity_target_param and "
                                 "axon_plasticity_function_params need to be specified.")
            self.axon_plasticity_target_param = axon_plasticity_target_param
            self.axon_plasticity_function_params = axon_plasticity_function_params
        self.axon_plasticity_function = axon_plasticity_function

        # for synapses
        self.synapse_plasticity_function = synapse_plasticity_function
        if synapse_plasticity_function_params:
            if len(synapse_plasticity_function_params) == 1:
                self.synapse_plasticity_function_params = [synapse_plasticity_function_params[0]
                                                           for _ in range(self.n_synapses)]
            else:
                self.synapse_plasticity_function_params = synapse_plasticity_function_params
        else:
            raise ValueError("synapse_plasticity_function_params needs to be given as a list of dictionaries.")

    def state_update(self,
                     synaptic_input: np.ndarray,
                     extrinsic_current: FloatLike = 0.,
                     extrinsic_synaptic_modulation: float = 1.0,
                     ) -> None:
        """Updates state of population by making a single step forward in time.

        Parameters
        ----------
        synaptic_input
            Vector with additional synaptic input per synapse arriving at the time of the state update [unit = 1/s].
        extrinsic_current
            Extrinsic current arriving at time-point `t`, affecting the membrane potential of the population.
            (default = 0.) [unit = A].
        extrinsic_synaptic_modulation
            modulatory input to each synapse. Can be scalar (applied to all synapses then) or vector with len = number
            of synapses (default = 1.0) [unit = 1].

        """

        ###########################
        # call super state update #
        ###########################

        super().state_update(synaptic_input=synaptic_input,
                             extrinsic_current=extrinsic_current,
                             extrinsic_synaptic_modulation=extrinsic_synaptic_modulation)

        ###################################
        # update axonal transfer function #
        ###################################

        if self.axon_plasticity_function:

            self.axon.transfer_function_args[self.axon_plasticity_target_param] = \
                Population.take_step(self,  # type: ignore
                                     f=self.axon_plasticity_function,
                                     y_old=self.axon.transfer_function_args[self.axon_plasticity_target_param],  # type: ignore
                                     firing_rate_target=self.current_firing_rate,
                                     **self.axon_plasticity_function_params)

        ###########################
        # update synaptic scaling #
        ###########################

        if self.synapse_plasticity_function:
            for i in range(self.n_synapses):
                if self.synapse_plasticity_function_params[i]:
                    self.synapses[i].depression = Population.take_step(self,
                                                                       f=self.synapse_plasticity_function,
                                                                       y_old=self.synapses[i].depression,
                                                                       firing_rate=self.synaptic_input[
                                                                            self.current_input_idx[i] - 1, i],
                                                                       **self.synapse_plasticity_function_params[i])

    def add_plastic_synapse(self,
                            synapse_idx: int,
                            synapse: Optional[Synapse] = None,
                            max_firing_rate: Optional[float] = None) -> None:
        """Adds copy of specified synapse to population.

        Parameters
        ----------
        synapse_idx
            Index of synapse to copy (default = None).
        synapse
            Synapse object to add (default = None)
        max_firing_rate
            Maximum firing rate of connecting population. Used for synaptic plasticity mechanism (default = None).

        """

        #####################
        # call super method #
        #####################

        if synapse:
            self.add_synapse(synapse, synapse_idx)
        else:
            self.copy_synapse(synapse_idx)

        ##################################
        # check plasticity related stuff #
        ##################################

        if max_firing_rate is None:
            max_firing_rate = self.axon.transfer_function_args['max_firing_rate']

        self.synapse_plasticity_function_params.append(self.synapse_plasticity_function_params[synapse_idx])
        self.synapse_plasticity_function_params[-1]['max_firing_rate'] = max_firing_rate


class SecondOrderPopulation(Population):
    """Neural mass or population class as defined in [1]_.

    A population is defined via a number of synapses and an axon.

    Parameters
    ----------
    synapses
        See docstring of :class:`Population`.
    axon
        See docstring of :class:`Population`.
    init_state
        See docstring of :class`PlasticPopulation`.
    step_size
        See docstring of :class:`Population`.
    max_synaptic_delay
        See docstring of :class:`Population`.
    resting_potential
        See docstring of :class:`Population`.
    max_population_delay
        See docstring of :class:`Population`.
    synapse_params
        See docstring of :class:`Population`.
    axon_params
        See docstring of :class:`Population`.
    store_state_variables
        See docstring of :class:`Population`.
    label
        See docstring of :class:`Population`.

    See Also
    --------
    :class:`Population`: Detailed description of parameters, attributes and methods.

    References
    ----------
    .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
       of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

    """

    def __init__(self,
                 synapses: Optional[List[str]] = None,
                 axon: Optional[str] = None,
                 init_state: FloatLike = 0.,
                 step_size: float = 0.0001,
                 max_synaptic_delay: Optional[Union[float, np.ndarray]] = None,
                 resting_potential: float = 0.,
                 max_population_delay: FloatLike = 0.,
                 synapse_params: Optional[List[dict]] = None,
                 axon_params: Optional[Dict[str, float]] = None,
                 synapse_class: Union[str, List[str]] = 'ExponentialSynapse',
                 axon_class: str = 'SigmoidAxon',
                 store_state_variables: bool = False,
                 label: str = 'Custom'
                 ) -> None:
        """Instantiation of second order population.
        """

        Population.__init__(self,
                            synapses=synapses,
                            axon=axon,
                            init_state=init_state,
                            step_size=step_size,
                            max_synaptic_delay=max_synaptic_delay,
                            resting_potential=resting_potential,
                            max_population_delay=max_population_delay,
                            synapse_params=synapse_params,
                            axon_params=axon_params,
                            synapse_class=synapse_class,
                            axon_class=axon_class,
                            store_state_variables=store_state_variables,
                            label=label)

    def take_step(self,
                  f: Callable,
                  y_old: Union[FloatLike, np.ndarray],
                  **kwargs
                  ) -> FloatLike:
        """Takes a step of an ODE with right-hand-side f using Euler formalism.

        Parameters
        ----------
        f
            Function that represents right-hand-side of ODE and takes `t` plus `y_old` as an argument.
        y_old
            Old value of y that needs to be updated according to dy(t)/dt = f(t, y)
        **kwargs
            Name-value pairs that are passed as parameters to f.

        Returns
        -------
        float
            Updated value of left-hand-side (y).

        """

        return f(y_old, **kwargs)

    def get_delta_membrane_potential(self,
                                     membrane_potential: FloatLike
                                     ) -> FloatLike:
        """Calculates change in membrane potential as function of synaptic current, leak current and
        extrinsic current.

        Parameters
        ----------
        membrane_potential
            Current membrane potential of population [unit = V].

        Returns
        -------
        float
            Delta membrane potential [unit = V].

        """

        return self.get_synaptic_currents(membrane_potential) + self.extrinsic_current


class SecondOrderPlasticPopulation(PlasticPopulation):
    """Neural mass or population class as defined in [1]_.

    A population is defined via a number of synapses and an axon.

    Parameters
    ----------
    synapses
        See docstring of :class:`Population`.
    axon
        See docstring of :class:`Population`.
    init_state
        See docstring of :class`PlasticPopulation`.
    step_size
        See docstring of :class:`Population`.
    max_synaptic_delay
        See docstring of :class:`Population`.
    resting_potential
        See docstring of :class:`Population`.
    max_population_delay
        See docstring of :class:`Population`.
    synapse_params
        See docstring of :class:`Population`.
    axon_params
        See docstring of :class:`Population`.
    store_state_variables
        See docstring of :class:`Population`.
    label
        See docstring of :class:`Population`.
    axon_plasticity_function
        See docstring of :class`PlasticPopulation`.
    axon_plasticity_target_param
        See docstring of :class`PlasticPopulation`.
    axon_plasticity_function_params
        See docstring of :class`PlasticPopulation`.
    synapse_plasticity_function
        See docstring of :class`PlasticPopulation`.
    synapse_plasticity_function_params
        See docstring of :class`PlasticPopulation`.

    See Also
    --------
    :class:`Population`: Detailed description of parameters, attributes and methods.
    :class:`PlasticPopulation`: Detailed description of plasticity parameters.

    References
    ----------
    .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
       of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

    """

    def __init__(self,
                 synapses: Optional[List[str]] = None,
                 axon: Optional[str] = None,
                 init_state: FloatLike = 0.,
                 step_size: float = 0.0001,
                 max_synaptic_delay: Optional[Union[float, np.ndarray]] = None,
                 resting_potential: float = 0.,
                 max_population_delay: FloatLike = 0.,
                 synapse_params: Optional[List[dict]] = None,
                 axon_params: Optional[Dict[str, float]] = None,
                 synapse_class: Union[str, List[str]] = 'ExponentialSynapse',
                 axon_class: str = 'SigmoidAxon',
                 store_state_variables: bool = False,
                 label: str = 'Custom',
                 axon_plasticity_function: Callable[[float], float] = None,
                 axon_plasticity_target_param: str = None,
                 axon_plasticity_function_params: dict = None,
                 synapse_plasticity_function: Callable[[float], float] = None,
                 synapse_plasticity_function_params: Optional[List[dict]] = None,
                 ) -> None:
        """Instantiation of second order population.
        """

        PlasticPopulation.__init__(self,
                                   synapses=synapses,
                                   axon=axon,
                                   init_state=init_state,
                                   step_size=step_size,
                                   max_synaptic_delay=max_synaptic_delay,
                                   resting_potential=resting_potential,
                                   max_population_delay=max_population_delay,
                                   synapse_params=synapse_params,
                                   axon_params=axon_params,
                                   synapse_class=synapse_class,
                                   axon_class=axon_class,
                                   store_state_variables=store_state_variables,
                                   label=label,
                                   axon_plasticity_function=axon_plasticity_function,
                                   axon_plasticity_target_param=axon_plasticity_target_param,
                                   axon_plasticity_function_params=axon_plasticity_function_params,
                                   synapse_plasticity_function=synapse_plasticity_function,
                                   synapse_plasticity_function_params=synapse_plasticity_function_params)

    def take_step(self,
                  f: Callable,
                  y_old: Union[FloatLike, np.ndarray],
                  **kwargs
                  ) -> FloatLike:
        """Takes a step of an ODE with right-hand-side f using Euler formalism.

        Parameters
        ----------
        f
            Function that represents right-hand-side of ODE and takes `t` plus `y_old` as an argument.
        y_old
            Old value of y that needs to be updated according to dy(t)/dt = f(t, y)
        **kwargs
            Name-value pairs that are passed as parameters to f.

        Returns
        -------
        float
            Updated value of left-hand-side (y).

        """

        return f(y_old, **kwargs)

    def get_delta_membrane_potential(self,
                                     membrane_potential: FloatLike
                                     ) -> FloatLike:
        """Calculates change in membrane potential as function of synaptic current, leak current and
        extrinsic current.

        Parameters
        ----------
        membrane_potential
            Current membrane potential of population [unit = V].

        Returns
        -------
        float
            Delta membrane potential [unit = V].

        """

        return self.get_synaptic_currents(membrane_potential) + self.extrinsic_current