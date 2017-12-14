"""Module that includes basic population class.

A population is supposed to be the basic computational unit in the neural mass model. It contains various synapses
plus an axon hillok.

"""

import matplotlib.pyplot as plt
import numpy as np

from core.axon import Axon, SigmoidAxon
from core.synapse import Synapse, DoubleExponentialSynapse
from core.utility import set_instance, check_nones

from typing import List, Optional, Union, Dict, Callable, TypeVar
FloatLike = Union[float, np.float64]
AxonLike = TypeVar('AxonLike', bound=Axon, covariant=True)
SynapseLike = TypeVar('SynapseLike', bound=Synapse, covariant=True)

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"

# TODO: Rework set-up of state vector to have fixed positions for certain state variables (i.e. use fixed size vector)
# TODO: Create base class plus that takes the different getters as init arguments


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
        Maximum time delay after arrival of synaptic input at synapse for which this input can still affect the synapse
        (default = 0.1) [unit = s].
    synaptic_modulation_direction
        2-dim array with first dimension being the additive synapses and the second dimension being the modulatory
        synapses. Powers used on the synaptic modulation values to define up- or down-modulation.
        Should be either 1.0 (up) or -1.0 (down) (default = None) [unit = 1].
    resting_potential
        Membrane potential at which no synaptic currents flow if no input arrives at population
        (default = -0.07) [unit = V].
    tau_leak
        time-scale with which the membrane potential of the population goes back to resting potential
        (default = 0.001) [unit = s].
    membrane_capacitance
        Average capacitance of the populations cell membrane (default = 1e-12) [unit = q/V].
    max_population_delay
        Maximum number of time-steps that external input is allowed to take to affect population
        (default = 0) [unit = s]
    synapse_params
        List of dictionaries containing parameters for custom synapse type. For parameter explanation see documentation
        of respective synapse class (:class:`DoubleExponentialSynapse`) (default = None).
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
        Collection of state variable vectors over state updates. Vector entries represent the following state variables:
        1) membrane potential [unit = V]
        2) membrane_potential_threshold [unit = V]
    t : float
        Time the current population state refers to [unit = s].
    synaptic_input : np.ndarray
        Vector containing synaptic input over time [unit = 1/s].
    current_input_idx : int
        Index referring to position in `synaptic_input` that corresponds to time `t` of the population [unit = 1].
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
    axon_plasticity : bool
        If true, axonal plasticity mechanism is enabled that adjusts the `membrane_potential_threshold` of the axon.
        Also adds that variable to the `state_variable` vector of the population.
    firing_rate_target : float
        Target firing rate that is used for synaptic plasticity mechanism. Only set if `axon_plasticity` is true.
    extrinsic_current : float
        Extrinsic current arriving at time-point `t`, affecting the membrane potential of the population.
    extrinsic_synaptic_modulation : float
        Extrinsic modulatory influences on the population at time-point `t`.
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

    Methods
    -------
    state_update
        See method docstring.
    take_step
        See method docstring.
    get_delta_membrane_potential
        See method docstring.
    get_synaptic_current
        See method docstring.
    get_leak_current
        See method docstring.
    get_synaptic_modulation
        See method docstring.
    get_delta_membrane_potential_threshold
        See method docstring.
    get_firing_rate
        See method docstring.
    plot_synaptic_kernels
        See method docstring.
    set_synapse
        See method docstring.
    set_axon
        See method docstring.

    """

    def __init__(self, synapses: Optional[List[str]]=None,
                 axon: Optional[str]=None,
                 init_state: Optional[FloatLike]=0.,
                 step_size: Optional[float]=0.0001,
                 max_synaptic_delay: Optional[Union[float, np.ndarray]]=0.1,
                 synaptic_modulation_direction: Optional[np.ndarray]=None,
                 tau_leak: Optional[float]=0.016,
                 resting_potential: Optional[float]=-0.075,
                 membrane_capacitance: Optional[float]=1e-12,
                 max_population_delay: Optional[FloatLike]=0.,
                 synapse_params: Optional[List[Dict[str, Union[bool, float]]]]=None,
                 axon_params: Optional[Dict[str, float]]=None,
                 store_state_variables: Optional[bool]=False,
                 label: Optional[str]='Custom') -> None:
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
        if step_size < 0 or tau_leak < 0 or max_synaptic_delay < 0 or max_population_delay < 0:
            raise ValueError('Time constants (tau, step_size, max_delay) cannot be negative. '
                             'See parameter docstring for further information.')
        if membrane_capacitance < 0:
            raise ValueError('Membrane capacitance cannot be negative. See docstring for further information.')

        #############################
        # set population parameters #
        #############################

        self.synapses = list()
        self.axon = None
        self.state_variables = list()
        self.store_state_variables = store_state_variables
        self.tau_leak = tau_leak
        self.resting_potential = resting_potential
        self.step_size = step_size
        self.membrane_capacitance = membrane_capacitance
        self.max_population_delay = max_population_delay
        self.synaptic_modulation_direction = synaptic_modulation_direction
        self.t = 0.
        self.label = label

        # set initial states
        self.state_variables.append([init_state]) if type(init_state) is float or np.float64 \
            else self.state_variables.append(init_state)

        ####################################
        # set axonal plasticity parameters #
        ####################################

        if axon_params:

            if 'tau' in axon_params:

                # if axon timescale tau is in axon parameters, set relevant parameters for axon plasticity
                self.axon_plasticity = True
                self.tau_axon = axon_params['tau']
                if 'firing_rate_target' in axon_params:
                    self.firing_rate_target = axon_params['firing_rate_target']
                else:
                    self.firing_rate_target = 2.5

            else:

                self.axon_plasticity = False

        else:

            self.axon_plasticity = False

        ######################################
        # set synaptic plasticity parameters #
        ######################################

        self.plastic_synapses = list()
        if synapse_params:

            self.tau_depression = list()
            self.tau_recycle = list()

            # loop over synapses and check for passed synaptic plasticity parameters
            for i, syn in enumerate(synapse_params):

                if syn and 'tau_depression' in syn and 'tau_recycle' in syn:
                    self.plastic_synapses.append(True)
                    self.tau_depression.append(syn['tau_depression'])
                    self.tau_recycle.append(syn['tau_recycle'])
                else:
                    self.plastic_synapses.append(False)

            # transform lists into arrays
            self.plastic_synapses = np.array(self.plastic_synapses, dtype=bool)
            self.tau_depression = np.array(self.tau_depression)
            self.tau_recycle = np.array(self.tau_recycle)

        ###############
        # set synapse #
        ###############

        # initialize synapse parameters
        n_synapses = len(synapses) if synapses else len(synapse_params)
        synapses = check_nones(synapses, n_synapses)
        synapse_params = check_nones(synapse_params, n_synapses)
        if type(max_synaptic_delay) is np.ndarray:
            self.max_synaptic_delay = max_synaptic_delay
        else:
            self.max_synaptic_delay = np.zeros(n_synapses) + max_synaptic_delay

        # instantiate synapses
        for i in range(n_synapses):
            self.set_synapse(self.max_synaptic_delay[i],
                             synapse_subtype=synapses[i],
                             synapse_params=synapse_params[i])

        # set synapse dependent attributes
        self.set_synapse_dependencies()

        ############
        # set axon #
        ############

        self.set_axon(axon, axon_params=axon_params)
        self.current_firing_rate = self.get_firing_rate()

        ###################################
        # initialize extrinsic influences #
        ###################################

        self.extrinsic_current = 0.
        self.extrinsic_synaptic_modulation = 1.

    def state_update(self, extrinsic_current: FloatLike=0.,
                     extrinsic_synaptic_modulation: float=1.0) -> None:
        """Updates state of population by making a single step forward in time.

        Parameters
        ----------
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

        self.extrinsic_current = extrinsic_current
        self.extrinsic_synaptic_modulation = extrinsic_synaptic_modulation

        ######################################
        # compute average membrane potential #
        ######################################

        membrane_potential = self.state_variables[-1][0]
        membrane_potential = self.take_step(f=self.get_delta_membrane_potential,
                                            y_old=membrane_potential)

        state_vars = [membrane_potential]

        ###################################
        # update axonal transfer function #
        ###################################

        if self.axon_plasticity:

            self.axon.membrane_potential_threshold = self.take_step(f=self.get_delta_membrane_potential_threshold,
                                                                    y_old=self.axon.membrane_potential_threshold)

            state_vars.append(self.axon.membrane_potential_threshold)

        ###########################
        # update synaptic scaling #
        ###########################

        if any(self.plastic_synapses):
            self.synaptic_scalings[self.plastic_synapses] = self.take_step(f=self.get_delta_synaptic_scaling,
                                                                           y_old=self.synaptic_scalings[
                                                                               self.plastic_synapses])
            #state_vars.append(self.synaptic_scalings[self.plastic_synapses])

        ##########################
        # update state variables #
        ##########################

        self.current_firing_rate = self.get_firing_rate()
        self.state_variables.append(state_vars)

        if not self.store_state_variables:
            self.state_variables.pop(0)

        ################################################
        # advance in time and in synaptic input vector #
        ################################################

        # time
        self.t += self.step_size

        # synaptic input
        idx = self.current_input_idx < self.kernel_lengths-1
        not_idx = np.invert(idx)
        self.current_input_idx[idx] += 1
        self.synaptic_input[0:-1, not_idx] = self.synaptic_input[1:, not_idx]
        self.synaptic_input[-1, not_idx] = self.dummy_input[0, not_idx]

    def take_step(self, f: Callable, y_old: FloatLike) -> FloatLike:
        """Takes a step of an ODE with right-hand-side f using Euler or Adams/BDF formalism.

        Parameters
        ----------
        f
            Function that represents right-hand-side of ODE and takes `t` plus `y_old` as an argument.
        y_old
            Old value of y that needs to be updated according to dy(t)/dt = f(t, y)

        Returns
        -------
        float
            Updated value of left-hand-side (y).

        """

        return y_old + self.step_size * f(y_old)

    def get_delta_membrane_potential(self, membrane_potential: FloatLike) -> FloatLike:
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

        net_current = self.get_synaptic_currents(membrane_potential) + \
                      self.get_leak_current(membrane_potential) + \
                      self.extrinsic_current

        return net_current / self.membrane_capacitance

    def get_synaptic_currents(self, membrane_potential: FloatLike) -> FloatLike:
        """Calculates the net synaptic current over all synapses for time `t`.

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

            # synaptic current
            if self.synapses[idx].conductivity_based:
                self.synaptic_currents[i] = self.synapses[idx].get_synaptic_current(
                    self.synaptic_input[0:self.current_input_idx[idx] + 1, idx], membrane_potential)
            else:
                self.synaptic_currents[i] = self.synapses[idx].get_synaptic_current(
                    self.synaptic_input[0:self.current_input_idx[idx] + 1, idx])

        # compute synaptic modulation value for each modulatory synapse and apply it to currents
        if self.modulatory_synapse_idx:

            self.get_synaptic_modulation(membrane_potential)

        return self.synaptic_currents @ self.synaptic_scalings

    def get_leak_current(self, membrane_potential: FloatLike) -> FloatLike:
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

    def get_synaptic_modulation(self, membrane_potential: FloatLike) -> None:
        """Calculates modulation weight for each additive synapse.

         Modulation values are applied to the synaptic current stored on the population afterwards.

        Parameters
        ----------
        membrane_potential
            Current membrane potential [unit = V].

        """

        for i, idx in enumerate(self.modulatory_synapse_idx):

            if self.synapses[idx].conductivity_based:
                self.synaptic_modulation[i] = self.synapses[idx].get_synaptic_current(
                    self.synaptic_input[0:self.current_input_idx + 1, idx], membrane_potential)
            else:
                self.synaptic_modulation[i] = self.synapses[idx].get_synaptic_current(
                    self.synaptic_input[0:self.current_input_idx + 1, idx])

        # apply modulation direction and get modulation value for each synapse
        synaptic_modulation_new = np.tile(self.synaptic_modulation, (len(self.synaptic_currents), 1))
        synaptic_modulation_new = synaptic_modulation_new ** self.synaptic_modulation_direction
        synaptic_modulation_new = np.prod(synaptic_modulation_new, axis=1)

        # combine extrinsic and intrinsic synaptic modulation
        if self.extrinsic_synaptic_modulation:
            synaptic_modulation_new *= self.extrinsic_synaptic_modulation

        self.synaptic_currents *= synaptic_modulation_new

    def get_delta_membrane_potential_threshold(self, membrane_potential_threshold: float) -> float:
        """Calculates change in axonal `membrane_potential_threshold` given current firing rate.

        Parameters
        ----------
        membrane_potential_threshold
            Current value of `membrane_potential_threshold` that needs to be updated [unit = V].

        Returns
        -------
        float
            Change in `membrane_potential_threshold` [unit = V].

        """

        return (self.current_firing_rate - self.firing_rate_target) / self.tau_axon

    def get_delta_synaptic_scaling(self, synaptic_scaling: float) -> float:
        """Calculates change in synaptic efficacy given current firing rate.

        Parameters
        ----------
        synaptic_scaling
            Current value used to scale the synaptic efficacy [unit = 1].

        Returns
        -------
        float
            Scaling of synaptic efficacy.

        """

        depression_rate = (synaptic_scaling * self.current_firing_rate) / \
                          (self.axon.transfer_function_args['max_firing_rate'] * self.tau_depression)
        recycle_rate = (1 - synaptic_scaling) / self.tau_recycle

        return recycle_rate - depression_rate

    def get_firing_rate(self) -> FloatLike:
        """Calculate the current average firing rate of the population.

        Returns
        -------
        float
            Average firing rate of population [unit = 1/s].

        """

        return self.axon.compute_firing_rate(self.state_variables[-1][0])

    def plot_synaptic_kernels(self, synapse_idx: Optional[List[int]]=None, create_plot: Optional[bool]=True,
                              axes=None) -> object:
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

        assert synapse_idx is None or type(synapse_idx) is list

        #############################
        # check positional argument #
        #############################

        if synapse_idx is None:
            synapse_idx = np.arange(len(self.synapses)).tolist()

        #########################
        # plot synaptic kernels #
        #########################

        if axes is None:
            fig, axes = plt.subplots(num='Synaptic Kernel Functions')

        synapse_types = list()
        for i in synapse_idx:
            axes = self.synapses[i].plot_synaptic_kernel(create_plot=False, axes=axes)
            synapse_types.append(self.synapses[i].synapse_type)

        plt.legend(synapse_types)

        if create_plot:
            fig.show()

        return axes

    def set_synapse(self, max_synaptic_delay: float, synapse_subtype: Optional[str]=None,
                    synapse_type: Optional[str]='DoubleExponentialSynapse',
                    synapse_params: Optional[dict]=None) -> None:
        """Instantiates synapse.

        Parameters
        ----------
        synapse_subtype
            Name of pre-parametrized synapse sub-class.
        max_synaptic_delay
            See docstring of parameter `max_synaptic_delay` of :class:`Population`.
        synapse_type
            Name of synapse class to instantiate.
        synapse_params
            Dictionary with synapse parameter name-value pairs.

        """

        if synapse_type == 'DoubleExponentialSynapse':
            self.synapses.append(set_instance(DoubleExponentialSynapse, synapse_subtype, synapse_params,
                                              bin_size=self.step_size, max_delay=max_synaptic_delay))
        else:
            raise AttributeError('Invalid synapse type!')

    def set_axon(self, axon_subtype, axon_type='SigmoidAxon', axon_params=None) -> None:
        """Instantiates axon

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
            self.axon = set_instance(SigmoidAxon, axon_subtype, axon_params)
        else:
            raise AttributeError('Invalid axon type!')

    def set_synapse_dependencies(self):
        """Sets important helper attributes on population to handle different synapse types etc.
        """

        n_synapses = len(self.synapses)

        # extract synapse types
        synapse_type = np.ones(n_synapses, dtype=bool)
        self.kernel_lengths = np.ones(n_synapses, dtype=int)
        for i, syn in enumerate(self.synapses):
            if syn.modulatory:
                synapse_type[i] = False
            self.kernel_lengths[i] = len(syn.synaptic_kernel)

        # set synaptic input array
        self.synaptic_input = np.zeros((int((np.max(self.max_synaptic_delay) + self.max_population_delay) / self.step_size),
                                        n_synapses))
        self.dummy_input = np.zeros((1, n_synapses))

        # set input index for each synapse
        self.current_input_idx = np.zeros(n_synapses, dtype=int)

        # set synaptic current and modulation collector vectors
        self.additive_synapse_idx = np.where(synapse_type == 1)[0]
        self.modulatory_synapse_idx = np.where(synapse_type == 0)[0]
        self.synaptic_currents = np.zeros(len(self.additive_synapse_idx))
        self.synaptic_modulation = np.zeros((1, len(self.modulatory_synapse_idx)))
        self.synaptic_scalings = np.ones(n_synapses)

        # set modulation direction for modulatory synapses
        if self.synaptic_modulation_direction is None and self.synaptic_modulation:
            self.synaptic_modulation_direction = np.ones(len(self.synaptic_currents), len(self.synaptic_modulation))
