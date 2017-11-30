"""Module that includes basic population class plus derivations of it.

A population is supposed to be the basic computational unit in the neural mass model. It contains various synapses
plus an axon hillok.

"""

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import LSODA
import numpy as np

from core.axon import Axon, SigmoidAxon
import core.axon.templates as axon_templates
from core.synapse import Synapse, DoubleExponentialSynapse
import core.synapse.templates as synapse_templates

from typing import List, Optional, Union, Dict, Callable, TypeVar, Iterable
FloatLike = Union[float, np.float64]
AxonLike = TypeVar('AxonLike', bound=Axon, covariant=True)
SynapseLike = TypeVar('SynapseLike', bound=Synapse, covariant=True)

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"

# TODO: Implement synaptic plasticity mechanism(s)
# TODO: Implement new function that updates synaptic input field according to input + delay
# TODO: Rework set-up of state vector to have fixed positions for certain state variables (i.e. use fixed size vector)
# TODO: Move helper functions to separate file and set_synapse/_axon to object


class Population(object):
    """Base neural mass or population class.

    A population is defined via a number of synapses and an axon.

    Parameters
    ----------
    synapses
        Can be set to use default synapse types. These include:
        :class:`synapse_templates.AMPACurrentSynapse`,
        :class:`synapse_templates.GABAACurrentSynapse`,
        :class:`synapse_templates.AMPAConductanceSynapse`,
        :class:`synapse_templates.GABAAConductanceSynapse`.
    axon
        Can be set to use default axon types. These include:
        :class:`axon_templates.JansenRitAxon`.
    init_state
        Vector defining initial state of the population. Vector entries represent the following state variables:
        1) membrane potential (default = 0.0) [unit = V].
    step_size
        Time step-size of a single state update (default = 0.0001) [unit = s].
    variable_step_size
        If true, time step-size of each state update will be chosen automatically by LSODA algorithm.
        Else fixed step_size will be used with Euler formalism (default = False).
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
    variable_step_size
        See documentation of parameter `variable_step_size`.
    membrane_capacitance
        See documentation of parameter `membrane_capacitance`.

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

    """

    def __init__(self, synapses: List[str],
                 axon: str,
                 init_state: FloatLike=0.,
                 step_size: float=0.0001,
                 variable_step_size: bool=False,
                 max_synaptic_delay: float=0.1,
                 synaptic_modulation_direction: Optional[np.ndarray]=None,
                 tau_leak: float=0.016,
                 resting_potential: float=-0.075,
                 membrane_capacitance: float=1e-12,
                 max_population_delay: FloatLike=0.,
                 synapse_params: Optional[List[Dict[str, Union[bool, float]]]]=None,
                 axon_params: Optional[Dict[str, float]]=None,
                 store_state_variables: bool=False) -> None:
        """Instantiation of base population.
        """

        ##########################
        # check input parameters #
        ##########################

        if step_size < 0 or tau_leak < 0 or max_synaptic_delay < 0 or max_population_delay < 0:
            raise ValueError('Time constants (tau, step_size, max_delay) cannot be negative. '
                             'See parameter docstring for further information.')

        if membrane_capacitance < 0:
            raise ValueError('Membrane capacitance cannot be negative. See docstring for further information.')

        #############################
        # set population parameters #
        #############################

        self.synapses = list()
        self.state_variables = list()
        self.store_state_variables = store_state_variables
        self.tau_leak = tau_leak
        self.resting_potential = resting_potential
        self.step_size = step_size
        self.variable_step_size = variable_step_size
        self.membrane_capacitance = membrane_capacitance
        self.t = 0.

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

        # TODO: implement synaptic plasticity mechanism

        ###############
        # set synapse #
        ###############

        synapse_type = np.ones(len(synapses), dtype=bool)

        if not synapse_params:

            # use pre-parametrized synapse types
            for i in range(len(synapses)):
                self.synapses.append(set_synapse(synapses[i], step_size, max_synaptic_delay))
                if self.synapses[-1].neuromodulatory:
                    synapse_type[i] = False

        else:

            # use custom synapse parametrization
            for i in range(len(synapse_params)):
                self.synapses.append(set_synapse(synapses[i], step_size, max_synaptic_delay, synapse_params[i]))
                if self.synapses[-1].neuromodulatory:
                    synapse_type[i] = False

        # set synaptic input array
        self.synaptic_input = np.zeros((max_synaptic_delay + max_population_delay, len(self.synapses)))
        self.dummy_input = np.zeros((1, len(self.synapses)))

        # set input index for each synapse
        self.current_input_idx = 0

        # set synaptic current and modulation collector vectors
        self.additive_synapse_idx = np.where(synapse_type == 1)[0]
        self.modulatory_synapse_idx = np.where(synapse_type == 0)[0]
        self.synaptic_currents = np.zeros(len(self.additive_synapse_idx))
        self.synaptic_modulation = np.zeros((1, len(self.modulatory_synapse_idx)))

        # set modulation direction for modulatory synapses
        if synaptic_modulation_direction is None and self.synaptic_modulation:
            self.synaptic_modulation_direction = np.ones(len(self.synaptic_currents), len(self.synaptic_modulation))

        ############
        # set axon #
        ############

        self.axon = set_axon(axon, axon_params)

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
        # update synaptic kernels #
        ###########################

        # TODO: implement differential equation that adapts synaptic efficiencies

        ##########################
        # update state variables #
        ##########################

        self.state_variables.append(state_vars)

        if not self.store_state_variables:
            self.state_variables.pop(0)

        ################################################
        # advance in time and in synaptic input vector #
        ################################################

        # time
        self.t += self.step_size

        # synaptic input
        if self.current_input_idx < self.synapses[0].kernel_length - 1:

            self.current_input_idx += 1

        else:

            self.synaptic_input = np.append(self.synaptic_input, self.dummy_input, axis=0)
            self.synaptic_input = self.synaptic_input[1:, :]

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

        if self.variable_step_size:

            # initialize RK45 solver
            solver = LSODA(fun=f,
                           t0=0.,
                           y0=[y_old],
                           t_bound=float('inf'),
                           min_step=self.synapses[0].step_size,
                           rtol=1e-2,
                           atol=1e-3)

            # perform integration step and calculate output
            solver.step()
            y_new = np.squeeze(solver.y)

            # update internal step-size
            self.step_size = solver.t - solver.t_old

        else:

            # perform Euler update
            y_new = y_old + self.step_size * f(self.t, y_old)

        return y_new

    def get_delta_membrane_potential(self, t: Union[int, float], membrane_potential: FloatLike
                                     ) -> FloatLike:
        """Calculates change in membrane potential as function of synaptic current, leak current and
        extrinsic current.

        Parameters
        ----------
        t
            Time for which to calculate the delta term [unit = s].
        membrane_potential
            Current membrane potential of population [unit = V].

        Returns
        -------
        float
            Delta membrane potential [unit = V].

        """

        # fixme: variable t is ambiguously defined and seemingly never used. Remove?
        net_current = self.get_synaptic_currents(t, membrane_potential) + \
                      self.get_leak_current(membrane_potential) + \
                      self.extrinsic_current

        return net_current / self.membrane_capacitance

    def get_synaptic_currents(self, t: Union[int, float], membrane_potential: FloatLike
                              ) -> FloatLike:
        """Calculates the net synaptic current over all synapses for time `t`.

        Parameters
        ----------
        t
            Time for which to calculate the synaptic current [unit = s].
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
                    self.synaptic_input[0:self.current_input_idx + 1, idx], membrane_potential)
            else:
                self.synaptic_currents[i] = self.synapses[idx].get_synaptic_current(
                    self.synaptic_input[0:self.current_input_idx + 1, idx])

        # compute synaptic modulation value for each modulatory synapse and apply it to currents
        if self.modulatory_synapse_idx:

            self.get_synaptic_modulation(membrane_potential)

        # (disable type checker for return value)
        # noinspection PyTypeChecker
        return np.sum(self.synaptic_currents)  # type hint fails here, because np.sum may return both an array or scalar

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

    def get_delta_membrane_potential_threshold(self, t: float, membrane_potential_threshold: float) -> float:
        """Calculates change in axonal `membrane_potential_threshold` given current firing rate.

        Parameters
        ----------
        t
            Time-point for which to calculate the change in `membrane_potential_threshold` [unit = s].
        membrane_potential_threshold
            Current value of `membrane_potential_threshold` that needs to be updated [unit = V].

        Returns
        -------
        float
            Change in `membrane_potential_threshold` [unit = V].

        """
        # fixme: the parameters don't seem to be used at all. Remove?
        return (self.get_firing_rate() - self.firing_rate_target) / self.tau_axon

    def get_firing_rate(self) -> FloatLike:
        """Calculate the current average firing rate of the population.

        Returns
        -------
        float
            Average firing rate of population [unit = 1/s].

        """

        return self.axon.compute_firing_rate(self.state_variables[-1][0])

    def plot_synaptic_kernels(self, synapse_idx: Optional[List[int]]=None, create_plot: Optional[bool]=True,
                              fig=None) -> None:
        """Creates plot of all specified synapses over time.

        Parameters
        ----------
        synapse_idx
            Index of synapses for which to plot kernel.
        create_plot
            If true, plot will be shown.
        fig
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
            synapse_idx = range(len(self.synapses))

        #########################
        # plot synaptic kernels #
        #########################

        if fig is None:
            fig = plt.figure('Synaptic Kernel Functions')

        synapse_types = list()
        for i in synapse_idx:           # fixme: what is the problem here?
            fig = self.synapses[i].plot_synaptic_kernel(create_plot=False, fig=fig)
            synapse_types.append(self.synapses[i].synapse_type)

        plt.legend(synapse_types)

        if create_plot:
            fig.show()

        return fig


def set_synapse(synapse: str, step_size: float, max_delay: float,
                synapse_params: Optional[Dict[str, Union[bool, float]]]=None) -> SynapseLike:
    """
    Instantiates a synapse, including a method to calculate the synaptic current resulting from the average firing rate
    arriving at the population.

    :param synapse: character string that indicates synapse type (see pre-implemented synapse sub-classes)
    :param step_size: scalar, size of the time step for which the population state will be updated according
           to euler formalism [unit = s].
    :param max_delay: scalar that indicates number of bins the kernel should be evaluated for
           [unit = 1].
    :param synapse_params: dictionary containing parameters for custom synapse type. For parameter explanation see
           synapse class (default = None).

    :return synapse_instance: instance of synapse class, including method to compute synaptic current

    """
    # fixme: set return type hint to a subclass of synapse

    ####################
    # check parameters #
    ####################

    # assert (type(synapse) is str) or (not synapse)
    # assert (type(synapse_params) is dict) or (not synapse_params)
    assert step_size > 0
    assert max_delay > 0

    ###############################
    # initialize synapse instance #
    ###############################

    pre_defined_synapse = True

    if synapse == 'AMPA_current':

        syn_instance = synapse_templates.AMPACurrentSynapse(step_size, max_delay)

    elif synapse == 'GABAA_current':

        syn_instance = synapse_templates.GABAACurrentSynapse(step_size, max_delay)

    elif synapse_params:  # fixme: this looks, like "synapse" could be made optional

        syn_instance = DoubleExponentialSynapse(efficacy=synapse_params['efficacy'],
                                                tau_decay=synapse_params['tau_decay'],
                                                tau_rise=synapse_params['tau_rise'],
                                                bin_size=step_size,
                                                max_delay=max_delay)
        pre_defined_synapse = False

    else:

        raise ValueError('Not a valid synapse type! Check synapses.py for all possible synapse types')

    #########################################################################
    # adjust pre-defined parametrization if relevant parameters were passed #
    #########################################################################

    if pre_defined_synapse and synapse_params:

        param_list = ['efficacy', 'tau_decay', 'tau_rise', 'conductivity_based', 'reversal_potential']

        for p in param_list:
            syn_instance = update_param(p, synapse_params, syn_instance)

    syn_instance.synaptic_kernel = syn_instance.evaluate_kernel(build_kernel=True)

    return syn_instance


def set_axon(axon: Union[str, None], axon_params: Optional[Dict[str, float]]=None) -> AxonLike:
    """
    Instantiates an axon, including a method to calculate the average output firing rate of a population given its
    current membrane potential.

    :param axon: character string that indicates axon type (see pre-implemented axon sub-classes)
    :param axon_params: dictionary containing parameters for custom axon type. For parameter explanation see axon
           class (default = None).

    :return ax_instance: instance of axon class, including method to compute firing rate

    """
    # fixme: set return type hint to a subclass of axon

    ####################
    # check parameters #
    ####################

    assert (type(axon) is str) or (not axon)
    assert (type(axon_params) is dict) or (not axon_params)

    ############################
    # initialize axon instance #
    ############################

    pre_defined_axon = True

    if axon == 'JansenRit':

        ax_instance = axon_templates.JansenRitAxon()

    elif axon_params:

        ax_instance = SigmoidAxon(axon_params['max_firing_rate'], axon_params['membrane_potential_threshold'],
                                  axon_params['sigmoid_steepness'])
        pre_defined_axon = False

    else:

        raise ValueError('Not a valid axon type!')

    #########################################################################
    # adjust pre-defined parametrization if relevant parameters were passed #
    #########################################################################

    if pre_defined_axon and axon_params:

        param_list = ['max_firing_rate', 'membrane_potential_threshold', 'sigmoid_steepness']

        for p in param_list:
            ax_instance = update_param(p, axon_params, ax_instance)

    return ax_instance


def update_param(param, param_dict, object_instance):
    """
    Checks whether param is a key in param_dict. If yes, the corresponding value in param_dict will be updated in
    object_instance.

    :param param: string, specifies parameter to check
    :param param_dict: dictionary, potentially contains param
    :param object_instance: object instance for which to update parameter

    :return: object
    """

    assert param in object_instance

    if param in param_dict:
        setattr(object_instance, param, param_dict[param])

    return object_instance


def interpolate_array(old_step_size, new_step_size, y, interpolation_type='cubic', axis=0):
    """
    Interpolates time-vectors with scipy.interpolate.interp1d.

    # :param old_t: total time-length of y [unit = s].
    :param old_step_size: old simulation step size [unit = s].
    :param new_step_size: new simulation step size [unit = s].
    # :param new_t: new total length of y [unit = s].
    :param y: vector to be interpolated
    :param interpolation_type: can be 'linear' or spline stuff.
    :param axis: axis along which y is to be interpolated (has to have same length as t/old_step_size)

    :return: interpolated vector

    """

    # create time vectors
    x_old = np.arange(y.shape[axis]) * old_step_size

    new_steps_n = int(np.ceil(x_old[-1] / new_step_size))
    x_new = np.linspace(0, x_old[-1], new_steps_n)

    # create interpolation function
    f = interp1d(x_old, y, axis=axis, kind=interpolation_type, bounds_error=False, fill_value='extrapolate')

    return f(x_new)
