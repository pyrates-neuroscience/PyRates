"""Module that includes basic population class.

A population is supposed to be the basic computational unit in the neural mass model. It contains various synapses
plus an axon hillok.

"""

import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from matplotlib.axes import Axes
from typing import List, Optional, Union, Dict, Callable, TypeVar


from core.axon import Axon, SigmoidAxon, BurstingAxon
from core.synapse import Synapse, DoubleExponentialSynapse, ExponentialSynapse, TransformedInputSynapse
from core.utility import set_instance, check_nones
from core.utility.filestorage import RepresentationBase

FloatLike = Union[float, np.float64]
AxonLike = TypeVar('AxonLike', bound=Axon, covariant=True)
SynapseLike = TypeVar('SynapseLike', bound=Synapse, covariant=True)

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


##############################
# leaky capacitor population #
##############################


class Population(RepresentationBase):
    """Base neural mass or population class, behaving like a leaky capacitor.

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
        synapses : :obj:`list` of :class:`Synapse` instances
            Synapse instances. See documentation of :class:`Synapse`.
        axon : :class:`Axon` instance
            Axon instance. See documentation of :class:`Axon`.
        state_variables : :obj:`list` of :obj:`np.ndarray`
            Collection of state variable vectors over state updates. Vector entries represent the following state
            variables:
            1) membrane potential [unit = V]
        synaptic_currents : np.ndarray
            Vector with synaptic currents produced by the non-modulatory synapses at time-point `t`.
        extrinsic_current : float
            Extrinsic current arriving at time-point `t`, affecting the membrane potential of the population.
        n_synapses : int
            Number of different synapse types in network.
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
                 tau_leak: float = 0.016,
                 resting_potential: float = -0.075,
                 membrane_capacitance: float = 1e-12,
                 max_population_delay: FloatLike = 0.,
                 synapse_params: Optional[List[dict]] = None,
                 axon_params: Optional[Dict[str, float]] = None,
                 synapse_class: Union[str, List[str]] = 'DoubleExponentialSynapse',
                 axon_class: str = 'SigmoidAxon',
                 store_state_variables: bool = True,
                 label: str = 'Custom'
                 ) -> None:
        """Instantiation of base population.
        """

        # check input parameters
        ########################

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

        # set population parameters
        ###########################

        self.synapses = []  # type: List[Synapse]
        self.state_variables = []  # type: List[List[FloatLike]]
        self.store_state_variables = store_state_variables
        self.tau_leak = tau_leak
        self.resting_potential = resting_potential
        self.step_size = step_size
        self.membrane_capacitance = membrane_capacitance
        self.label = label
        self.max_population_delay = max_population_delay

        # set initial states
        if type(init_state) not in (list, np.ndarray):
            self.state_variables.append([init_state])
        else:
            self.state_variables.append(init_state)

        # set synapses
        ##############

        # initialize synapse parameters
        self.n_synapses = len(synapses) if synapses else len(synapse_params)

        if max_synaptic_delay is None:
            max_synaptic_delay = np.array(check_nones(max_synaptic_delay, self.n_synapses))
        elif not isinstance(max_synaptic_delay, np.ndarray):
            max_synaptic_delay = np.zeros(self.n_synapses) + max_synaptic_delay

        # instantiate synapses
        self._set_synapses(synapse_subtypes=synapses,
                           synapse_params=synapse_params,
                           synapse_types=synapse_class,
                           max_synaptic_delay=max_synaptic_delay)

        # set synaptic current vector
        self.synaptic_currents = np.zeros(self.n_synapses)

        # set axon
        ##########

        self._set_axon(axon, axon_params=axon_params, axon_type=axon_class)

        # initialize extrinsic influences
        #################################

        self.extrinsic_current = 0.
        self.extrinsic_synaptic_modulation = np.ones(self.n_synapses)

    def _set_synapses(self,
                      synapse_subtypes: Optional[List[str]] = None,
                      synapse_types: Union[str, List[str]] = 'DoubleExponentialSynapse',
                      synapse_params: Optional[List[dict]] = None,
                      max_synaptic_delay: Optional[np.ndarray] = None
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
        max_synaptic_delay
            Array with maximal length of synaptic responses [unit = s].
        
        """

        # check synapse parameter formats
        #################################

        if isinstance(synapse_types, str):
            synapse_types = [synapse_types for _ in range(self.n_synapses)]

        synapse_subtypes = check_nones(synapse_subtypes, self.n_synapses)
        synapse_params = check_nones(synapse_params, self.n_synapses)
        self.synapse_labels = list()

        # set all given synapses
        ########################

        for i in range(self.n_synapses):

            # instantiate synapse
            if synapse_types[i] == 'DoubleExponentialSynapse':
                self.synapses.append(set_instance(DoubleExponentialSynapse,
                                                  synapse_subtypes[i],
                                                  synapse_params[i],
                                                  bin_size=self.step_size,
                                                  max_delay=max_synaptic_delay[i],
                                                  buffer_size=self.max_population_delay))
            elif synapse_types[i] == 'ExponentialSynapse':
                self.synapses.append(set_instance(ExponentialSynapse,
                                                  synapse_subtypes[i],
                                                  synapse_params[i],
                                                  bin_size=self.step_size,
                                                  max_delay=max_synaptic_delay[i],
                                                  buffer_size=self.max_population_delay))
            elif synapse_types[i] == 'TransformedInputSynapse':
                self.synapses.append(set_instance(TransformedInputSynapse,
                                                  synapse_subtypes[i],
                                                  synapse_params[i],
                                                  bin_size=self.step_size,
                                                  max_delay=max_synaptic_delay[i],
                                                  buffer_size=self.max_population_delay))
            elif synapse_types[i] == 'Synapse':
                self.synapses.append(set_instance(Synapse,
                                                  synapse_subtypes[i],
                                                  synapse_params[i],
                                                  bin_size=self.step_size,
                                                  max_delay=max_synaptic_delay[i],
                                                  buffer_size=self.max_population_delay))
            else:
                raise AttributeError('Invalid synapse type!')
            self.synapse_labels.append(self.synapses[-1].synapse_type)

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
            self.axon = set_instance(SigmoidAxon, axon_subtype, axon_params)  # type: ignore
        elif axon_type == 'BurstingAxon':
            self.axon = set_instance(BurstingAxon, axon_subtype, axon_params)  # type: ignore
            # re-build axonal kernel if synapse params have been passed
            if axon_params:
                self.axon.axon_kernel = self.axon.build_kernel()
        elif axon_type == 'Axon':
            self.axon = set_instance(Axon, axon_subtype, axon_params)  # type: ignore
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

        # compute synaptic currents and modulations
        ###########################################

        # calculate synaptic currents for each additive synapse
        for i, syn in enumerate(self.synapses):

            self.synaptic_currents[i] = syn.get_synaptic_current(membrane_potential)
        # TODO: multiplying with a vector of 1s by default costs computation time.
        return self.synaptic_currents @ self.extrinsic_synaptic_modulation.T


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

    def state_update(self,
                     extrinsic_current: FloatLike = 0.,
                     extrinsic_synaptic_modulation: Optional[np.ndarray] = None
                     ) -> None:
        """Updates state of population by making a single step forward in time.

        Parameters
        ----------
        extrinsic_current
            Extrinsic current arriving at time-point `t`, affecting the membrane potential of the population.
            (default = 0.) [unit = A].
        extrinsic_synaptic_modulation
            Modulatory (multiplicatory) input to each synapse. Vector with len = number of synapses 
            (default = None) [unit = 1].

        """

        # add inputs to internal state variables
        ########################################

        # extrinsic current
        self.extrinsic_current = extrinsic_current

        # extrinsic modulation
        if extrinsic_synaptic_modulation is not None:
            self.extrinsic_synaptic_modulation[0:len(extrinsic_synaptic_modulation)] = extrinsic_synaptic_modulation
        # fixme: this is called with arrays of 1s, although no acutal extrinsic synaptic modulation is defined.
        # fixme: --> performance

        # compute average membrane potential
        ####################################

        membrane_potential = self.state_variables[-1][0]
        membrane_potential = self.take_step(f=self.get_delta_membrane_potential,
                                            y_old=membrane_potential)

        state_vars = [membrane_potential]

        # update state variables
        ########################

        # TODO: Implement observer system here!!!
        self.state_variables.append(state_vars)
        if not self.store_state_variables:
            self.state_variables.pop(0)

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

        # copy synapse
        synapse = deepcopy(self.synapses[synapse_idx])

        # add copy to synapse list
        self.add_synapse(synapse, synapse_idx)

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

        # add synapse
        self.synapses.append(synapse)

        # update synapse dependencies
        #############################

        self.n_synapses = len(self.synapses)

        # current vector
        self.synaptic_currents = np.zeros(self.n_synapses)

        # extrinsic modulation vector
        ext_syn_mod = self.extrinsic_synaptic_modulation
        self.extrinsic_synaptic_modulation = np.ones(self.n_synapses)
        if synapse_idx:
            self.extrinsic_synaptic_modulation[0:-1] = ext_syn_mod
            self.extrinsic_synaptic_modulation[-1] = ext_syn_mod[synapse_idx]

    def clear(self):
        """Clears states stored on population and synaptic input.
        """

        # population attributes
        init_state = self.state_variables[0]
        self.state_variables.clear()
        self.state_variables.append(init_state)

        # synapse attributes
        for syn in self.synapses:
            syn.clear()

        # axon attributes
        self.axon.clear()

    def update(self) -> None:
        """update population attributes.
        """

        # update synapses
        for syn in self.synapses:
            syn.bin_size = self.step_size
            syn.buffer_size = self.max_population_delay
            syn.update()

        # update axon
        if hasattr(self.axon, 'bin_size'):
            self.axon.bin_size = self.step_size
        self.axon.update()

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

        # check parameters
        ##################

        assert synapse_idx is None or isinstance(synapse_idx, list)

        # check positional argument
        ###########################

        if synapse_idx is None:
            synapse_idx = list(range(self.n_synapses))

        # plot synaptic kernels
        #######################

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


######################################
# plastic leaky capacitor population #
######################################


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

        # call super init
        #################

        super().__init__(synapses=synapses,
                         axon=axon,
                         init_state=init_state,
                         step_size=step_size,
                         max_synaptic_delay=max_synaptic_delay,
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

        # set plasticity attributes
        ###########################

        # for axon
        if axon_plasticity_function:
            if not axon_plasticity_target_param or not axon_plasticity_function_params:
                raise ValueError("If an axon_plasticity_function was given, then also axon_plasticity_target_param and "
                                 "axon_plasticity_function_params need to be specified.")
            self.axon_plasticity_target_param = axon_plasticity_target_param
            if axon_plasticity_function_params:
                self.axon_plasticity_function_params = axon_plasticity_function_params
            else:
                self.axon_plasticity_function_params = dict()
            self.state_variables[-1] += [self.axon.transfer_function_args[self.axon_plasticity_target_param]]
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
            self.synapse_plasticity_function_params = check_nones(synapse_plasticity_function_params, self.n_synapses)

        if self.synapse_plasticity_function:
            for i in range(self.n_synapses):
                if self.synapse_plasticity_function_params[i]:
                    self.state_variables[-1] += [self.synapses[i].depression]

    def state_update(self,
                     extrinsic_current: FloatLike = 0.,
                     extrinsic_synaptic_modulation: Optional[np.ndarray] = None,
                     ) -> None:
        """Updates state of population by making a single step forward in time.

        Parameters
        ----------
        extrinsic_current
            Extrinsic current arriving at time-point `t`, affecting the membrane potential of the population.
            (default = 0.) [unit = A].
        extrinsic_synaptic_modulation
            Modulatory input to each synapse. Vector with len = number of synapses (default = 1.0) [unit = 1].

        """

        # call super state update
        #########################

        super().state_update(extrinsic_current=extrinsic_current,
                             extrinsic_synaptic_modulation=extrinsic_synaptic_modulation)

        # update axonal transfer function
        #################################

        if self.axon_plasticity_function:

            # update axon
            self.axon.transfer_function_args[self.axon_plasticity_target_param] = \
                Population.take_step(self,
                                     f=self.axon_plasticity_function,
                                     y_old=self.axon.transfer_function_args[self.axon_plasticity_target_param],
                                     firing_rate_target=self.get_firing_rate(),
                                     **self.axon_plasticity_function_params)

            # update state vector
            self.state_variables[-1] += [self.axon.transfer_function_args[self.axon_plasticity_target_param]]

        # update synaptic scaling
        #########################

        if self.synapse_plasticity_function:

            for i in range(self.n_synapses):

                if self.synapse_plasticity_function_params[i]:

                    # update synaptic depression
                    self.synapses[i].depression = Population.take_step(self,
                                                                       f=self.synapse_plasticity_function,
                                                                       y_old=self.synapses[i].depression,
                                                                       firing_rate=self.synapses[i].synaptic_input
                                                                       [self.synapses[i].kernel_length],
                                                                       **self.synapse_plasticity_function_params[i])
                    # update state vector
                    self.state_variables[-1] += [self.synapses[i].depression]

    def add_plastic_synapse(self,
                            synapse_idx: int,
                            synapse: Optional[Synapse] = None,
                            max_firing_rate: Optional[float] = None) -> None:
        """Adds copy of specified synapse to population.

        Parameters
        ----------
        synapse
            Synapse object to add (default = None)
        synapse_idx
            Index of synapse to copy (default = None).
        max_firing_rate
            Maximum firing rate of connecting population. Used for synaptic plasticity mechanism (default = None).

        """

        # call super method
        ###################

        if synapse:
            self.add_synapse(synapse, synapse_idx)
        else:
            self.copy_synapse(synapse_idx)

        # check plasticity related stuff
        ################################

        if max_firing_rate is None:
            max_firing_rate = self.axon.transfer_function_args['max_firing_rate']

        self.synapse_plasticity_function_params.append(self.synapse_plasticity_function_params[synapse_idx])
        self.synapse_plasticity_function_params[-1]['max_firing_rate'] = max_firing_rate


##############################
# jansen-rit type population #
##############################


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


######################################
# plastic jansen-rit type population #
######################################


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


class DummyPopulation(object):
    """Population object used to pass certain inputs to other populations in a network.
    
    Parameters
    ----------
    output
        Firing rates to be passed through network
    
    """

    def __init__(self, output: np.ndarray):
        """Instantiate dummy population.
        """

        self.output = output
        self.idx = 0

    def get_firing_rate(self):
        """Pass output at current index position.
        """

        output = self.output[self.idx]
        self.idx += 1

        return output
