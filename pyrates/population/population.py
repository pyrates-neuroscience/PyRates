"""Module that includes basic population class.

A population is supposed to be the basic computational unit in the neural mass model. It contains various synapses
plus an axon hillok.

"""

import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from matplotlib.axes import Axes
from typing import List, Optional, Union, Dict, Callable, TypeVar
from types import MethodType


from pyrates.axon import Axon, SigmoidAxon, BurstingAxon, PlasticSigmoidAxon
from pyrates.synapse import Synapse, DoubleExponentialSynapse, ExponentialSynapse, TransformedInputSynapse, \
    DEExponentialSynapse, DESynapse, DEDoubleExponentialSynapse
from pyrates.utility import set_instance, check_nones
from pyrates.population.population_methods import construct_state_update_function, \
    construct_get_delta_membrane_potential_function

from pyrates.utility.filestorage import RepresentationBase

FloatLike = Union[float, np.float64]
AxonLike = TypeVar('AxonLike', bound=Axon, covariant=True)
SynapseLike = TypeVar('SynapseLike', bound=Synapse, covariant=True)

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


##############################
# leaky capacitor population #
##############################

class AbstractBasePopulation(RepresentationBase):
    """Very base class that includes DummyPopulation, to ensure consistency"""
    def __init__(self):
        self.targets = []
        self.target_weights = []
        self.target_delays = []

    def connect(self, target_synapse: Synapse, weight: float, delay: int):
        """Connect to a given synapse at a target population with a given weight and delay.

        Parameters
        ----------
        target_synapse
            Synapse that population is supposed to connect to
        weight
            Weight that is to be applied to the connection (=connectivity)
        delay
            Index (=time delay) at which the output is supposed to arrive at the target synapse"""

        self.targets.append(target_synapse)
        self.target_weights.append(weight)
        self.target_delays.append(delay)

    def disconnect(self):
        """Removes all connections of population.
        """

        self.targets.clear()
        self.target_weights.clear()
        self.target_delays.clear()

    def project_to_targets(self) -> None:
        """Projects output of given population to the other circuit populations its connected to.

        Parameters
        ----------


        """

        # get source firing rate
        source_fr = self.get_firing_rate()

        # project source firing rate to connected populations
        #####################################################

        # extract network connections
        # targets = self.targets

        # loop over target populations connected to source
        for i, syn in enumerate(self.targets):
            syn.pass_input(source_fr * self.target_weights[i], delay=self.target_delays[i])
            # it would be possible to wrap this function using functools.partial to remove the delay lookup

    def get_firing_rate(self):
        """Returns the firing rate as output of the population."""

        raise NotImplementedError("Method `_get_firing_rate` needs to be implemented in subclass.")
        # a cleaner way would be to introduce a metaclass and decorate with abc.abstractmethod


class PopulationOld(AbstractBasePopulation):
    """Base neural mass or population class, behaving like a leaky capacitor.

        A population is defined via a number of synapses and an axon.

        Parameters
        ----------
        synapses
            Can be set to use default synapse types. These include:
            :class:`pyrates.synapse.templates.AMPACurrentSynapse`,
            :class:`pyrates.synapse.templates.GABAACurrentSynapse`,
            :class:`pyrates.synapse.templates.AMPAConductanceSynapse`,
            :class:`pyrates.synapse.templates.GABAAConductanceSynapse`.
        axon
            Can be set to use default axon types. These include:
            :class:`pyrates.axon.templates.JansenRitAxon`.
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
                 tau_leak: Optional[float] = None,
                 resting_potential: float = -0.075,
                 membrane_capacitance: float = 1e-12,
                 max_population_delay: FloatLike = 0.,
                 synapse_params: Optional[List[dict]] = None,
                 axon_params: Optional[Dict[str, float]] = None,
                 synapse_class: Union[str, List[str]] = 'DoubleExponentialSynapse',
                 axon_class: str = 'SigmoidAxon',
                 store_state_variables: bool = True,
                 label: str = 'Custom',
                 synapse_labels: Optional[list] = None,
                 enable_modulation: bool = False
                 ) -> None:
        """Instantiation of base population.
        """
        super().__init__()

        # check input parameters
        ########################

        # check synapse/axon attributes
        if not synapses and not synapse_params:
            raise AttributeError('Either synapses or synapse_params have to be passed!')
        if not axon and not axon_params:
            raise AttributeError('Either axon or axon_params have to be passed!')

        # check attribute values
        if step_size < 0 or max_population_delay < 0 or (tau_leak and tau_leak < 0):
            raise ValueError('Time constants (tau, step_size, max_delay) cannot be negative. '
                             'See parameter docstring for further information.')
        if membrane_capacitance < 0:
            raise ValueError('Membrane capacitance cannot be negative. See docstring for further information.')

        # set population parameters
        ###########################

        self.synapses = []  # type: List[Union[Synapse, DESynapse]]
        self.state_variables = []  # type: List[List[FloatLike]]
        self.store_state_variables = store_state_variables
        self.step_size = step_size
        self.label = label
        self.max_population_delay = max_population_delay

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
                           max_synaptic_delay=max_synaptic_delay,
                           synapse_labels=synapse_labels)

        # set synaptic current vector
        self.synaptic_currents = np.zeros(self.n_synapses)

        # set axon
        ##########

        self._set_axon(axon, axon_params=axon_params, axon_type=axon_class)

        # set initial states
        if type(init_state) not in (list, np.ndarray):
            self.state_variables.append([init_state])
        else:
            self.state_variables.append(init_state)

        self.axon.compute_firing_rate(self.state_variables[-1][0])

        # set appropriate state update functions
        ########################################

        # choose between leaky-capacitor or kernel description of change in membrane potential
        if tau_leak:

            self.tau_leak = tau_leak
            self.membrane_capacitance = membrane_capacitance
            self.resting_potential = resting_potential

            self.get_delta_membrane_potential = MethodType(get_delta_membrane_potential_lc, self)

        else:

            self.get_delta_membrane_potential = MethodType(get_delta_membrane_potential, self)

        # choose between enabling modulatory influences or not
        if enable_modulation:

            self.state_update = MethodType(state_update, self)
            self.extrinsic_synaptic_modulation = np.ones(self.n_synapses)
            self.get_synaptic_currents = MethodType(get_synaptic_currents, self)

        else:

            self.state_update = MethodType(state_update_no_modulation, self)
            self.get_synaptic_currents = MethodType(get_synaptic_currents_no_modulation, self)

        # initialize extrinsic influences
        #################################

        self.extrinsic_current = 0.

    def _set_synapses(self,
                      synapse_subtypes: Optional[List[str]] = None,
                      synapse_types: Union[str, List[str]] = 'DoubleExponentialSynapse',
                      synapse_params: Optional[List[dict]] = None,
                      max_synaptic_delay: Optional[np.ndarray] = None,
                      synapse_labels: Optional[List[str]] = None
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
        synapse_labels
        
        """

        # check synapse parameter formats
        #################################

        if isinstance(synapse_types, str):
            synapse_types = [synapse_types for _ in range(self.n_synapses)]

        synapse_subtypes = check_nones(synapse_subtypes, self.n_synapses)
        synapse_params = check_nones(synapse_params, self.n_synapses)
        synapse_labels = check_nones(synapse_labels, self.n_synapses)
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
            if synapse_labels[i]:
                self.synapses[i].synapse_type = synapse_labels[i]
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
            self.axon = set_instance(SigmoidAxon, axon_subtype, axon_params)  # type: Union[Axon, BurstingAxon]
        elif axon_type == 'BurstingAxon':
            self.axon = set_instance(BurstingAxon, axon_subtype, axon_params)  # type: Union[Axon, BurstingAxon]
            # re-build axonal kernel if synapse params have been passed
            if axon_params:
                self.axon.axon_kernel = self.axon.build_kernel()
        elif axon_type == 'Axon':
            self.axon = set_instance(Axon, axon_subtype, axon_params)  # type: Union[Axon, BurstingAxon]
        else:
            raise AttributeError('Invalid axon type!')

    def get_firing_rate(self) -> FloatLike:
        """Calculate the current average firing rate of the population.

        Returns
        -------
        float
            Average firing rate of population [unit = 1/s].

        """

        return self.axon.firing_rate

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
        self.synaptic_currents[:] = 0.

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


class PlasticPopulationOld(PopulationOld):
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
    spike_frequency_adaptation
        Function defining the axonal plasticity mechanisms to be used on population.
    spike_frequency_adaptation_args
        Name-value pairs defining the parameters for the axonal plasticity function.
    synapse_efficacy_adaptation
        Function defining the synaptic plasticity mechanisms to be used on population.
    synapse_efficacy_adaptation_args
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
                 axon_class: str = 'PlasticSigmoidAxon',
                 store_state_variables: bool = False,
                 label: str = 'Custom',
                 enable_modulation: bool = False,
                 spike_frequency_adaptation: Optional[Callable[[float], float]] = None,
                 spike_frequency_adaptation_args: Optional[dict] = None,
                 synapse_efficacy_adaptation: Optional[List[Callable[[float], float]]] = None,
                 synapse_efficacy_adaptation_args: Optional[List[dict]] = None
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
                         label=label,
                         enable_modulation=enable_modulation)

        # set plasticity attributes
        ###########################

        # for axon
        self.spike_frequency_adaptation = spike_frequency_adaptation
        if self.spike_frequency_adaptation:
            self.spike_frequency_adaptation_args = spike_frequency_adaptation_args if spike_frequency_adaptation_args \
                else dict()
            self.state_variables[-1] += [self.axon.transfer_function_args['adaptation']]

        # synaptic plasticity function
        if synapse_efficacy_adaptation and (type(synapse_efficacy_adaptation) is not list):
            self.synapse_efficacy_adaptation = [synapse_efficacy_adaptation for _ in range(self.n_synapses)]
        elif synapse_efficacy_adaptation and (len(synapse_efficacy_adaptation) != self.n_synapses):
            raise ValueError('If list of synaptic plasticity functions is passed, its length must correspond to the'
                             'number of synapses of the population.')
        elif not synapse_efficacy_adaptation:
            self.synapse_efficacy_adaptation = [None for _ in range(self.n_synapses)]
        else:
            self.synapse_efficacy_adaptation = synapse_efficacy_adaptation

        # synaptic plasticity state variables
        for i in range(self.n_synapses):
            if self.synapse_efficacy_adaptation[i]:
                self.state_variables[-1] += [self.synapses[i].depression]

        # synaptic plasticity function arguments
        if synapse_efficacy_adaptation_args is None or type(synapse_efficacy_adaptation_args) is dict:
            self.synapse_efficacy_adaptation_args = [synapse_efficacy_adaptation_args for _ in range(self.n_synapses)]
        else:
            self.synapse_efficacy_adaptation_args = synapse_efficacy_adaptation_args

        if len(self.synapse_efficacy_adaptation) != len(self.synapse_efficacy_adaptation_args):
            raise AttributeError('Number of synaptic plasticity functions and plasticity function parameter '
                                 'dictionaries has to be equal.')

        # overwrite state update function
        #################################

        if enable_modulation:
            self.state_update = MethodType(state_update_plastic, self)
        else:
            self.state_update = MethodType(state_update_plastic_no_modulation, self)

    def _set_axon(self,
                  axon_subtype: Optional[str] = None,
                  axon_type: str = 'SigmoidAxon',
                  axon_params: Optional[dict] = None
                  ):

        if axon_type == 'PlasticSigmoidAxon':
            self.axon = set_instance(PlasticSigmoidAxon, axon_subtype, axon_params)  # type: Union[Axon, BurstingAxon]
        else:
            super()._set_axon(axon_subtype, axon_type, axon_params)

    def axon_update(self):
        """Updates adaptation field of axon.
        """

        self.axon.transfer_function_args['adaptation'] = self.take_step(f=self.spike_frequency_adaptation,
                                                                        y_old=self.axon.transfer_function_args
                                                                        ['adaptation'],
                                                                        firing_rate=self.get_firing_rate(),
                                                                        **self.spike_frequency_adaptation_args)

    def synapse_update(self, idx: int):
        """Updates depression field of synapse.
        
        Parameters
        ----------
        idx
            Synapse index.
            
        """

        self.synapses[idx].depression = self.take_step(f=self.synapse_efficacy_adaptation[idx],
                                                       y_old=self.synapses[idx].depression,
                                                       firing_rate=self.synapses[idx].synaptic_input[
                                                                   self.synapses[idx].kernel_length-2],
                                                       **self.synapse_efficacy_adaptation_args[idx])

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

        self.synapse_efficacy_adaptation.append(self.synapse_efficacy_adaptation[synapse_idx])
        self.synapse_efficacy_adaptation_args.append(self.synapse_efficacy_adaptation_args[synapse_idx])
        self.synapse_efficacy_adaptation_args[-1]['max_firing_rate'] = max_firing_rate
        self.state_variables[-1] += [self.synapses[synapse_idx].depression]


##############################
# jansen-rit type population #
##############################


class SecondOrderPopulationOld(PopulationOld):
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
    synapse_labels
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
                 step_size: float = 1e-4,
                 tau_leak: Optional[float] = None,
                 resting_potential: float = 0.,
                 membrane_capacitance: float = 1e-12,
                 max_population_delay: FloatLike = 0.,
                 max_synaptic_delay: Optional[float]=None,
                 synapse_params: Optional[List[dict]] = None,
                 axon_params: Optional[Dict[str, float]] = None,
                 synapse_class: Union[str, List[str]] = 'DEExponentialSynapse',
                 axon_class: str = 'SigmoidAxon',
                 store_state_variables: bool = False,
                 label: str = 'Custom',
                 enable_modulation: bool = False,
                 synapse_labels: Optional[List[str]] = None
                 ) -> None:
        """Instantiation of second order population.
        """

        # call super init
        #################

        super().__init__(synapses=synapses,
                         axon=axon,
                         init_state=init_state,
                         step_size=step_size,
                         resting_potential=resting_potential,
                         max_population_delay=max_population_delay,
                         synapse_params=synapse_params,
                         axon_params=axon_params,
                         synapse_class=synapse_class,
                         axon_class=axon_class,
                         store_state_variables=store_state_variables,
                         label=label,
                         enable_modulation=enable_modulation,
                         synapse_labels=synapse_labels,
                         tau_leak=tau_leak,
                         membrane_capacitance=membrane_capacitance)

        # set additional synapse handling vectors
        #########################################

        self.synaptic_currents_old = np.zeros_like(self.synaptic_currents)
        self.PSPs = np.zeros_like(self.synaptic_currents) + init_state

        # set appropriate state update methods
        ######################################

        if enable_modulation:

            self.get_delta_psp = MethodType(get_delta_psp, self)

            if tau_leak:
                self.state_update = MethodType(state_update_differential_lc, self)
            else:
                self.state_update = MethodType(state_update_differential, self)

        else:

            self.get_delta_psp = MethodType(get_delta_psp_no_modulation, self)

            if tau_leak:
                self.state_update = MethodType(state_update_differential_lc_no_modulation, self)
            else:
                self.state_update = MethodType(state_update_differential_no_modulation, self)

    def _set_synapses(self,
                      synapse_subtypes: Optional[List[str]] = None,
                      synapse_types: Union[str, List[str]] = 'DEExponentialSynapse',
                      synapse_params: Optional[List[dict]] = None,
                      max_synaptic_delay: Optional[float] = None,
                      synapse_labels: Optional[List[str]] = None
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
        synapse_labels

        """

        # check synapse parameter formats
        #################################

        if isinstance(synapse_types, str):
            synapse_types = [synapse_types for _ in range(self.n_synapses)]

        synapse_subtypes = check_nones(synapse_subtypes, self.n_synapses)
        synapse_params = check_nones(synapse_params, self.n_synapses)
        synapse_labels = check_nones(synapse_labels, self.n_synapses)
        self.synapse_labels = list()

        # set all given synapses
        ########################

        for i in range(self.n_synapses):

            # instantiate synapse
            if synapse_types[i] == 'ExponentialSynapse':
                self.synapses.append(set_instance(DEExponentialSynapse,
                                                  synapse_subtypes[i],
                                                  synapse_params[i],
                                                  buffer_size=int(self.max_population_delay / self.step_size)))
            elif synapse_types[i] == 'DoubleExponentialSynapse':
                self.synapses.append(set_instance(DEDoubleExponentialSynapse,
                                                  synapse_subtypes[i],
                                                  synapse_params[i],
                                                  buffer_size=int(self.max_population_delay / self.step_size)))
            else:
                raise AttributeError('Invalid synapse type!')
            if synapse_labels[i]:
                self.synapses[i].synapse_type = synapse_labels[i]
            self.synapse_labels.append(self.synapses[-1].synapse_type)

    def update(self) -> None:
        """update population attributes.
        """

        # update synapses
        for syn in self.synapses:
            syn.buffer_size = int(self.max_population_delay / self.step_size)
            syn.update()

        # update axon
        if hasattr(self.axon, 'bin_size'):
            self.axon.bin_size = self.step_size
        self.axon.update()

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

        # call super method
        ###################

        super().add_synapse(synapse, synapse_idx)

        # additional state vector updates
        #################################

        self.synaptic_currents_old = np.zeros_like(self.synaptic_currents)
        self.PSPs = np.zeros_like(self.synaptic_currents)

    def clear(self):
        """Clears states stored on population and synaptic input.
        """

        # call super method
        ###################

        super().clear()

        # additional state vector clearing
        ##################################

        self.synaptic_currents_old[:] = 0.
        self.PSPs[:] = self.state_variables[0][0]

    def plot_synaptic_kernels(self, synapse_idx: Optional[List[int]] = None, create_plot: Optional[bool] = True,
                              axes: Axes = None, max_kernel_length: Optional[float] = None) -> object:
        """Creates plot of all specified synapses over time.

        Parameters
        ----------
        synapse_idx
            Index of synapses for which to plot kernel.
        create_plot
            If true, plot will be shown.
        axes
            Can be used to pass figure handle of figure to plot into.
        max_kernel_length
            Maximum length for which to plot synaptic kernel [unit = s].

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
            self.clear()
            synaptic_response = list()
            synaptic_response.append(self.state_variables[-1][0] + self.step_size * self.synaptic_currents[i])
            self.synapses[i].synaptic_input[0] = 1.
            self.state_update()
            synaptic_response.append(self.state_variables[-1][0] + self.step_size * self.synaptic_currents[i])
            if not max_kernel_length:
                while np.abs(synaptic_response[-1]) > 1e-14:
                    self.state_update()
                    synaptic_response.append(self.state_variables[-1][0] + self.step_size * self.synaptic_currents[i])
            else:
                t = 0.
                while t < max_kernel_length:
                    self.state_update()
                    synaptic_response.append(self.state_variables[-1][0] + self.step_size * self.synaptic_currents[i])
                    t += self.step_size
            axes.plot(np.array(synaptic_response))
            synapse_types.append(self.synapses[i].synapse_type)

        plt.legend(synapse_types)

        if create_plot:
            fig.show()

        return axes

######################################
# plastic jansen-rit type population #
######################################


class SecondOrderPlasticPopulationOld(PlasticPopulationOld):
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
    spike_frequency_adaptation
        See docstring of :class:`PlasticPopulation`.
    spike_frequency_adaptation_args
        See docstring of :class:`PlasticPopulation`.
    synapse_efficacy_adaptation
        See docstring of :class:`PlasticPopulation`.
    synapse_efficacy_adaptation_args
        See docstring of :class:`PlasticPopulation`.

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
                 axon_class: str = 'PlasticSigmoidAxon',
                 store_state_variables: bool = True,
                 label: str = 'Custom',
                 spike_frequency_adaptation: Optional[Callable[[float], float]] = None,
                 spike_frequency_adaptation_args: Optional[dict] = None,
                 synapse_efficacy_adaptation: Optional[List[Callable[[float], float]]] = None,
                 synapse_efficacy_adaptation_args: Optional[List[dict]] = None
                 ) -> None:
        """Instantiation of second order population.
        """

        # call super init
        #################

        super().__init__(synapses=synapses,
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
                         spike_frequency_adaptation=spike_frequency_adaptation,
                         spike_frequency_adaptation_args=spike_frequency_adaptation_args,
                         synapse_efficacy_adaptation=synapse_efficacy_adaptation,
                         synapse_efficacy_adaptation_args=synapse_efficacy_adaptation_args)

        # set additional synapse handling vectors
        #########################################

        self.synaptic_currents_old = np.zeros_like(self.synaptic_currents)
        self.PSPs = np.zeros_like(self.synaptic_currents) + init_state

    def _set_synapses(self,
                      synapse_subtypes: Optional[List[str]] = None,
                      synapse_types: Union[str, List[str]] = 'ExponentialSynapse',
                      synapse_params: Optional[List[dict]] = None,
                      max_synaptic_delay: Optional[float] = None,
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
            if synapse_types[i] == 'ExponentialSynapse':
                self.synapses.append(set_instance(DEExponentialSynapse,
                                                  synapse_subtypes[i],
                                                  synapse_params[i],
                                                  buffer_size=int(self.max_population_delay / self.step_size)))
            elif synapse_types[i] == 'DoubleExponentialSynapse':
                self.synapses.append(set_instance(DEDoubleExponentialSynapse,
                                                  synapse_subtypes[i],
                                                  synapse_params[i],
                                                  buffer_size=int(self.max_population_delay / self.step_size)))
            else:
                raise AttributeError('Invalid synapse type!')
            self.synapse_labels.append(self.synapses[-1].synapse_type)

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
        # fixme: this is called with arrays of 1s, although no actual extrinsic synaptic modulation is defined.
        # fixme: --> performance

        # compute average membrane potential
        ####################################

        for i, psp in enumerate(self.PSPs):
            self.PSPs[i] = self.take_step(f=self.get_delta_psp, y_old=self.PSPs[i], synapse_idx=i)
        membrane_potential = np.sum(self.PSPs)
        state_vars = [membrane_potential]

        # compute average firing rate
        #############################

        self.axon.compute_firing_rate(membrane_potential)

        # update state variables
        ########################

        # TODO: Implement observer system here!!!
        self.state_variables.append(state_vars)
        if not self.store_state_variables:
            self.state_variables.pop(0)

        # update axonal transfer function
        #################################

        if self.spike_frequency_adaptation:
            self.axon_update()
            self.state_variables[-1] += [self.axon.transfer_function_args['adaptation']]

        # update synaptic scaling
        #########################

        for i in range(self.n_synapses):

            if self.synapse_efficacy_adaptation[i]:
                self.synapse_update(i)
                self.state_variables[-1] += [self.synapses[i].depression]

    def get_delta_psp(self,
                      psp: FloatLike,
                      synapse_idx: int
                      ) -> FloatLike:
        """Calculates change in membrane potential as function of synaptic current, leak current and
        extrinsic current.

        Parameters
        ----------
        psp
            Current membrane potential of population [unit = V].
        synapse_idx
            Index of synapse for which to calculate the PSP.
        Returns
        -------
        float
            Delta membrane potential [unit = V].

        """

        # update synaptic currents
        ###########################

        # update synaptic currents old
        self.synaptic_currents_old[synapse_idx] = self.synaptic_currents[synapse_idx]

        # calculate synaptic current
        self.synaptic_currents[synapse_idx] = self.take_step(f=self.synapses[synapse_idx].get_delta_synaptic_current,
                                                             y_old=self.synaptic_currents_old[synapse_idx],
                                                             membrane_potential=psp)

        return self.synaptic_currents_old[synapse_idx] * self.extrinsic_synaptic_modulation[synapse_idx] + \
               self.extrinsic_current

    def update(self) -> None:
        """update population attributes.
        """

        # update synapses
        for syn in self.synapses:
            syn.buffer_size = int(self.max_population_delay / self.step_size)
            syn.update()

        # update axon
        if hasattr(self.axon, 'bin_size'):
            self.axon.bin_size = self.step_size
        self.axon.update()

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
        self.synaptic_currents_old = np.zeros_like(self.synaptic_currents)
        self.PSPs = np.zeros_like(self.synaptic_currents)

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
        self.synaptic_currents[:] = 0.
        self.synaptic_currents_old[:] = 0.
        self.PSPs[:] = 0.

        # axon attributes
        self.axon.clear()

    def plot_synaptic_kernels(self, synapse_idx: Optional[List[int]] = None, create_plot: Optional[bool] = True,
                              axes: Axes = None, max_kernel_length: Optional[float] = None) -> object:
        """Creates plot of all specified synapses over time.

        Parameters
        ----------
        synapse_idx
            Index of synapses for which to plot kernel.
        create_plot
            If true, plot will be shown.
        axes
            Can be used to pass figure handle of figure to plot into.
        max_kernel_length
            Maximum length for which to plot synaptic kernel [unit = s].

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
            self.clear()
            synaptic_response = list()
            synaptic_response.append(self.state_variables[-1][0] + self.step_size * self.synaptic_currents[i])
            self.synapses[i].synaptic_input[0] = 1.
            self.state_update()
            synaptic_response.append(self.state_variables[-1][0] + self.step_size * self.synaptic_currents[i])
            if not max_kernel_length:
                while np.abs(synaptic_response[-1]) > 1e-14:
                    self.state_update()
                    synaptic_response.append(self.state_variables[-1][0] + self.step_size * self.synaptic_currents[i])
            else:
                t = 0.
                while t < max_kernel_length:
                    self.state_update()
                    synaptic_response.append(self.state_variables[-1][0] + self.step_size * self.synaptic_currents[i])
                    t += self.step_size
            axes.plot(np.array(synaptic_response))
            synapse_types.append(self.synapses[i].synapse_type)

        plt.legend(synapse_types)

        if create_plot:
            fig.show()

        return axes


######################################################################
# dummy population that passes input to other populations in network #
######################################################################


class DummyPopulation(AbstractBasePopulation):
    """Population object used to pass certain inputs to other populations in a network.
    
    Parameters
    ----------
    output
        Firing rates to be passed through network
    
    """

    def __init__(self, output: np.ndarray):
        """Instantiate dummy population.
        """
        super().__init__()

        self.output = output
        self.idx = 0

    def get_firing_rate(self):
        """Update output firing rate.
        """

        self.idx += 1

        return self.output[self.idx - 1]


################################
# new, all-covering population #
################################


class Population(AbstractBasePopulation):
    """Base neural mass or population class, behaving like a leaky capacitor.

        A population is defined via a number of synapses and an axon.

        Parameters
        ----------
        synapses
            Can be set to use default synapse types. These include:
            :class:`pyrates.synapse.templates.AMPACurrentSynapse`,
            :class:`pyrates.synapse.templates.GABAACurrentSynapse`,
            :class:`pyrates.synapse.templates.AMPAConductanceSynapse`,
            :class:`pyrates.synapse.templates.GABAAConductanceSynapse`.
        axon
            Can be set to use default axon types. These include:
            :class:`pyrates.axon.templates.JansenRitAxon`.
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
                 synapse_params: Optional[List[dict]] = None,
                 axon_params: Optional[Dict[str, float]] = None,
                 synapse_class: Union[str, List[str]] = 'DoubleExponentialSynapse',
                 axon_class: str = 'SigmoidAxon',
                 init_state: Optional[FloatLike] = None,
                 step_size: float = 0.0001,
                 tau_leak: Optional[float] = None,
                 resting_potential: float = -0.075,
                 membrane_capacitance: float = 1e-12,
                 max_population_delay: FloatLike = 0.,
                 max_synaptic_delay: Optional[Union[float, np.ndarray]] = None,
                 store_state_variables: bool = True,
                 label: str = 'Custom',
                 synapse_labels: Optional[list] = None,
                 enable_modulation: bool = False,
                 spike_frequency_adaptation: Optional[Callable[[float], float]] = None,
                 spike_frequency_adaptation_args: Optional[dict] = None,
                 synapse_efficacy_adaptation: Optional[List[Callable[[float], float]]] = None,
                 synapse_efficacy_adaptation_args: Optional[List[dict]] = None
                 ) -> None:
        """Instantiation of base population.
        """

        # call super class init to establish network functionality
        ##########################################################

        super().__init__()

        # check input parameters
        ########################

        # check synapse/axon attributes
        if not synapses and not synapse_params:
            raise AttributeError('Either synapses or synapse_params have to be passed!')
        if not axon and not axon_params:
            raise AttributeError('Either axon or axon_params have to be passed!')

        # check attribute values
        if step_size < 0 or max_population_delay < 0 or (tau_leak and tau_leak < 0):
            raise ValueError('Time constants (tau, step_size, max_delay) cannot be negative. '
                             'See parameter docstring for further information.')
        if membrane_capacitance and membrane_capacitance < 0:
            raise ValueError('Membrane capacitance cannot be negative. See docstring for further information.')

        if spike_frequency_adaptation and not spike_frequency_adaptation_args:
            raise ValueError('Arguments for spike frequency adaptation mechanism have to be passed!')

        if synapse_efficacy_adaptation and not synapse_efficacy_adaptation_args:
            raise ValueError('Arguments for synaptic efficacy adaptation mechanism have to be passed!')

        # set general population params
        ###############################

        self.synapses = []  # type: List[Union[Synapse, DESynapse]]
        self.state_variables = []  # type: List[List[FloatLike]]
        self.store_state_variables = store_state_variables
        self.step_size = step_size
        self.label = label
        self.max_population_delay = max_population_delay
        self.n_synapses = len(synapses) if synapses else len(synapse_params)
        self.synaptic_currents = np.zeros(self.n_synapses)
        self.extrinsic_current = 0.
        self.spike_frequency_adaptation = spike_frequency_adaptation
        self.synapse_efficacy_adaptation = synapse_efficacy_adaptation

        # choose between different population set-ups based on passed parameters
        ########################################################################

        # leaky capacitor attributes
        if tau_leak:
            self.tau_leak = tau_leak
            self.membrane_capacitance = membrane_capacitance
            self.resting_potential = resting_potential
            self.leaky_capacitor = True
        else:
            self.leaky_capacitor = False

        # extrinsic modulation attributes
        if enable_modulation:
            self.extrinsic_synaptic_modulation = np.ones(self.n_synapses)
            self.modulatory_influences_enabled = True
        else:
            self.modulatory_influences_enabled = False

        # synaptic plasticity
        if self.synapse_efficacy_adaptation:
            self.synaptic_plasticity = True
        else:
            self.synaptic_plasticity = False

        # axonal plasticity
        if self.spike_frequency_adaptation:
            self.axonal_plasticity = True
        else:
            self.axonal_plasticity = False

        # integro-differential vs. purely differential
        if type(synapse_class) is list:
            n_differential_syns = 0
            for syn in synapse_class:
                if syn == 'ExponentialSynapse':
                    n_differential_syns += 1
            if n_differential_syns == len(synapse_class):
                self.integro_differential = False
            else:
                self.integro_differential = True
        else:
            if synapse_class == 'ExponentialSynapse':
                self.integro_differential = False
            else:
                self.integro_differential = True

        # set population dependencies
        if not init_state:
            init_state = resting_potential if resting_potential else 0.

        self.PSPs = np.zeros_like(self.synaptic_currents)
        if self.leaky_capacitor:
            self.leak_current = 0.
        else:
            self.PSPs += init_state
        if not self.integro_differential:
            self.synaptic_currents_new = np.zeros_like(self.synaptic_currents)
            self.PSPs_new = self.PSPs.copy()

        # build state update function
        #############################

        exec(construct_state_update_function(spike_frequency_adaptation=self.axonal_plasticity,
                                             synapse_efficacy_adaptation=self.synaptic_plasticity,
                                             enable_modulation=self.modulatory_influences_enabled,
                                             leaky_capacitor=self.leaky_capacitor,
                                             store_state_variables=store_state_variables))

        self.state_update = MethodType(locals()['state_update'], self)

        # build delta membrane potential function
        #########################################

        exec(construct_get_delta_membrane_potential_function(leaky_capacitor=self.leaky_capacitor,
                                                             enable_modulation=self.modulatory_influences_enabled,
                                                             integro_differential=self.integro_differential))

        self.get_delta_membrane_potential = MethodType(locals()['get_delta_membrane_potential'], self)

        # set synapses
        ##############

        # make sure max_synaptic_delay has the correct format
        if max_synaptic_delay is None:
            max_synaptic_delay = np.array(check_nones(max_synaptic_delay, self.n_synapses))
        elif not isinstance(max_synaptic_delay, np.ndarray):
            max_synaptic_delay = np.zeros(self.n_synapses) + max_synaptic_delay

        # instantiate synapses
        self._set_synapses(synapse_subtypes=synapses,
                           synapse_params=synapse_params,
                           synapse_types=synapse_class,
                           max_synaptic_delay=max_synaptic_delay,
                           synapse_labels=synapse_labels)

        # set axon
        ##########

        self._set_axon(axon, axon_params=axon_params, axon_type=axon_class)

        # set initial states
        if type(init_state) not in (list, np.ndarray):
            self.state_variables.append([init_state])
        else:
            self.state_variables.append(init_state)

        self.axon.compute_firing_rate(self.state_variables[-1][0])

        # set plasticity attributes
        ###########################

        # for axon
        if self.spike_frequency_adaptation:
            self.spike_frequency_adaptation_args = spike_frequency_adaptation_args if spike_frequency_adaptation_args \
                else dict()
            self.state_variables[-1] += [self.axon.transfer_function_args['adaptation']]

        # synaptic plasticity function
        if synapse_efficacy_adaptation and (type(synapse_efficacy_adaptation) is not list):
            self.synapse_efficacy_adaptation = [synapse_efficacy_adaptation for _ in range(self.n_synapses)]
        elif synapse_efficacy_adaptation and (len(synapse_efficacy_adaptation) != self.n_synapses):
            raise ValueError('If list of synaptic plasticity functions is passed, its length must correspond to the'
                             'number of synapses of the population.')
        elif not synapse_efficacy_adaptation:
            self.synapse_efficacy_adaptation = [None for _ in range(self.n_synapses)]
        else:
            self.synapse_efficacy_adaptation = synapse_efficacy_adaptation

        # synaptic plasticity state variables
        for i in range(self.n_synapses):
            if self.synapse_efficacy_adaptation[i]:
                self.state_variables[-1] += [self.synapses[i].depression]

        # synaptic plasticity function arguments
        if synapse_efficacy_adaptation_args is None or type(synapse_efficacy_adaptation_args) is dict:
            self.synapse_efficacy_adaptation_args = [synapse_efficacy_adaptation_args for _ in
                                                     range(self.n_synapses)]
        else:
            self.synapse_efficacy_adaptation_args = synapse_efficacy_adaptation_args

        if len(self.synapse_efficacy_adaptation) != len(self.synapse_efficacy_adaptation_args):
            raise AttributeError('Number of synaptic plasticity functions and plasticity function parameter '
                                 'dictionaries has to be equal.')

    def _set_synapses(self,
                      synapse_subtypes: Optional[List[str]] = None,
                      synapse_types: Union[str, List[str]] = 'DoubleExponentialSynapse',
                      synapse_params: Optional[List[dict]] = None,
                      max_synaptic_delay: Optional[np.ndarray] = None,
                      synapse_labels: Optional[List[str]] = None
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
        synapse_labels

        """

        # check synapse parameter formats
        #################################

        if isinstance(synapse_types, str):
            synapse_types = [synapse_types for _ in range(self.n_synapses)]

        synapse_subtypes = check_nones(synapse_subtypes, self.n_synapses)
        synapse_params = check_nones(synapse_params, self.n_synapses)
        synapse_labels = check_nones(synapse_labels, self.n_synapses)
        self.synapse_labels = list()

        # set all given synapses
        ########################

        for i in range(self.n_synapses):

            # instantiate synapse
            if synapse_types[i] == 'DoubleExponentialSynapse':
                if self.integro_differential:
                    self.synapses.append(set_instance(DoubleExponentialSynapse,
                                                      synapse_subtypes[i],
                                                      synapse_params[i],
                                                      bin_size=self.step_size,
                                                      max_delay=max_synaptic_delay[i],
                                                      buffer_size=self.max_population_delay))
                else:
                    self.synapses.append(set_instance(DEDoubleExponentialSynapse,
                                                      synapse_subtypes[i],
                                                      synapse_params[i],
                                                      buffer_size=int(self.max_population_delay / self.step_size)))
            elif synapse_types[i] == 'ExponentialSynapse':
                if self.integro_differential:
                    self.synapses.append(set_instance(ExponentialSynapse,
                                                      synapse_subtypes[i],
                                                      synapse_params[i],
                                                      bin_size=self.step_size,
                                                      max_delay=max_synaptic_delay[i],
                                                      buffer_size=self.max_population_delay))
                else:
                    self.synapses.append(set_instance(DEExponentialSynapse,
                                                      synapse_subtypes[i],
                                                      synapse_params[i],
                                                      buffer_size=int(self.max_population_delay / self.step_size)))
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
            if synapse_labels[i]:
                self.synapses[i].synapse_type = synapse_labels[i]
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
            self.axon = set_instance(SigmoidAxon, axon_subtype, axon_params)  # type: Union[Axon, BurstingAxon]
        elif axon_type == 'BurstingAxon':
            self.axon = set_instance(BurstingAxon, axon_subtype, axon_params)  # type: Union[Axon, BurstingAxon]
        elif axon_type == 'Axon':
            self.axon = set_instance(Axon, axon_subtype, axon_params)  # type: Union[Axon, BurstingAxon]
        elif axon_type == 'PlasticSigmoidAxon':
            self.axon = set_instance(PlasticSigmoidAxon, axon_subtype, axon_params)  # type: Union[Axon, BurstingAxon]
        else:
            raise AttributeError('Invalid axon type!')

    def get_firing_rate(self) -> FloatLike:
        """Calculate the current average firing rate of the population.

        Returns
        -------
        float
            Average firing rate of population [unit = 1/s].

        """

        return self.axon.firing_rate

    def get_delta_psp(self, psp: FloatLike, synapse_idx: int) -> FloatLike:
        """Calculates the change in PSP.
        """

        return self.synaptic_currents[synapse_idx]

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

    def copy_synapse(self, synapse_idx: int, **kwargs) -> None:
        """Copies an existing synapse

        Parameters
        ----------
        synapse_idx
            Index of synapse to copy (default = None).

        """

        # copy synapse
        synapse = deepcopy(self.synapses[synapse_idx])

        # add copy to synapse list
        self.add_synapse(synapse, synapse_idx, **kwargs)

    def add_synapse(self,
                    synapse: Synapse,
                    synapse_idx: Optional[int] = None,
                    max_firing_rate: Optional[float] = None
                    ) -> None:
        """Adds copy of specified synapse to population.

        Parameters
        ----------
        synapse
            Synapse object to add (default = None)
        synapse_idx
            Index of synapse to copy (default = None).
        max_firing_rate
            Maximum firing rate of pre-synapse (only needed for plastic synapses, default = None).

        """

        # add synapse
        self.synapses.append(synapse)

        # update synapse dependencies
        #############################

        self.n_synapses = len(self.synapses)

        # current vector
        self.synaptic_currents = np.zeros(self.n_synapses)
        self.PSPs = np.zeros(self.n_synapses)
        if not self.integro_differential:
            self.synaptic_currents_new = np.zeros_like(self.synaptic_currents)
            self.PSPs_new = np.zeros_like(self.synaptic_currents)

        # extrinsic modulation
        ######################

        if self.modulatory_influences_enabled:
            ext_syn_mod = self.extrinsic_synaptic_modulation
            self.extrinsic_synaptic_modulation = np.ones(self.n_synapses)
            if synapse_idx:
                self.extrinsic_synaptic_modulation[0:-1] = ext_syn_mod
                self.extrinsic_synaptic_modulation[-1] = ext_syn_mod[synapse_idx]

        # check plasticity related stuff
        ################################

        if self.synaptic_plasticity and self.synapse_efficacy_adaptation[synapse_idx]:

            if max_firing_rate is None:
                max_firing_rate = self.axon.transfer_function_args['max_firing_rate']

            self.synapse_efficacy_adaptation.append(self.synapse_efficacy_adaptation[synapse_idx])
            self.synapse_efficacy_adaptation_args.append(self.synapse_efficacy_adaptation_args[synapse_idx])
            self.synapse_efficacy_adaptation_args[-1]['max_firing_rate'] = max_firing_rate
            self.state_variables[-1] += [self.synapses[synapse_idx].depression]

    def clear(self, disconnect: bool = False):
        """Clears states stored on population and synaptic input.
        """

        # population attributes
        init_state = self.state_variables[0]
        self.state_variables.clear()
        self.state_variables.append(init_state)

        # synapse attributes
        for syn in self.synapses:
            syn.clear()
        self.synaptic_currents[:] = 0.
        self.PSPs[:] = 0.
        if not self.integro_differential:
            self.synaptic_currents_new[:] = 0.
            self.PSPs[:] = self.state_variables[0][0]

        # axon attributes
        self.axon.clear()
        self.axon.compute_firing_rate(self.state_variables[-1][0])

        # network connections
        if disconnect:
            self.disconnect()

    def update(self) -> None:
        """update population attributes.
        """

        # update synapses
        for syn in self.synapses:
            syn.bin_size = self.step_size
            if not self.integro_differential:
                syn.buffer_size = int(self.max_population_delay / self.step_size)
            else:
                syn.buffer_size = self.max_population_delay
            syn.update()

        # update axon
        if hasattr(self.axon, 'bin_size'):
            self.axon.bin_size = self.step_size
        self.axon.update()

    def axon_update(self):
        """Updates adaptation field of axon.
        """

        self.axon.transfer_function_args['adaptation'] = self.take_step(f=self.spike_frequency_adaptation,
                                                                        y_old=self.axon.transfer_function_args
                                                                        ['adaptation'],
                                                                        firing_rate=self.get_firing_rate(),
                                                                        **self.spike_frequency_adaptation_args)

    def synapse_update(self, idx: int):
        """Updates depression field of synapse.

        Parameters
        ----------
        idx
            Synapse index.

        """

        self.synapses[idx].depression = self.take_step(f=self.synapse_efficacy_adaptation[idx],
                                                       y_old=self.synapses[idx].depression,
                                                       firing_rate=self.synapses[idx].synaptic_input[
                                                           self.synapses[idx].kernel_length - 1],
                                                       **self.synapse_efficacy_adaptation_args[idx])

    def plot_synaptic_kernels(self, synapse_idx: Optional[List[int]] = None, create_plot: Optional[bool] = True,
                              axes: Axes = None) -> object:
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
