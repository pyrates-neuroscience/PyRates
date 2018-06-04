"""Module that includes all-covering population class plus dummy population classes for network information flow.

A population is defined as the basic computational unit in the neural mass model. It contains various synapses
plus an axon hillock. There are various options and formalisms for the population state-update equations, which are all
covered within a single population class. For a more detailed description, see the docstring of the Population class

"""

# external packages
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from matplotlib.axes import Axes
from typing import List, Optional, Union, Dict, Callable, TypeVar
import tensorflow as tf

# pyrates internal imports
from pyrates.axon import Axon, SigmoidAxon, BurstingAxon, PlasticSigmoidAxon
from pyrates.synapse import Synapse, DoubleExponentialSynapse, ExponentialSynapse, TransformedInputSynapse, \
    DEExponentialSynapse, DEDoubleExponentialSynapse
from pyrates.utility import set_instance, make_iterable
from pyrates.utility.filestorage import RepresentationBase

# type definition
FloatLike = Union[float, np.float64]
AxonLike = TypeVar('AxonLike', bound=Axon, covariant=True)
SynapseLike = TypeVar('SynapseLike', bound=Synapse, covariant=True)

# meta infos
__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


##############################
# leaky capacitor population #
##############################


class AbstractBasePopulation(RepresentationBase):
    """Very base class used as parent class for all nodes within a circuit graph.

    Attributes
    ----------
    targets: List[object]
        List of target objects the population connects to.
    target_weights: List[float]
        List with scaling factors for each connection.
    target_delays: List[int]
        List with absolute delays [unit = time-steps] for each connection.
    firing_rate: float
        Current output firing rate of node [unit = 1/s].

    """

    def __init__(self,
                 tf_graph: Optional[tf.Graph] = None):
        """Instantiates basic circuit node.
        """

        self.tf_graph = tf.get_default_graph() if tf_graph is None \
            else tf_graph
        self.targets = []
        self.target_weights = []
        self.target_delays = []

    def connect(self,
                target: object,
                weight: float,
                delay: int
                ):
        """Connect to a given target object (population or synapse) with a given weight and delay.

        Parameters
        ----------
        target
            Synapse or population that population is supposed to connect to
        weight
            Weight that is to be applied to the connection (=connectivity)
        delay
            Index (=time delay) at which the output is supposed to arrive at the target synapse"""

        self.targets.append(target)
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
        """

        raise AttributeError('This method needs to be implemented at the child class level.')


#####################################################################
# dummy populations that pass input to other populations in network #
#####################################################################


class SynapticInputPopulation(AbstractBasePopulation):
    """Population object used to pass certain inputs to other populations in a network.
    
    Parameters
    ----------
    output
        Firing rates to be passed through network
    key
        Name of network node (default = None).

    """

    def __init__(self,
                 output: np.ndarray,
                 tf_graph: Optional[tf.Graph] = None,
                 key: Optional[str] = None
                 ):
        """Instantiate dummy population for synaptic input passage.
        """

        super().__init__(tf_graph=tf_graph)

        if len(self.targets) > 1:
            raise ValueError('Synaptic input dummy populations can only connect to a single target. Use multiple dummy'
                             ' populations to connect to multiple targets.')

        self.key = key if key else 'dummy_synaptic_input'

        # tensorflow stuff
        ##################

        with self.tf_graph.as_default():

            with tf.variable_scope(self.key):

                self.output = tf.get_variable(shape=(),
                                              name=self.key + '_output',
                                              trainable=False,
                                              dtype=tf.float32,
                                              initializer=tf.constant_initializer(value=0., dtype=tf.float32)
                                              )
                self.output_buffer = tf.Variable(output,
                                                 name=self.key + '_output_buffer',
                                                 trainable=False,
                                                 dtype=tf.float32)

                self.idx = tf.get_variable(name=self.key + '_idx',
                                           shape=(),
                                           dtype=tf.int32,
                                           initializer=tf.constant_initializer(value=0, dtype=tf.int32)
                                           )

                increment = self.idx.assign_add(1)
                with tf.control_dependencies([increment]):
                    update_output = self.output.assign(self.output_buffer[self.idx])
                self.state_update = tf.group(increment, update_output, name=self.key + '_state_update')

    def connect_to_targets(self):
        """Project output to pass synapse function of target.
        """

        return self.targets[0].synaptic_buffer[0].assign(self.targets[0].synaptic_buffer[0] + self.output) \
            if len(self.targets) > 0 else tf.no_op()


class ExtrinsicCurrentPopulation(AbstractBasePopulation):
    """Population object used to pass certain inputs to other populations in a network.

    Parameters
    ----------
    output
        Extrinsic current to be passed through network
    key
        Name of node in graph.

    """

    def __init__(self,
                 output: np.ndarray,
                 key: Optional[str] = None
                 ):
        """Instantiate dummy population that passes extrinsic current to a target.
        """

        super().__init__()

        if len(self.targets) > 1:
            raise ValueError('Extrinsic current dummy populations can only connect to a single target. Use multiple '
                             'dummy populations to connect to multiple targets.')

        self.output = output
        self.idx = 0
        self.key = key if key else 'dummy_extrinsic_current'

    def project_to_targets(self):
        """Pass extrinsic current to target.
        """

        self.targets[0].extrinsic_current = self.output[self.idx]

    def state_update(self):
        """Advance in output array by one position.
        """

        self.idx += 1


class ExtrinsicModulationPopulation(AbstractBasePopulation):
    """Population object used to pass certain inputs to other populations in a network.

    Parameters
    ----------
    output
        Extrinsic modulation to be passed through network
    key
        Name of node in network graph.

    """

    def __init__(self,
                 output: np.ndarray,
                 key: Optional[str] = None
                 ):
        """Instantiate dummy population that modulates the synaptic activity of a target.
        """

        super().__init__()

        if len(self.targets) > 1:
            raise ValueError('Extrinsic modulation dummy populations can only connect to a single target. Use multiple '
                             'dummy populations to connect to multiple targets.')

        self.output = output
        self.idx = 0
        self.key = key if key else 'dummy_extrinsic_modulation'

    def project_to_targets(self):
        """Project extrinsic synaptic modulation to target.
        """

        self.targets[0].extrinsic_synaptic_modulation = self.output[self.idx]

    def state_update(self):
        """Advance in output array by one position.
        """

        self.idx += 1


################################
# new, all-covering population #
################################


class Population(AbstractBasePopulation):
    """Base neural mass or population class, defined via a number of synapses and an axon.

        Parameters
        ----------
        synapses
            Can be passed to use default synapse types. These include:
                :class:`pyrates.synapse.templates.AMPACurrentSynapse`,
                :class:`pyrates.synapse.templates.GABAACurrentSynapse`,
                :class:`pyrates.synapse.templates.AMPAConductanceSynapse`,
                :class:`pyrates.synapse.templates.GABAAConductanceSynapse`
                :class:`pyrates.synapse.templates.JansenRitExcitatorySynapse`
                :class:`pyrates.synapse.templates.JansenRitInhibitorySynapse`.
        axon
            Can be passed to use default axon types. These include:
                :class:`pyrates.axon.templates.JansenRitAxon`
                :class:`pyrates.axon.templates.KnoescheAxon`
                :class:`pyrates.axon.templates.PlasticKnoescheAxon`
                :class:`pyrates.axon.templates.SuffczynskiAxon`
                :class:`pyrates.axon.templates.MoranAxon`.
        init_state
            Dictionary or scalar defining the initial state of the population (default = None).
            If a scalar is passed, it will be used as the initial membrane potential of the population.
            If a dictionary is passed, the following key-value pairs can be used:
                1) 'membrane_potential': scalar [unit = V].
                2) 'firing_rate': scalar [unit = 1/s].
                3) 'PSPs': scalar array with length equal to number of synapses [unit = V].
                4) 'synaptic_currents': scalar array with length equal to number of synapses [unit = A].
                5) 'extrinsic_current': scalar [unit = V or A].
                6) 'extrinsic_synaptic_modulation': scalar array with length equal to number of synapses [unit = 1].
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
            documentation of respective synapse class (default = None).
        axon_params
            Parameters for custom axon type. For parameter explanation see documentation of respective axon type
            (default = False).
        key
            If passed, will be used as (unique) identifier of the population.

        Attributes
        ----------
        synapses : :obj:`list` of :class:`Synapse` instances
            Synapse instances. See documentation of :class:`Synapse`.
        axon : :class:`Axon` instance
            Axon instance. See documentation of :class:`Axon`.
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
        resting_potential
            See documentation of parameter `resting_potential`.
        tau_leak
            See documentation of parameter `tau_leak`.
        step_size
            See documentation of parameter `step_size`.
        membrane_capacitance
            See documentation of parameter `membrane_capacitance`.
        key
            See documentation of parameter 'key'.
        verbose
            If true, a summary of the population features will be displayed during the initialization.

        """

    def __init__(self,
                 synapses: Optional[List[str]] = None,
                 axon: Optional[str] = None,
                 synapse_params: Optional[List[dict]] = None,
                 axon_params: Optional[Dict[str, float]] = None,
                 synapse_class: Union[str, List[str]] = 'ExponentialSynapse',
                 axon_class: str = 'SigmoidAxon',
                 init_state: Optional[Union[float, dict]] = None,
                 step_size: float = 1e-4,
                 tau_leak: Optional[float] = None,
                 resting_potential: float = -0.075,
                 membrane_capacitance: float = 1e-12,
                 max_population_delay: FloatLike = 0.,
                 max_synaptic_delay: Optional[Union[float, np.ndarray]] = None,
                 enable_modulation: bool = False,
                 spike_frequency_adaptation: Optional[Callable[[float], float]] = None,
                 spike_frequency_adaptation_kwargs: Optional[dict] = None,
                 synapse_efficacy_adaptation: Optional[List[Callable[[float], float]]] = None,
                 synapse_efficacy_adaptation_kwargs: Optional[List[dict]] = None,
                 tf_graph: Optional[tf.Graph] = None,
                 key: Optional[str] = None,
                 verbose: bool = False
                 ) -> None:
        """Instantiation of base population.
        """

        # call super class init to establish network functionality
        ##########################################################

        super().__init__(tf_graph)

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

        if spike_frequency_adaptation and not spike_frequency_adaptation_kwargs:
            raise ValueError('Arguments for spike frequency adaptation mechanism have to be passed!')

        if synapse_efficacy_adaptation and not synapse_efficacy_adaptation_kwargs:
            raise ValueError('Arguments for synaptic efficacy adaptation mechanism have to be passed!')

        # set general population params
        ###############################

        self.synapses = dict()
        self.step_size = step_size
        self.key = key if key else 'nokey'
        self.max_population_delay = max_population_delay
        self.n_synapses = len(synapses) if synapses else len(synapse_params)
        self.spike_frequency_adaptation = deepcopy(spike_frequency_adaptation)
        self.synapse_efficacy_adaptation = deepcopy(synapse_efficacy_adaptation)  # type: List[Callable]
        self.synapse_efficacy_adaptation_args = deepcopy(synapse_efficacy_adaptation_kwargs)  # type: List[dict]
        self.features = dict()
        self.project_to_targets = tf.no_op()

        # choose between different population set-ups based on passed parameters
        ########################################################################

        # leaky capacitor attributes
        if tau_leak:
            self.tau_leak = tau_leak
            self.membrane_capacitance = membrane_capacitance
            self.resting_potential = resting_potential
            self.features['leaky_capacitor'] = True
        else:
            self.features['leaky_capacitor'] = False

        # extrinsic modulation attributes
        if enable_modulation:
            self.features['modulation_enabled'] = True
        else:
            self.features['modulation_enabled'] = False

        # synaptic plasticity
        if self.synapse_efficacy_adaptation:
            self.features['synaptic_plasticity'] = True
        else:
            self.features['synaptic_plasticity'] = False

        # axonal plasticity
        if self.spike_frequency_adaptation:
            self.features['axonal_plasticity'] = True
        else:
            self.features['axonal_plasticity'] = False

        # integro-differential vs. purely differential
        if type(synapse_class) is list:
            n_differential_syns = 0
            for syn in synapse_class:
                if syn == 'ExponentialSynapse':
                    n_differential_syns += 1
            if n_differential_syns == len(synapse_class):
                self.features['integro_differential'] = False
            else:
                self.features['integro_differential'] = True
        else:
            if synapse_class == 'ExponentialSynapse':
                self.features['integro_differential'] = False
            else:
                self.features['integro_differential'] = True

        # display the features of the population
        if self.features['leaky_capacitor']:
            lc_snippet = 'The membrane potential dynamics will be described via the leaky-capacitor formalism.'
        else:
            lc_snippet = 'The membrane potential dynamics will be described by convolutions of the synaptic kernels ' \
                         'with their respective input.'
        if self.features['integro_differential']:
            kernel_snippet = 'The synaptic kernels will be evaluated numerically.'
        else:
            kernel_snippet = 'The synaptic kernel convolutions will be evaluated analytically.'
        if self.features['synaptic_plasticity'] and self.features['axonal_plasticity']:
            plasticity_snippet = 'Both synaptic and axonal plasticity will be enabled.'
        elif self.features['synaptic_plasticity']:
            plasticity_snippet = 'Synaptic plasticity will be enabled.'
        elif self.features['axonal_plasticity']:
            plasticity_snippet = 'Axonal plasticity will be enabled.'
        else:
            plasticity_snippet = 'Short-term plasticity mechanisms will be disabled.'
        if self.features['modulation_enabled']:
            modulation_snippet = 'Extrinsic synaptic modulation will be enabled.'
        else:
            modulation_snippet = 'Extrinsic synaptic modulation will be disabled.'

        setup_msg = f"""Given the passed arguments, the population {key} will be initialized with the following 
        features:
                a) {lc_snippet}
                b) {kernel_snippet}
                c) {plasticity_snippet}
                d) {modulation_snippet}"""
        if verbose:
            print(setup_msg)
            print('\n')

        # set synapses
        ##############

        # make sure max_synaptic_delay has the correct format
        max_synaptic_delay = make_iterable(max_synaptic_delay, self.n_synapses)

        # instantiate synapses
        self._set_synapses(synapse_subtypes=synapses,
                           synapse_params=synapse_params,
                           synapse_types=synapse_class,
                           max_synaptic_delay=max_synaptic_delay)

        self.synapse_keys = list(self.synapses.keys())

        # check whether synapse parameters fit with state-update formalism
        for _, syn in self.synapses.items():
            if self.features['leaky_capacitor'] and abs(syn.efficacy) > 1e-7:
                raise Warning('The synaptic efficacy value does not seem to reflect a proper synaptic current.'
                              'Consider to either change the synaptic parameters or turn of the leaky capacitor'
                              'description of the populations membrane potential change by setting tau_leak = None.')
            elif not self.features['leaky_capacitor'] and abs(syn.efficacy) < 1e-7:
                raise Warning('The synaptic efficacy value does not seem to reflect a proper membrane potential.'
                              'Consider to either change the synaptic parameters or turn on the leaky capacitor'
                              'description of the populations membrane potential change by passing appropriate '
                              'parameters for tau_leak, membrane_capacitance and resting_potential.')

        # set axon
        ##########

        self._set_axon(axon, axon_params=axon_params, axon_type=axon_class)

        # set up DE system for state updates
        ####################################

        self.state_update = self._set_state_update_equations()

    def _set_synapses(self,
                      synapse_subtypes: Optional[List[str]] = None,
                      synapse_types: Union[str, List[str]] = 'DoubleExponentialSynapse',
                      synapse_params: Optional[List[dict]] = None,
                      max_synaptic_delay: Optional[Union[np.ndarray, list]] = None
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

        synapse_types = make_iterable(synapse_types, self.n_synapses)
        synapse_subtypes = make_iterable(synapse_subtypes, self.n_synapses)
        synapse_params = make_iterable(synapse_params, self.n_synapses)

        # set all given synapses
        ########################

        with tf.variable_scope(self.key):

            for i in range(self.n_synapses):

                synapse_key = synapse_subtypes[i] if synapse_subtypes[i] else 'synapse' + str(i)

                # instantiate synapse
                if synapse_types[i] == 'DoubleExponentialSynapse':
                    if self.features['integro_differential']:
                        synapse_instance = set_instance(class_handle=DoubleExponentialSynapse,
                                                        instance_type=synapse_subtypes[i],
                                                        instance_params=synapse_params[i],
                                                        step_size=self.step_size,
                                                        max_delay=max_synaptic_delay[i],
                                                        buffer_size=self.max_population_delay,
                                                        tf_graph=self.tf_graph,
                                                        key=self.key + '_' + synapse_key)
                    else:
                        synapse_instance = set_instance(class_handle=DEDoubleExponentialSynapse,
                                                        instance_type=synapse_subtypes[i],
                                                        instance_params=synapse_params[i],
                                                        step_size=self.step_size,
                                                        buffer_size=int(self.max_population_delay / self.step_size),
                                                        tf_graph=self.tf_graph,
                                                        key=self.key + '_' + synapse_key)
                elif synapse_types[i] == 'ExponentialSynapse':
                    if self.features['integro_differential']:
                        synapse_instance = set_instance(class_handle=ExponentialSynapse,
                                                        instance_type=synapse_subtypes[i],
                                                        instance_params=synapse_params[i],
                                                        step_size=self.step_size,
                                                        max_delay=max_synaptic_delay[i],
                                                        buffer_size=self.max_population_delay,
                                                        tf_graph=self.tf_graph,
                                                        key=self.key + '_' + synapse_key)
                    else:
                        synapse_instance = set_instance(class_handle=DEExponentialSynapse,
                                                        instance_type=synapse_subtypes[i],
                                                        instance_params=synapse_params[i],
                                                        step_size=self.step_size,
                                                        buffer_size=int(self.max_population_delay / self.step_size),
                                                        tf_graph=self.tf_graph,
                                                        key=self.key + '_' + synapse_key)
                elif synapse_types[i] == 'TransformedInputSynapse':
                    synapse_instance = set_instance(class_handle=TransformedInputSynapse,
                                                    instance_type=synapse_subtypes[i],
                                                    instance_params=synapse_params[i],
                                                    step_size=self.step_size,
                                                    max_delay=max_synaptic_delay[i],
                                                    buffer_size=self.max_population_delay,
                                                    tf_graph=self.tf_graph,
                                                    key=self.key + '_' + synapse_key)
                elif synapse_types[i] == 'Synapse':
                    synapse_instance = set_instance(class_handle=Synapse,
                                                    instance_type=synapse_subtypes[i],
                                                    instance_params=synapse_params[i],
                                                    bin_size=self.step_size,
                                                    max_delay=max_synaptic_delay[i],
                                                    buffer_size=self.max_population_delay,
                                                    tf_graph=self.tf_graph,
                                                    key=self.key + '_' + synapse_key)
                else:
                    raise AttributeError('Invalid synapse type!')

                # name synapse according to the instance's key
                self.synapses[synapse_instance.key] = synapse_instance

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

        with tf.variable_scope(self.key):

            axon_key = axon_subtype if axon_subtype is not None else 'axon'

            if axon_type == 'SigmoidAxon':
                self.axon = set_instance(class_handle=SigmoidAxon,
                                         instance_type=axon_subtype,
                                         instance_params=axon_params,
                                         tf_graph=self.tf_graph,
                                         key=self.key + '_' + axon_key)  # type: Union[Axon, BurstingAxon]
            elif axon_type == 'BurstingAxon':
                self.axon = set_instance(class_handle=BurstingAxon,
                                         instance_type=axon_subtype,
                                         instance_params=axon_params,
                                         tf_graph=self.tf_graph,
                                         key=self.key + '_' + axon_key)  # type: Union[Axon, BurstingAxon]
            elif axon_type == 'Axon':
                self.axon = set_instance(class_handle=Axon,
                                         instance_type=axon_subtype,
                                         instance_params=axon_params,
                                         tf_graph=self.tf_graph,
                                         key=self.key + '_' + axon_key)  # type: Union[Axon, BurstingAxon]
            elif axon_type == 'PlasticSigmoidAxon':
                self.axon = set_instance(class_handle=PlasticSigmoidAxon,
                                         instance_type=axon_subtype,
                                         instance_params=axon_params,
                                         tf_graph=self.tf_graph,
                                         key=self.key + '_' + axon_key)  # type: Union[Axon, BurstingAxon]
            else:
                raise AttributeError('Invalid axon type!')

    def _set_state_update_equations(self):
        """Creates a system of ODEs that describes the total state update.
        """

        with self.tf_graph.as_default():

            with tf.variable_scope(self.key):

                self.membrane_potential = tf.get_variable(name=self.key + '_membrane_potential',
                                                          shape=(),
                                                          dtype=tf.float32,
                                                          initializer=tf.constant_initializer(value=0.)
                                                          )

                self.psps = tf.Variable(np.zeros(self.n_synapses, dtype=np.float32),
                                        name=self.key + '_psps',
                                        trainable=False
                                        )

                # rate-to-potential operator: synapse level
                ###########################################

                # update synaptic states
                syn_updates = []
                for i, syn in enumerate(self.synapses.values()):

                    update_synapse = tf.group(syn.update_psp,
                                              syn.update_synaptic_response,
                                              name=self.key + '_update_synapse_' + str(i))

                    with tf.control_dependencies([update_synapse]):
                        get_psp = self.psps[i].assign(syn.psp)
                        with tf.control_dependencies([get_psp]):
                            update_buffer_1 = syn.update_buffer_1
                            with tf.control_dependencies([update_buffer_1]):
                                update_buffer_2 = syn.update_buffer_2
                            update_buffer = tf.group(update_buffer_1, update_buffer_2,
                                                     name=self.key + '_update_synapse_buffer_' + str(i))

                    syn_updates.append(tf.group(update_synapse, get_psp, update_buffer))

                update_all_synapses = tf.tuple(syn_updates, name=self.key + '_update_all_synapses')

                # rate-to-potential operator: soma level
                ########################################

                # update membrane potential
                with tf.control_dependencies(update_all_synapses):
                    update_membrane_potential = self.membrane_potential.assign(tf.reduce_sum(self.psps))

                # group rtp operations
                rate_to_potential = tf.group(update_all_synapses, update_membrane_potential,
                                             name=self.key + '_rpo')

                # potential-to-rate operator
                ############################

                with tf.control_dependencies([rate_to_potential]):
                    potential_to_rate = self.axon.firing_rate.assign(
                        self.axon.transfer_function(self.membrane_potential, **self.axon.transfer_function_args))

        return tf.group(potential_to_rate, rate_to_potential, name=self.key + '_state_update')

    def connect_to_targets(self):
        """Project population output firing rate to connected targets.
        """

        # project source firing rate to connected populations
        #####################################################

        with self.tf_graph.as_default():

            with tf.variable_scope(self.key):

                # loop over target populations connected to source
                project_to_targets = [syn.synaptic_buffer[self.target_delays[i]].assign(
                    syn.synaptic_buffer[self.target_delays[i]] + self.axon.firing_rate * self.target_weights[i])
                                      for i, syn in enumerate(self.targets)]

        return tf.tuple(project_to_targets, name=self.key + '_project_to_targets') if len(project_to_targets) > 0 \
            else tf.no_op()

    def copy_synapse(self,
                     synapse_key: str,
                     copy_key: Optional[str] = None,
                     **kwargs
                     ) -> str:
        """Copies an existing synapse

        Parameters
        ----------
        synapse_key
            Label of synapse to copy.
        copy_key
            Label of synapse copy.

        Returns
        -------
        copy_key
            Identifier of the synapse copy.

        """

        # copy synapse
        synapse = deepcopy(self.synapses[synapse_key])

        # add copy to synapse list
        self.add_synapse(synapse, copy_of=synapse_key, synapse_key=copy_key, **kwargs)

        return synapse

    def add_synapse(self,
                    synapse: Synapse,
                    copy_of: Optional[str] = None,
                    synapse_key: Optional[str] = None,
                    max_firing_rate: Optional[float] = None
                    ) -> None:
        """Adds copy of specified synapse to population.

        Parameters
        ----------
        synapse
            Synapse object to add (default = None)
        copy_of
            Label of synapse of which new synapse is a copy.
        synapse_key
            Label of synapse to add (default = None).
        max_firing_rate
            Maximum firing rate of pre-synapse (only needed for plastic synapses, default = None).

        """

        # add synapse to dictionary
        ###########################

        # get unique synapse identifier
        if not synapse_key:
            if not copy_of:
                synapse_key = str(self.n_synapses)
            else:
                synapse_key = copy_of + '_copy'
        count = 2
        while True:
            try:
                if synapse_key in list(self.synapses.keys()):
                    if count == 2:
                        synapse_key += '_copy_'
                    else:
                        synapse_key[-1] = str(count)
                    count += 1
                else:
                    break
            except ValueError:
                raise ValueError

        # add synapse
        self.synapses[synapse_key] = synapse
        synapse.key = synapse_key
        self.synapse_keys += [synapse_key]

        # update synapse dependencies
        #############################

        self.n_synapses = len(self.synapses)

        # current vector
        self.synaptic_currents = np.zeros(self.n_synapses)
        self.PSPs = np.zeros(self.n_synapses)
        if not self.features['integro_differential']:
            self.synaptic_currents_new = np.zeros(self.n_synapses)
            self.PSPs_new = np.zeros(self.n_synapses)

        # extrinsic modulation
        ######################

        if self.features['modulation_enabled']:
            self.extrinsic_synaptic_modulation = np.ones(self.n_synapses)

        # check plasticity related stuff
        ################################

        if self.features['synaptic_plasticity'] and copy_of:

            idx = self.synapse_keys.index(copy_of)
            if self.synapse_efficacy_adaptation[idx]:

                if max_firing_rate is None:
                    max_firing_rate = self.axon.transfer_function_args['max_firing_rate']

                self.synapse_efficacy_adaptation.append(self.synapse_efficacy_adaptation[idx])
                self.synapse_efficacy_adaptation_args.append(self.synapse_efficacy_adaptation_args[idx])
                self.synapse_efficacy_adaptation_args[-1]['max_firing_rate'] = max_firing_rate
                self.synaptic_depression = np.ones(self.n_synapses)

    def clear(self,
              disconnect: bool = False,
              init_state: Optional[dict] = None
              ):
        """Clears states stored on population and synaptic input.
        """

        # population states
        if init_state:
            if type(init_state) is dict:
                for key, state in init_state.items():
                    setattr(self, key, state)
            else:
                for key in ['membrane_potential', 'firing_rate', 'PSPs', 'PSPs_new', 'synaptic_currents',
                            'synaptic_currents_new', 'extrinsic_current', 'extrinsic_synaptic_modulation']:
                    if hasattr(self, key):
                        attr = getattr(self, key)
                        if type(attr) is np.ndarray:
                            if attr.sum() > 0:
                                setattr(self, key, np.ones_like(attr))
                            else:
                                setattr(self, key, np.zeros_like(attr))
                        elif attr == 'firing_rate':
                            setattr(self, key, self.get_firing_rate())
                        elif attr == 'membrane_potential':
                            setattr(self, key, init_state)
                        else:
                            setattr(self, key, 0.)
        else:
            for key in ['membrane_potential', 'firing_rate', 'PSPs', 'PSPs_new', 'synaptic_currents',
                        'synaptic_currents_new', 'extrinsic_current', 'extrinsic_synaptic_modulation']:
                if hasattr(self, key):
                    attr = getattr(self, key)
                    if type(attr) is np.ndarray:
                        if attr.sum() > 0:
                            setattr(self, key, np.ones_like(attr))
                        else:
                            setattr(self, key, np.zeros_like(attr))
                    elif attr == 'firing_rate':
                        setattr(self, key, self.get_firing_rate())
                    else:
                        setattr(self, key, 0.)

        # synapse attributes
        for _, syn in self.synapses.items():
            syn.clear()

        # axon attributes
        self.axon.clear()
        self.firing_rate = self.get_firing_rate()

        # network connections
        if disconnect:
            self.disconnect()

    def update(self) -> None:
        """update population attributes.
        """

        # update synapses
        for _, syn in self.synapses.items():
            syn.bin_size = self.step_size
            if not self.features['integro_differential']:
                syn.buffer_size = int(self.max_population_delay / self.step_size)
            else:
                syn.buffer_size = self.max_population_delay
            syn.update()

        # update axon
        if hasattr(self.axon, 'bin_size'):
            self.axon.bin_size = self.step_size
        self.axon.update()

    def axon_update(self,
                    t: float,
                    adaptation_old: float
                    ) -> float:
        """Updates adaptation field of axon.
        """

        return self.spike_frequency_adaptation(adaptation_old,
                                               self.firing_rate,
                                               **self.spike_frequency_adaptation_args)

    def synapse_updates(self,
                        t: float,
                        depression_old: np.ndarray
                        ) -> np.ndarray:
        """Updates synaptic depressions of all synapses on population.
        """

        return np.array([self.synapse_efficacy_adaptation[idx](depression_old,
                                                               self.synapses[key].synaptic_input[
                                                                   self.synapses[key].kernel_length - 1],
                                                               **self.synapse_efficacy_adaptation_args[idx])
                         for idx, key in self.synapses.items()])

    def plot_synaptic_kernels(self,
                              synapse_idx: Optional[List[int]] = None,
                              create_plot: Optional[bool] = True,
                              axis: Axes = None
                              ) -> object:
        """Creates plot of all specified synapses over time.

        Parameters
        ----------
        synapse_idx
            Index of synapses for which to plot kernel.
        create_plot
            If true, plot will be shown.
        axis
            Can be used to pass figure handle of figure to plot into.

        Returns
        -------
        object
            Handle of newly created or updated figure axis.

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

        if axis is None:
            fig, axis = plt.subplots(num='Synaptic Kernel Functions')
        else:
            fig = axis.get_figure()

        synapse_types = list()
        for i in synapse_idx:
            axis = self.synapses[i].plot_synaptic_kernel(create_plot=False, axis=axis)
            synapse_types.append(self.synapses[i].synapse_type)

        plt.legend(synapse_types)

        if create_plot:
            fig.show()

        return axis
