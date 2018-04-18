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
from types import MethodType

# pyrates internal imports
from pyrates.axon import Axon, SigmoidAxon, BurstingAxon, PlasticSigmoidAxon
from pyrates.synapse import Synapse, DoubleExponentialSynapse, ExponentialSynapse, TransformedInputSynapse, \
    DEExponentialSynapse, DEDoubleExponentialSynapse
from pyrates.utility import set_instance, make_iterable
from pyrates.population.population_methods import construct_state_update_function, \
    construct_get_delta_membrane_potential_function, construct_get_synaptic_responses
from pyrates.utility.filestorage import RepresentationBase
import pyrates.solver as solv

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

    def __init__(self):
        """Instantiates basic circuit node.
        """

        self.targets = []
        self.target_weights = []
        self.target_delays = []
        self.firing_rate = 0.

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

        # get source firing rate
        source_fr = self.firing_rate

        # project source firing rate to connected populations
        #####################################################

        # extract network connections
        # targets = self.targets

        # loop over target populations connected to source
        for i, syn in enumerate(self.targets):
            syn.pass_input(source_fr * self.target_weights[i], delay=self.target_delays[i])
            # it would be possible to wrap this function using functools.partial to remove the delay lookup


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
                 key: Optional[str] = None
                 ):
        """Instantiate dummy population for synaptic input passage.
        """

        super().__init__()

        if len(self.targets) > 1:
            raise ValueError('Synaptic input dummy populations can only connect to a single target. Use multiple dummy'
                             ' populations to connect to multiple targets.')

        self.output = output
        self.idx = 0
        self.key = key if key else 'dummy_synaptic_input'

    def project_to_targets(self):
        """Project output to pass synapse function of target.
        """

        self.targets[0].pass_input(self.output[self.idx])

    def state_update(self):
        """Advance in output array by one position.
        """

        self.idx += 1


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
                 synapse_class: Union[str, List[str]] = 'DoubleExponentialSynapse',
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
                 solver: str = 'ForwardEuler',
                 synapse_keys: Optional[list] = None,
                 key: Optional[str] = None,
                 verbose: bool = False
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
                           max_synaptic_delay=max_synaptic_delay,
                           synapse_keys=synapse_keys)

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

        # set initial states
        ####################

        if not init_state:
            init_state = resting_potential if resting_potential else 0.

        # initialize state variables
        self.PSPs = np.zeros(self.n_synapses)
        self.synaptic_currents = np.zeros(self.n_synapses)
        self.extrinsic_current = 0.

        if self.features['leaky_capacitor']:
            self.leak_current = 0.
        if not self.features['integro_differential']:
            self.synaptic_currents_new = np.zeros(self.n_synapses)
            self.PSPs_new = np.zeros(self.n_synapses)
        if self.features['modulation_enabled']:
            self.extrinsic_synaptic_modulation = np.ones(self.n_synapses)

        # set to passed initial states
        if type(init_state) is not dict:
            self.membrane_potential = init_state
        else:
            for key, state in init_state.items():
                setattr(self, key, state)
        self.firing_rate = self.get_firing_rate()

        # set plasticity attributes
        ###########################

        # for axon
        if self.spike_frequency_adaptation:
            self.spike_frequency_adaptation_args = deepcopy(spike_frequency_adaptation_kwargs) \
                if spike_frequency_adaptation_kwargs else dict()
            self.axonal_adaptation = self.axon.transfer_function_args['adaptation']

        # synaptic plasticity function
        if synapse_efficacy_adaptation and (type(synapse_efficacy_adaptation) is not list):
            self.synapse_efficacy_adaptation = [self.synapse_efficacy_adaptation for _ in range(self.n_synapses)]
        elif synapse_efficacy_adaptation and (len(synapse_efficacy_adaptation) != self.n_synapses):
            raise ValueError('If list of synaptic plasticity functions is passed, its length must correspond to the'
                             ' number of synapses of the population.')
        elif not synapse_efficacy_adaptation:
            self.synapse_efficacy_adaptation = [None for _ in range(self.n_synapses)]

        # synaptic plasticity function arguments
        if synapse_efficacy_adaptation_kwargs is None or type(synapse_efficacy_adaptation_kwargs) is dict:
            self.synapse_efficacy_adaptation_args = [self.synapse_efficacy_adaptation_args
                                                     for _ in range(self.n_synapses)]

        if len(self.synapse_efficacy_adaptation) != len(self.synapse_efficacy_adaptation_args):
            raise AttributeError('Number of synaptic plasticity functions and plasticity function parameter '
                                 'dictionaries has to be equal.')

        # synaptic plasticity state variables
        self.synaptic_depression = np.ones(self.n_synapses)
        for i, key in enumerate(self.synapses.keys()):
            if self.synapse_efficacy_adaptation[i]:
                self.synaptic_depression[i] = self.synapses[key].depression

        # set-up the solver system for the state-update equations
        #########################################################

        self._set_solver(solver=solver)

    def _set_synapses(self,
                      synapse_subtypes: Optional[List[str]] = None,
                      synapse_types: Union[str, List[str]] = 'DoubleExponentialSynapse',
                      synapse_params: Optional[List[dict]] = None,
                      max_synaptic_delay: Optional[Union[np.ndarray, list]] = None,
                      synapse_keys: Optional[List[str]] = None
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
        synapse_keys
            List of synapse identifiers.

        """

        # check synapse parameter formats
        #################################

        synapse_types = make_iterable(synapse_types, self.n_synapses)
        synapse_subtypes = make_iterable(synapse_subtypes, self.n_synapses)
        synapse_params = make_iterable(synapse_params, self.n_synapses)

        # set all given synapses
        ########################

        for i in range(self.n_synapses):

            # instantiate synapse
            if synapse_types[i] == 'DoubleExponentialSynapse':
                if self.features['integro_differential']:
                    synapse_instance = set_instance(DoubleExponentialSynapse,
                                                    synapse_subtypes[i],
                                                    synapse_params[i],
                                                    bin_size=self.step_size,
                                                    max_delay=max_synaptic_delay[i],
                                                    buffer_size=self.max_population_delay)
                else:
                    synapse_instance = set_instance(DEDoubleExponentialSynapse,
                                                    synapse_subtypes[i],
                                                    synapse_params[i],
                                                    buffer_size=int(self.max_population_delay / self.step_size))
            elif synapse_types[i] == 'ExponentialSynapse':
                if self.features['integro_differential']:
                    synapse_instance = set_instance(ExponentialSynapse,
                                                    synapse_subtypes[i],
                                                    synapse_params[i],
                                                    bin_size=self.step_size,
                                                    max_delay=max_synaptic_delay[i],
                                                    buffer_size=self.max_population_delay)
                else:
                    synapse_instance = set_instance(DEExponentialSynapse,
                                                    synapse_subtypes[i],
                                                    synapse_params[i],
                                                    buffer_size=int(self.max_population_delay
                                                                    / self.step_size))
            elif synapse_types[i] == 'TransformedInputSynapse':
                synapse_instance = set_instance(TransformedInputSynapse,
                                                synapse_subtypes[i],
                                                synapse_params[i],
                                                bin_size=self.step_size,
                                                max_delay=max_synaptic_delay[i],
                                                buffer_size=self.max_population_delay)
            elif synapse_types[i] == 'Synapse':
                synapse_instance = set_instance(Synapse,
                                                synapse_subtypes[i],
                                                synapse_params[i],
                                                bin_size=self.step_size,
                                                max_delay=max_synaptic_delay[i],
                                                buffer_size=self.max_population_delay)
            else:
                raise AttributeError('Invalid synapse type!')

            # bind instance to object
            if synapse_keys:

                # define unique synapse key
                occurrence = 0
                for syn_idx in range(i):
                    if synapse_keys[syn_idx] == synapse_keys[i]:
                        occurrence += 1
                if occurrence > 0:
                    synapse_keys[i] = synapse_keys[i] + '(' + str(occurrence) + ')'

                # key synapse according to provided key
                self.synapses[synapse_keys[i]] = synapse_instance
                self.synapses[synapse_keys[i]].key = synapse_keys[i]

            else:

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

    def _set_solver(self, solver: str):
        """Instantiates solver on population.

        Parameters
        ----------
        solver
            Name of the solver sub-class that is to be used to solve the state update DE's.

        """

        # build delta membrane potential method
        #######################################

        # create string representation of method and make it executable
        exec(construct_get_delta_membrane_potential_function(leaky_capacitor=self.features['leaky_capacitor'],
                                                             enable_modulation=self.features['modulation_enabled'],
                                                             integro_differential=self.features['integro_differential'])
             )

        # bind method to self
        self.get_delta_membrane_potential = MethodType(locals()['get_delta_membrane_potential'], self)

        # build get synaptic responses method
        #####################################

        # create string representation of method and make it executable
        exec(construct_get_synaptic_responses(self.features['integro_differential']))

        # bind method to self
        self.get_synaptic_responses = MethodType(locals()['get_synaptic_responses'], self)

        # instantiate the solvers
        #########################

        solver_args = {'max_step': self.step_size, 'atol': 1e-3, 'rtol': 1e-2}

        # solvers needed for updating the membrane potential
        if self.features['leaky_capacitor']:

            self.membrane_potential_solver = set_instance(solv.Solver,
                                                          instance_type=solver,
                                                          instance_params=None,
                                                          f=self.get_delta_membrane_potential,
                                                          y0=self.membrane_potential,
                                                          **solver_args)

        else:

            self.psp_solver = set_instance(solv.Solver,
                                           instance_type=solver,
                                           instance_params=None,
                                           f=self.get_delta_psps,
                                           y0=self.PSPs,
                                           **solver_args)

            self.synaptic_current_solver = set_instance(solv.Solver,
                                                        instance_type=solver,
                                                        instance_params=None,
                                                        f=self.get_synaptic_responses,
                                                        y0=self.synaptic_currents,
                                                        **solver_args)

        # additional solvers needed for plasticity mechanisms
        if self.features['axonal_plasticity']:
            self.axonal_adaptation_solver = set_instance(solv.Solver,
                                                         instance_type=solver,
                                                         instance_params=None,
                                                         f=self.axon_update,
                                                         y0=self.axonal_adaptation,
                                                         **solver_args)

        if self.features['synaptic_plasticity']:
            self.synaptic_depression_solver = set_instance(solv.Solver,
                                                           instance_type=solver,
                                                           instance_params=None,
                                                           f=self.synapse_updates,
                                                           y0=self.synaptic_depression,
                                                           **solver_args)

        # build state update method
        ###########################

        # create string representation of method and make it executable
        exec(construct_state_update_function(spike_frequency_adaptation=self.features['axonal_plasticity'],
                                             synapse_efficacy_adaptation=self.features['synaptic_plasticity'],
                                             leaky_capacitor=self.features['leaky_capacitor'])
             )

        # bind method to self
        self.state_update = MethodType(locals()['state_update'], self)

    def get_firing_rate(self) -> FloatLike:
        """Calculate the current average firing rate of the population.

        Returns
        -------
        float
            Average firing rate of population [unit = 1/s].

        """

        return self.axon.compute_firing_rate(self.membrane_potential)

    def get_delta_psps(self,
                       t: float,
                       psps_old: np.ndarray
                       ) -> np.ndarray:
        """Calculates the change in PSP.
        """

        return self.synaptic_currents

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
