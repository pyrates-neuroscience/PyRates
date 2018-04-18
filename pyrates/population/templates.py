"""Templates for specific population parametrizations.
"""

# external packages
from typing import Optional, List, Dict, Union

# pyrates internal imports
from pyrates.population import Population
from pyrates.utility import moran_spike_frequency_adaptation, synaptic_efficacy_adaptation

# meta infos
__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


##################################
# JansenRit population templates #
##################################


class JansenRitPyramidalCells(Population):
    """Pyramidal cell population with excitatory and inhibitory synapse as defined in [1]_.

    See Also
    --------
    :class:`Population`: Detailed documentation of population parameters, attributes and methods.

    References
    ----------
    .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
       of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

    """

    def __init__(self,
                 synapses: Optional[List[str]] = None,
                 axon: str = 'JansenRitAxon',
                 init_state: float = 0.,
                 step_size: float = 5e-4,
                 max_population_delay: float = 0.,
                 synapse_params: Optional[List[Dict[str, Union[bool, float]]]] = None,
                 axon_params: Optional[Dict[str, float]] = None,
                 synapse_keys: Optional[List[str]] = None,
                 key: str = 'JR_PCs'
                 ) -> None:
        """Instantiates JansenRit PC population with two alpha-kernel based synapses (excitatory + inhibitory) and
        an sigmoidal axon.
        """

        # check synapse parameters
        ##########################

        # synapse type
        synapses = ['JansenRitExcitatorySynapse', 'JansenRitInhibitorySynapse'] if not synapses else synapses

        # synapse key
        synapse_keys = ['excitatory', 'inhibitory'] if not synapse_keys else synapse_keys

        # call super init
        #################

        super().__init__(synapses=synapses,
                         axon=axon,
                         init_state=init_state,
                         step_size=step_size,
                         max_population_delay=max_population_delay,
                         synapse_params=synapse_params,
                         axon_params=axon_params,
                         key=key,
                         synapse_class='ExponentialSynapse',
                         synapse_keys=synapse_keys,
                         axon_class='SigmoidAxon')


class JansenRitInterneurons(Population):
    """Interneuron population with excitatory synapse as defined in [1]_.

    See Also
    --------
    :class:`Population`: Detailed documentation of population parameters, attributes and methods.

    References
    ----------
    .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
       of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

    """

    def __init__(self,
                 synapses: Optional[List[str]] = None,
                 axon: str = 'JansenRitAxon',
                 init_state: float = 0.,
                 step_size: float = 5e-4,
                 max_population_delay: float = 0.,
                 synapse_params: Optional[List[Dict[str, Union[bool, float]]]] = None,
                 axon_params: Optional[Dict[str, float]] = None,
                 synapse_keys: Optional[List[str]] = None,
                 key: str = 'JR_INs'
                 ) -> None:
        """Instantiates JansenRit interneuron population with second order synapse and Jansen-Rit axon.
        """

        # check synapse parameters
        ##########################

        # synapse type
        synapses = ['JansenRitExcitatorySynapse'] if not synapses else synapses

        # synapse key
        synapse_keys = ['excitatory'] if not synapse_keys else synapse_keys

        # call super init
        #################

        super().__init__(synapses=synapses,
                         axon=axon,
                         init_state=init_state,
                         step_size=step_size,
                         max_population_delay=max_population_delay,
                         synapse_params=synapse_params,
                         axon_params=axon_params,
                         key=key,
                         synapse_class='ExponentialSynapse',
                         synapse_keys=synapse_keys,
                         axon_class='SigmoidAxon')


#####################################################
# moran populations with spike frequency adaptation #
#####################################################


class MoranPyramidalCells(Population):
    """Population of pyramidal cells as described in [1]_.

    See Also
    --------
    :class:`Population`: Detailed documentation of population parameters, attributes and methods.

    References
    ----------
    .. [1] R.J. Moran, S.J. Kiebel, K.E. Stephan, R.B. Reilly, J. Daunizeau & K.J. Friston, "A Neural Mass Model of
       Spectral Responses in Electrophysiology" NeuroImage, vol. 37, pp. 706-720, 2007.

    """
    def __init__(self,
                 synapses: Optional[List[str]] = None,
                 axon: str = 'MoranAxon',
                 init_state: float = 0.,
                 step_size: float = 5e-4,
                 resting_potential: float = 0.,
                 max_population_delay: float = 0.,
                 synapse_params: Optional[List[dict]] = None,
                 axon_params: Optional[Dict[str, float]] = None,
                 synapse_class: Union[str, List[str]] = 'ExponentialSynapse',
                 axon_class: str = 'PlasticSigmoidAxon',
                 tau: Optional[float] = None,
                 synapse_keys: Optional[List[str]] = None,
                 key: str = 'Moran_PCs'
                 ) -> None:
        """Instantiates a population as defined in [1]_.
        """

        # check synapse parameters
        ##########################

        # synapse type
        synapses = ['MoranExcitatorySynapse', 'MoranInhibitorySynapse'] if not synapses else synapses

        # synapse keys
        synapse_keys = ['excitatory', 'inhibitory'] if not synapse_keys else synapse_keys

        # set parameter dictionaries
        ############################

        # axonal plasticity
        params = {'tau': tau}

        # axon
        if not axon_params:
            axon_params = [{'normalize': True} for _ in range(len(synapses))]

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
                         spike_frequency_adaptation=moran_spike_frequency_adaptation if tau else None,
                         spike_frequency_adaptation_kwargs=params,
                         synapse_keys=synapse_keys,
                         key=key
                         )


class MoranExcitatoryInterneurons(Population):
    """Population of excitatory interneurons as defined in [1]_.

    See Also
    --------
    :class:`Population`: Detailed documentation of population parameters, attributes and methods.

    References
    ----------
    .. [1] R.J. Moran, S.J. Kiebel, K.E. Stephan, R.B. Reilly, J. Daunizeau & K.J. Friston, "A Neural Mass Model of
       Spectral Responses in Electrophysiology" NeuroImage, vol. 37, pp. 706-720, 2007.

    """

    def __init__(self,
                 synapses: Optional[List[str]] = None,
                 axon: str = 'MoranAxon',
                 init_state: float = 0.,
                 step_size: float = 5e-4,
                 resting_potential: float = 0.,
                 max_population_delay: float = 0.,
                 synapse_params: Optional[List[dict]] = None,
                 axon_params: Optional[Dict[str, float]] = None,
                 synapse_class: Union[str, List[str]] = 'ExponentialSynapse',
                 axon_class: str = 'PlasticSigmoidAxon',
                 synapse_keys: Optional[List[str]] = None,
                 key: str = 'Moran_EINs'
                 ) -> None:
        """Instantiates a population as defined in [1]_ with a spike-frequency-adaptation mechanism.
        """

        # check synapse parameters
        ##########################

        # synapse type
        synapses = ['MoranExcitatorySynapse'] if not synapses else synapses

        # synapse keys
        synapse_keys = ['excitatory'] if not synapse_keys else synapse_keys

        # axon params
        if not axon_params:
            axon_params = [{'normalize': True} for _ in range(len(synapses))]

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
                         synapse_keys=synapse_keys,
                         key=key)


class MoranInhibitoryInterneurons(Population):
    """Population of inhibitory interneurons without spike-frequency-adaptation (see [1]_).

    See Also
    --------
    :class:`Population`: Detailed documentation of population parameters, attributes and methods.

    References
    ----------
    .. [1] R.J. Moran, S.J. Kiebel, K.E. Stephan, R.B. Reilly, J. Daunizeau & K.J. Friston, "A Neural Mass Model of
       Spectral Responses in Electrophysiology" NeuroImage, vol. 37, pp. 706-720, 2007.

    """

    def __init__(self,
                 synapses: Optional[List[str]] = None,
                 axon: str = 'MoranAxon',
                 init_state: float = 0.,
                 step_size: float = 5e-4,
                 resting_potential: float = 0.,
                 max_population_delay: float = 0.,
                 synapse_params: Optional[List[dict]] = None,
                 axon_params: Optional[Dict[str, float]] = None,
                 synapse_class: Union[str, List[str]] = 'ExponentialSynapse',
                 axon_class: str = 'PlasticSigmoidAxon',
                 synapse_keys: Optional[List[str]] = None,
                 key: str = 'Moran_IINs'
                 ) -> None:
        """Instantiates a population as defined in [1]_ with a spike-frequency-adaptation mechanism.
        """

        # check synapse parameters
        ##########################

        # synapse type

        if not synapses:
            synapses = ['MoranExcitatorySynapse', 'MoranInhibitorySynapse'] if not synapses else synapses

        # synapse keys
        synapse_keys = ['excitatory', 'inhibitory'] if not synapse_keys else synapse_keys

        # axon params
        if not axon_params:
            axon_params = [{'normalize': True} for _ in range(len(synapses))]

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
                         synapse_keys=synapse_keys,
                         key=key
                         )


#####################################################
# wang-knoesche population with synaptic plasticity #
#####################################################


class WangKnoescheCells(Population):
    """Population of cells with synaptic-efficacy-adaptation (see [1]_).

    See Also
    --------
    :class:`Population`: Detailed documentation of population parameters, attributes and methods.

    References
    ----------
    .. [1] P. Wang & T.R. Knoesche, "A realistic neural mass model of the cortex with laminar-specific connections and
       synaptic plasticity-evaluation with auditory habituation." PloS one, vol. 8(10): e77876, 2013.

    """
    def __init__(self,
                 synapses: Optional[List[str]] = None,
                 axon: str = 'JansenRitAxon',
                 init_state: float = 0.,
                 step_size: float = 5e-4,
                 resting_potential: float = 0.,
                 max_population_delay: float = 0.,
                 synapse_params: Optional[List[dict]] = None,
                 axon_params: Optional[Dict[str, float]] = None,
                 synapse_class: Union[str, List[str]] = 'ExponentialSynapse',
                 axon_class: str = 'SigmoidAxon',
                 tau_depression: float = 0.05,
                 tau_recycle: float = 0.5,
                 plastic_synapses: Optional[List[bool]] = None,
                 synapse_keys: Optional[List[str]] = None,
                 key: str = 'WangKnoeschePopulation'
                 ) -> None:
        """Instantiates a population as defined in [1]_ with a synaptic efficacy adaptation mechanism.
        """

        # check synapse parameters
        ##########################

        # synapse type

        if not synapses:
            synapses = ['JansenRitExcitatorySynapse', 'JansenRitInhibitorySynapse'] if not synapses else synapses

        # synapse keys
        synapse_keys = ['excitatory', 'inhibitory'] if not synapse_keys else synapse_keys

        # synaptic plasticity
        if not plastic_synapses:  # A list [False] evaluates correctly
            plastic_synapses = [True, False]

        # synaptic plasticity params
        ############################

        params = {'tau_depression': tau_depression,
                  'tau_recycle': tau_recycle}
        param_list = list()
        func_list = list()

        for p in plastic_synapses:
            if p:
                param_list.append(params)
                func_list.append(synaptic_efficacy_adaptation)
            else:
                param_list.append(None)
                func_list.append(None)

        # axon params
        #############

        if axon_params is None:
            axon_params = {'normalize': True}

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
                         synapse_efficacy_adaptation=func_list,
                         synapse_efficacy_adaptation_kwargs=param_list,
                         synapse_keys=synapse_keys,
                         key=key
                         )
