"""Templates for specific population parametrizations.
"""

from core.population import Population, PlasticPopulation, SecondOrderPopulation, SecondOrderPlasticPopulation
from typing import Optional, List, Dict, Union
import numpy as np

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


class JansenRitPyramidalCells(SecondOrderPopulation):
    """Pyramidal cell population with excitatory and inhibitory synapse as defined in [1]_.

    Parameters
    ----------
    synapses
        Default = JansenRitExcitatorySynapse, JansenRitInhibitorySynapse.
    axon
        Default = JansenRitAxon.
    init_state
        Default = 0 V.
    step_size
        Default = 0.0001 s.
    max_synaptic_delay
        Default = 0.1 s.
    max_population_delay
        Default = 0.
    synapse_params
        Default = None.
    axon_params
        Default = None.
    store_state_variables
        Default = False.

    See Also
    --------
    :class:`Population`: Detailed documentation of population parameters, attributes and methods.

    References
    ----------
    .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
       of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

    """

    def __init__(self,
                 synapses: List[str] = ['JansenRitExcitatorySynapse', 'JansenRitInhibitorySynapse'],
                 axon: str = 'JansenRitAxon',
                 init_state: float = 0.,
                 step_size: float = 5e-4,
                 max_synaptic_delay: float = 0.05,
                 max_population_delay: float = 0.,
                 synapse_params: Optional[List[Dict[str, Union[bool, float]]]] = None,
                 axon_params: Optional[Dict[str, float]] = None,
                 store_state_variables: bool = False,
                 label: str = 'JR_PCs'
                 ) -> None:
        """Instantiates JansenRit PC population with second order synapses and Jansen-Rit axon.
        """

        super().__init__(synapses=synapses,
                         axon=axon,
                         init_state=init_state,
                         step_size=step_size,
                         max_synaptic_delay=max_synaptic_delay,
                         max_population_delay=max_population_delay,
                         synapse_params=synapse_params,
                         axon_params=axon_params,
                         store_state_variables=store_state_variables,
                         label=label,
                         synapse_class='ExponentialSynapse',
                         axon_class='SigmoidAxon')


class JansenRitInterneurons(SecondOrderPopulation):
    """Interneuron population with excitatory synapse as defined in [1]_.

    Parameters
    ----------
    synapses
        Default = JansenRitExcitatorySynapse.
    axon
        Default = JansenRitAxon.
    init_state
        Default = 0 V.
    step_size
        Default = 0.0001 s.
    max_synaptic_delay
        Default = 0.1 s.
    max_population_delay
        Default = 0.
    synapse_params
        Default = None.
    axon_params
        Default = None.
    store_state_variables
        Default = False.

    See Also
    --------
    :class:`Population`: Detailed documentation of population parameters, attributes and methods.

    References
    ----------
    .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
       of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

    """

    def __init__(self,
                 synapses: List[str] = ['JansenRitExcitatorySynapse'],
                 axon: str = 'JansenRitAxon',
                 init_state: float = 0.,
                 step_size: float = 5e-4,
                 max_synaptic_delay: float = 0.05,
                 max_population_delay: float = 0.,
                 synapse_params: Optional[List[Dict[str, Union[bool, float]]]] = None,
                 axon_params: Optional[Dict[str, float]] = None,
                 store_state_variables: bool = False,
                 label: str = 'JR_INs'
                 ) -> None:
        """Instantiates JansenRit interneuron population with second order synapse and Jansen-Rit axon.
        """

        super().__init__(synapses=synapses,
                         axon=axon,
                         init_state=init_state,
                         step_size=step_size,
                         max_synaptic_delay=max_synaptic_delay,
                         max_population_delay=max_population_delay,
                         synapse_params=synapse_params,
                         axon_params=axon_params,
                         store_state_variables=store_state_variables,
                         label=label,
                         synapse_class='ExponentialSynapse',
                         axon_class='SigmoidAxon')


class MoranPyramidalCells(SecondOrderPopulation):
    """Population of pyramidal cells as described in [1]_.

    Parameters
    ----------

    See Also
    --------

    References
    ----------

    """
    def __init__(self,
                 synapses: List[str] = ['MoranExcitatorySynapse', 'MoranInhibitorySynapse'],
                 axon: str = 'MoranAxon',
                 init_state: float = 0.,
                 step_size: float = 0.0001,
                 max_synaptic_delay: Optional[Union[float, np.ndarray]] = None,
                 resting_potential: float = 0.,
                 max_population_delay: float = 0.,
                 synapse_params: Optional[List[dict]] = None,
                 axon_params: Optional[Dict[str, float]] = None,
                 synapse_class: Union[str, List[str]] = 'ExponentialSynapse',
                 axon_class: str = 'Axon',
                 store_state_variables: bool = False,
                 label: str = 'MoranCells'
                 ) -> None:
        """Instantiates a population as defined in [1]_.
        """

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
                         label=label
                         )


class MoranExcitatoryInterneurons(SecondOrderPlasticPopulation):
    """Population of excitatory interneurons without spike-frequency-adaptation (see [1]_).

    Parameters
    ----------

    See Also
    --------

    References
    ----------

    """

    def __init__(self,
                 synapses: List[str] = ['MoranExcitatorySynapse'],
                 axon: str = 'MoranAxon',
                 init_state: float = 0.,
                 step_size: float = 0.0001,
                 max_synaptic_delay: Optional[Union[float, np.ndarray]] = None,
                 resting_potential: float = 0.,
                 max_population_delay: float = 0.,
                 synapse_params: Optional[List[dict]] = None,
                 axon_params: Optional[Dict[str, float]] = None,
                 synapse_class: Union[str, List[str]] = 'ExponentialSynapse',
                 axon_class: str = 'Axon',
                 store_state_variables: bool = False,
                 tau: float = 0.512,
                 label: str = 'MoranCells'
                 ) -> None:
        """Instantiates a population as defined in [1]_ with a spike-frequency-adaptation mechanism.
        """

        ###############################################
        # define spike frequency adaptation mechanism #
        ###############################################

        # function
        def spike_frequency_adaptation(adaption, firing_rate_target, tau):
            """Calculates adaption in sigmoid threshold of axonal transfer function.

            Parameters
            ----------
            adaption
            firing_rate_target
            tau

            Returns
            -------
            float
                Change in threshold of sigmoidal axonal transfer function.

            """

            return (firing_rate_target - adaption) / tau

        # adaptation time constant
        params = {'tau': tau}

        ###################
        # call super init #
        ###################

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
                         axon_plasticity_function=spike_frequency_adaptation,
                         axon_plasticity_target_param='adaption',
                         axon_plasticity_function_params=params,
                         label=label)


class MoranInhibitoryInterneurons(SecondOrderPopulation):
    """Population of inhibitory interneurons without spike-frequency-adaptation (see [1]_).

    Parameters
    ----------

    See Also
    --------

    References
    ----------

    """

    def __init__(self,
                 synapses: List[str] = ['MoranExcitatorySynapse', 'MoranInhibitorySynapse'],
                 axon: str = 'MoranAxon',
                 init_state: float = 0.,
                 step_size: float = 0.0001,
                 max_synaptic_delay: Optional[Union[float, np.ndarray]] = None,
                 resting_potential: float = 0.,
                 max_population_delay: float = 0.,
                 synapse_params: Optional[List[dict]] = None,
                 axon_params: Optional[Dict[str, float]] = None,
                 synapse_class: Union[str, List[str]] = 'ExponentialSynapse',
                 axon_class: str = 'Axon',
                 store_state_variables: bool = False,
                 label: str = 'MoranCells'
                 ) -> None:
        """Instantiates a population as defined in [1]_ with a spike-frequency-adaptation mechanism.
        """

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
                         label=label
                         )


class WangKnoescheCells(SecondOrderPlasticPopulation):
    """Population of cells with synaptic-efficacy-adaptation (see [1]_).

    Parameters
    ----------

    See Also
    --------

    References
    ----------

    """
    def __init__(self,
                 synapses: List[str] = ['JansenRitExcitatorySynapse', 'JansenRitInhibitorySynapse'],
                 axon: str = 'JansenRitAxon',
                 init_state: float = 0.,
                 step_size: float = 0.0001,
                 max_synaptic_delay: Optional[Union[float, np.ndarray]] = None,
                 resting_potential: float = 0.,
                 max_population_delay: float = 0.,
                 synapse_params: Optional[List[dict]] = None,
                 axon_params: Optional[Dict[str, float]] = None,
                 synapse_class: Union[str, List[str]] = 'ExponentialSynapse',
                 axon_class: str = 'SigmoidAxon',
                 store_state_variables: bool = False,
                 label: str = 'MoranCells',
                 tau_depression: float = 0.05,
                 tau_recycle: float = 0.5,
                 plastic_synapses: List[bool] = [True, False]
                 ) -> None:
        """Instantiates a population as defined in [1]_ with a synaptic efficacy adaptation mechanism.
        """

        ########################################
        # define synaptic adaptation mechanism #
        ########################################

        # function
        def synaptic_efficacy_adaptation(efficacy, firing_rate, max_firing_rate, tau_depression, tau_recycle):
            """Calculates synaptic efficacy change.

            Parameters
            ----------
            efficacy
            firing_rate
            max_firing_rate
            tau_depression
            tau_recycle

            Returns
            -------
            float
                Synaptic efficacy change.

            """

            depression_rate = (efficacy * firing_rate) / (max_firing_rate * tau_depression)
            recycle_rate = (1 - efficacy) / tau_recycle

            return recycle_rate - depression_rate if firing_rate > 0. else recycle_rate

        # function parameters
        params = {'tau_depression': tau_depression,
                  'tau_recycle': tau_recycle}
        param_list = list()
        for p in plastic_synapses:
            if p:
                param_list.append(params)
            else:
                param_list.append(None)

        ###################
        # call super init #
        ###################

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
                         synapse_plasticity_function=synaptic_efficacy_adaptation,
                         synapse_plasticity_function_params=param_list
                         )

    def get_firing_rate(self) -> Union[float, np.float64]:
        """Calculate the current average firing rate of the population.

        Returns
        -------
        float
            Average firing rate of population [unit = 1/s].

        """

        return self.axon.compute_firing_rate(self.state_variables[-1][0]) - self.axon.compute_firing_rate(0.)
