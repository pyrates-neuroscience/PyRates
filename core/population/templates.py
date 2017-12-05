"""Templates for specific population parametrizations.
"""

from core.population import Population
from typing import Optional, List, Dict, Union
import numpy as np

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


class JansenRitPyramidalCells(Population):
    """Pyramidal cell population as defined in [1]_.

    Parameters
    ----------
    init_state
        Default = 0 V.
    step_size
        Default = 0.0001 s.
    variable_step_size
        Default = False.
    max_synaptic_delay
        Default = 0.1 s.
    synaptic_modulation_direction
        Default = None.
    tau_leak
        Default = 0..016 s.
    resting_potential
        Default = -0.075 V.
    membrane_capacitance
        Default = 1e-12 q/V
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

    def __init__(self, init_state: Optional[float]=0.,
                 step_size: Optional[float] = 5e-4,
                 variable_step_size: Optional[bool] = False,
                 max_synaptic_delay: Optional[float] = 0.05,
                 synaptic_modulation_direction: Optional[np.ndarray] = None,
                 tau_leak: Optional[float] = 0.016,
                 resting_potential: Optional[float] = -0.075,
                 membrane_capacitance: Optional[float] = 1e-12,
                 max_population_delay: Optional[float] = 0.,
                 synapse_params: Optional[List[Dict[str, Union[bool, float]]]] = None,
                 axon_params: Optional[Dict[str, float]] = None,
                 store_state_variables: Optional[bool] = False,
                 label: str='JR_PCs') -> None:
        """Instantiates JansenRitPyramidalCell population with AMPA+GABAA current based synapses and Jansen-Rit Axon.
        """

        super().__init__(synapses=['AMPA_current', 'GABAA_current'],
                         axon='JansenRit',
                         init_state=init_state,
                         step_size=step_size,
                         variable_step_size=variable_step_size,
                         max_synaptic_delay=max_synaptic_delay,
                         synaptic_modulation_direction=synaptic_modulation_direction,
                         tau_leak=tau_leak,
                         resting_potential=resting_potential,
                         membrane_capacitance=membrane_capacitance,
                         max_population_delay=max_population_delay,
                         synapse_params=synapse_params,
                         axon_params=axon_params,
                         store_state_variables=store_state_variables,
                         label=label)


class JansenRitExcitatoryInterneurons(Population):
    """Excitatory interneuron population as defined in [1]_.

    Parameters
    ----------
    init_state
        Default = 0 V.
    step_size
        Default = 0.0001 s.
    variable_step_size
        Default = False.
    max_synaptic_delay
        Default = 0.1 s.
    synaptic_modulation_direction
        Default = None.
    tau_leak
        Default = 0..016 s.
    resting_potential
        Default = -0.075 V.
    membrane_capacitance
        Default = 1e-12 q/V
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

    def __init__(self, init_state: Optional[float] = 0.,
                 step_size: Optional[float] = 5e-4,
                 variable_step_size: Optional[bool] = False,
                 max_synaptic_delay: Optional[float] = 0.05,
                 synaptic_modulation_direction: Optional[np.ndarray] = None,
                 tau_leak: Optional[float] = 0.016,
                 resting_potential: Optional[float] = -0.075,
                 membrane_capacitance: Optional[float] = 1e-12,
                 max_population_delay: Optional[float] = 0.,
                 synapse_params: Optional[List[Dict[str, Union[bool, float]]]] = None,
                 axon_params: Optional[Dict[str, float]] = None,
                 store_state_variables: Optional[bool] = False,
                 label: str='JR_EINs') -> None:
        """Instantiates JansenRitExcitatoryInterneuron population with AMPA current based synapse and Jansen-Rit Axon.
        """

        super().__init__(synapses=['AMPA_current'],
                         axon='JansenRit',
                         init_state=init_state,
                         step_size=step_size,
                         variable_step_size=variable_step_size,
                         max_synaptic_delay=max_synaptic_delay,
                         synaptic_modulation_direction=synaptic_modulation_direction,
                         tau_leak=tau_leak,
                         resting_potential=resting_potential,
                         membrane_capacitance=membrane_capacitance,
                         max_population_delay=max_population_delay,
                         synapse_params=synapse_params,
                         axon_params=axon_params,
                         store_state_variables=store_state_variables,
                         label=label)


class JansenRitInhibitoryInterneurons(Population):
    """Inhibitory interneuron population as defined in [1]_.

    Parameters
    ----------
    init_state
        Default = 0 V.
    step_size
        Default = 0.0001 s.
    variable_step_size
        Default = False.
    max_synaptic_delay
        Default = 0.1 s.
    synaptic_modulation_direction
        Default = None.
    tau_leak
        Default = 0..016 s.
    resting_potential
        Default = -0.075 V.
    membrane_capacitance
        Default = 1e-12 q/V
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

    def __init__(self, init_state: Optional[float] = 0.,
                 step_size: Optional[float] = 5e-4,
                 variable_step_size: Optional[bool] = False,
                 max_synaptic_delay: Optional[float] = 0.05,
                 synaptic_modulation_direction: Optional[np.ndarray] = None,
                 tau_leak: Optional[float] = 0.016,
                 resting_potential: Optional[float] = -0.075,
                 membrane_capacitance: Optional[float] = 1e-12,
                 max_population_delay: Optional[float] = 0.,
                 synapse_params: Optional[List[Dict[str, Union[bool, float]]]] = None,
                 axon_params: Optional[Dict[str, float]] = None,
                 store_state_variables: Optional[bool] = False,
                 label: str = 'JR_IINs') -> None:
        """Instantiates JansenRitInhibitoryInterneuron population with AMPA current based synapse and Jansen-Rit Axon.
        """

        super().__init__(synapses=['AMPA_current'],
                         axon='JansenRit',
                         init_state=init_state,
                         step_size=step_size,
                         variable_step_size=variable_step_size,
                         max_synaptic_delay=max_synaptic_delay,
                         synaptic_modulation_direction=synaptic_modulation_direction,
                         tau_leak=tau_leak,
                         resting_potential=resting_potential,
                         membrane_capacitance=membrane_capacitance,
                         max_population_delay=max_population_delay,
                         synapse_params=synapse_params,
                         axon_params=axon_params,
                         store_state_variables=store_state_variables,
                         label=label)
