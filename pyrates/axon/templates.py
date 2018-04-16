"""Templates for specific axon parametrizations.
"""

import numpy as np
from pyrates.axon import SigmoidAxon, Axon, BurstingAxon, PlasticSigmoidAxon  # type: ignore
from pyrates.utility import plastic_sigmoid, plastic_normalized_sigmoid, activation_sigmoid, inactivation_sigmoid
from pyrates.utility import axon_exponential
from typing import Optional, List, Union

__author__ = "Daniel F. Rose, Richard Gast"
__status__ = "Development"


########################
# Leaky-Capacitor axon #
########################


class KnoescheAxon(SigmoidAxon):
    """Sigmoid axon with parameters set according to Thomas Knoesche's document.

    Parameters
    ----------
    max_firing_rate
        Default = 5.0 Hz. See documentation of parameter 'max_firing_rate' of :class:`SigmoidAxon`.
    membrane_potential_threshold
        Default = -0.069 V. See documentation of parameter 'membrane_potential_threshold' of :class:`SigmoidAxon`.
    sigmoid_steepness
        Default = 555.56 Hz. See documentation of parameter 'sigmoid_steepness' of :class:`SigmoidAxon`.

    See Also
    --------
    :class:`SigmoidAxon`: Detailed description of parameters.
    :class:`Axon`: Detailed description of attributes and methods.

    """

    def __init__(self, max_firing_rate: float = 5.,
                 membrane_potential_threshold: float = -0.069,
                 sigmoid_steepness: float = 560.0) -> None:
        """Instantiates sigmoid axon with Knoesche's parameters.
        """

        super().__init__(max_firing_rate=max_firing_rate,
                         membrane_potential_threshold=membrane_potential_threshold,
                         sigmoid_steepness=sigmoid_steepness,
                         label='Knoesche')


class PlasticKnoescheAxon(PlasticSigmoidAxon):
    """Sigmoid axon with parameters set according to Thomas Knoesche's document.

    Parameters
    ----------
    max_firing_rate
        Default = 5.0 Hz. See documentation of parameter 'max_firing_rate' of :class:`SigmoidAxon`.
    membrane_potential_threshold
        Default = -0.069 V. See documentation of parameter 'membrane_potential_threshold' of :class:`SigmoidAxon`.
    sigmoid_steepness
        Default = 555.56 Hz. See documentation of parameter 'sigmoid_steepness' of :class:`SigmoidAxon`.

    See Also
    --------
    :class:`SigmoidAxon`: Detailed description of parameters.
    :class:`Axon`: Detailed description of attributes and methods.

    """

    def __init__(self,
                 max_firing_rate: float = 5.,
                 sigmoid_steepness: float = 560.,
                 membrane_potential_threshold: float = -0.069
                 ) -> None:
        """Instantiate plastic sigmoidal axon.
        """

        super().__init__(label='plastic_knoesche_axon',
                         max_firing_rate=max_firing_rate,
                         membrane_potential_threshold=membrane_potential_threshold,
                         sigmoid_steepness=sigmoid_steepness,
                         normalize=False)

########################
# JansenRit-type axons #
########################


class JansenRitAxon(SigmoidAxon):
    """Sigmoid axon with parameters set according to [1]_.

    Parameters
    ----------
    max_firing_rate
        Default = 5.0 Hz. See documentation of parameter 'max_firing_rate' of :class:`SigmoidAxon`.
    membrane_potential_threshold
        Default = 0.006 V. See documentation of parameter 'membrane_potential_threshold' of :class:`SigmoidAxon`.
    sigmoid_steepness
        Default = 560.0 Hz. See documentation of parameter 'sigmoid_steepness' of :class:`SigmoidAxon`.

    See Also
    --------
    :class:`SigmoidAxon`: Detailed description of parameters.
    :class:`Axon`: Detailed description of attributes and methods.

    References
    ----------
    .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
       of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

    """

    def __init__(self, max_firing_rate: float = 5.,
                 membrane_potential_threshold: float = 0.006,
                 sigmoid_steepness: float = 560.) -> None:
        """Instantiates sigmoid axon with Jansen & Rit's parameters.
        """

        super().__init__(max_firing_rate=max_firing_rate,
                         membrane_potential_threshold=membrane_potential_threshold,
                         sigmoid_steepness=sigmoid_steepness,
                         label='JansenRit')


class MoranAxon(PlasticSigmoidAxon):
    """Sigmoidal axon as defined in [1]_.

    Parameters
    ----------
    max_firing_rate
        Default = 1.0 Hz. See documentation of parameter 'max_firing_rate' of :class:`SigmoidAxon`.
    sigmoid_steepness
        Default = 2000 Hz. See documentation of parameter 'sigmoid_steepness' of :class:`SigmoidAxon`.
    membrane_potential_threshold
        Default = 0.001 V. See documentation of parameter 'membrane_potential_threshold' of :class:`SigmoidAxon`.

    See Also
    --------
    :class:`Axon`: Detailed description of attributes and methods.

    References
    ----------
    .. [1] R.J. Moran, S.J. Kiebel, K.E. Stephan, R.B. Reilly, J. Daunizeau & K.J. Friston, "A Neural Mass Model of
       Spectral Responses in Electrophysiology" NeuroImage, vol. 37, pp. 706-720, 2007.

    """

    def __init__(self,
                 max_firing_rate: float = 1.,
                 sigmoid_steepness: float = 2000.,
                 membrane_potential_threshold: float = 0.001
                 ) -> None:
        """Instantiate moran axon.
        """

        super().__init__(max_firing_rate=max_firing_rate,
                         membrane_potential_threshold=membrane_potential_threshold,
                         sigmoid_steepness=sigmoid_steepness,
                         normalize=True,
                         label='Moran_axon')


#################
# bursting axon #
#################


class SuffczynskiAxon(BurstingAxon):
    """Bursting axon including low-threshold spikes as described in [1]_.
    
    Parameters
    ----------
    bin_size
        time window one bin of the axonal kernel is representing [unit = s].
    axon_type
        See description of parameter `axon_type` of :class:`Axon`.
    max_delay
        Maximal time delay after which a certain membrane potential still affects the firing rate [unit = s].
    epsilon
        Accuracy of the synaptic kernel representation.
    resting_potential
        Resting membrane potential of the population the axon belongs to [unit = V].
    tau_rise
        Rise time of the axonal kernel [unit = s].
    tau_decay
        Decay time of the axonal kernel [unit = s].
    max_firing_rate
        Maximum firing rate of the axon [unit = 1/s].
    activation_threshold
        Membrane potential threshold for activating normal firing [unit = V].
    activation_steepness
        Steepness of sigmoid representing the standard transfer function [unit = V].
    inactivation_threshold
        Membrane potential threshold for de-activating LTS firing [unit = V].
    inactivation_steepness
        Steepness of sigmoid representing the LTS bursts [unit = V].
    
    See Also
    --------
    :class:`BurstingAxon`: ... for a detailed description of the parameters, methods and attributes.
    
    References
    ----------
    """

    def __init__(self,
                 bin_size: float = 1e-3,
                 label: Optional[str] = None,
                 epsilon: float = 1e-10,
                 max_delay: Optional[float] = None,
                 resting_potential: float = -0.065,
                 tau_rise: float = 0.05,
                 tau_decay: float = 0.1,
                 max_firing_rate: float = 800.,
                 activation_threshold: float = -0.059,
                 activation_steepness: float = 670.,
                 inactivation_threshold: float = -0.081,
                 inactivation_steepness: float = 170.,
                 ) -> None:
        """Instantiates suffczynski axon.
        """

        # define kernel function attributes
        ###################################

        double_exp_args = {'tau_rise': tau_rise,
                           'tau_decay': tau_decay}
        sigmoid_args = {'inactivation_threshold': inactivation_threshold,
                        'inactivation_steepness': inactivation_steepness}

        # call super method
        ###################

        super().__init__(transfer_function=activation_sigmoid,
                         kernel_functions=[axon_exponential, inactivation_sigmoid],
                         bin_size=bin_size,
                         label=label,
                         epsilon=epsilon,
                         max_delay=max_delay,
                         max_firing_rate=max_firing_rate,
                         resting_potential=resting_potential,
                         kernel_function_args=[double_exp_args, sigmoid_args],
                         activation_threshold=activation_threshold,
                         activation_steepness=activation_steepness
                         )
