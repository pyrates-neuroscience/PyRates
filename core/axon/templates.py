"""Templates for specific axon parametrizations.
"""

import numpy as np
from core.axon import SigmoidAxon, Axon, BurstingAxon  # type: ignore
from core.axon import parametric_sigmoid
from core.synapse import double_exponential
from typing import Optional, List

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
                         axon_type='Knoesche')


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
                         axon_type='JansenRit')


def moran_sigmoid(membrane_potential,
                  max_firing_rate,
                  membrane_potential_threshold,
                  sigmoid_steepness,
                  adaptation):
    """

    Parameters
    ----------
    membrane_potential
        See above parameter description.
    max_firing_rate
        See above parameter description.
    membrane_potential_threshold
        See above parameter description.
    sigmoid_steepness
        See above parameter description.
    adaptation
        See above parameter description.

    Returns
    -------
    float
        firing rate [unit = 1/s]

    """
    return max_firing_rate / (1 + np.exp(sigmoid_steepness *
                                         (membrane_potential_threshold - membrane_potential + adaptation))) - \
           max_firing_rate / (1 + np.exp(sigmoid_steepness * membrane_potential_threshold))


class MoranAxon(Axon):
    """Sigmoidal axon as defined in [1]_.

    Parameters
    ----------
    max_firing_rate
        Default = 1.0 Hz. See documentation of parameter 'max_firing_rate' of :class:`SigmoidAxon`.
    sigmoid_steepness
        Default = 2000 Hz. See documentation of parameter 'sigmoid_steepness' of :class:`SigmoidAxon`.
    membrane_potential_threshold
        Default = 0.001 V. See documentation of parameter 'membrane_potential_threshold' of :class:`SigmoidAxon`.
    adaptation
        Default = 0. V. Added to incoming membrane potential of transfer function to account for spike frequency
        adaptation.

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
                 membrane_potential_threshold: float = 0.001,
                 adaptation: float = 0.) -> None:
        """Instantiate moran axon.
        """

        super().__init__(moran_sigmoid,
                         'Moran_Axon',
                         max_firing_rate=max_firing_rate,
                         membrane_potential_threshold=membrane_potential_threshold,
                         sigmoid_steepness=sigmoid_steepness,
                         adaptation=adaptation)


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
    lts_threshold
        Membrane potential threshold for de-activating LTS firing [unit = V].
    lts_steepness
        Steepness of sigmoid representing the LTS bursts [unit = V].
    
    See Also
    --------
    :class:`BurstingAxon`: ... for a detailed description of the parameters, methods and attributes.
    
    References
    ----------
    """

    def __init__(self,
                 bin_size: float = 1e-3,
                 axon_type: Optional[str] = None,
                 epsilon: float = 1e-10,
                 max_delay: Optional[float] = None,
                 resting_potential: float = -0.065,
                 tau_rise: float = 0.05,
                 tau_decay: float = 0.1,
                 max_firing_rate: float = 800.,
                 activation_threshold: float = -0.059,
                 activation_steepness: float = 670.,
                 lts_threshold: float = -0.081,
                 lts_steepness: float = 170.,
                 ) -> None:
        """Instantiates suffczynski axon.
        """

        # define kernel function attributes
        ###################################

        double_exp_args = {'tau_rise': tau_rise,
                           'tau_decay': tau_decay}
        sigmoid_args = {'membrane_potential_threshold': lts_threshold,
                        'sigmoid_steepness': lts_steepness,
                        'max_firing_rate': 1.}

        # call super method
        ###################

        super().__init__(transfer_function=parametric_sigmoid,
                         kernel_functions=[double_exponential, parametric_sigmoid],
                         bin_size=bin_size,
                         axon_type=axon_type,
                         epsilon=epsilon,
                         max_delay=max_delay,
                         resting_potential=resting_potential,
                         kernel_function_args=[double_exp_args, sigmoid_args],
                         membrane_potential_threshold=activation_threshold,
                         sigmoid_steepness=activation_steepness,
                         max_firing_rate=max_firing_rate
                         )
