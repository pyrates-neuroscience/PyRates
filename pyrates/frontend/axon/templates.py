"""Templates for specific axon parametrizations.
"""

# external packages
from typing import Optional
import tensorflow as tf

# pyrates internal imports
from pyrates.axon import SigmoidAxon, BurstingAxon, PlasticSigmoidAxon  # type: ignore
from pyrates.utility import activation_sigmoid, inactivation_sigmoid
from pyrates.utility import axon_exponential

# meta infos
__author__ = "Richard Gast, Daniel F. Rose"
__status__ = "Development"


########################
# Leaky-Capacitor axon #
########################


class KnoescheAxon(SigmoidAxon):
    """Sigmoid axon with parameters set according to Thomas Knoesche's document.

    See Also
    --------
    :class:`SigmoidAxon`: Detailed description of parameters.
    :class:`Axon`: Detailed description of attributes and methods.

    """

    def __init__(self,
                 max_firing_rate: float = 5.,
                 firing_threshold: float = -0.069,
                 slope: float = 560.0
                 ) -> None:
        """Instantiates sigmoid axon with T. Knoesche's parameters.
        """

        super().__init__(max_firing_rate=max_firing_rate,
                         firing_threshold=firing_threshold,
                         slope=slope,
                         key='LC_axon')


class PlasticKnoescheAxon(PlasticSigmoidAxon):
    """Sigmoid axon with parameters set according to Thomas Knoesche's document. Allows for spike-frequency adaptation.

    See Also
    --------
    :class:`PlasticSigmoidAxon`: Detailed description of parameters.
    :class:`Axon`: Detailed description of attributes and methods.

    """

    def __init__(self,
                 max_firing_rate: float = 5.,
                 slope: float = 560.,
                 firing_threshold: float = -0.069
                 ) -> None:
        """Instantiate plastic sigmoidal axon.
        """

        super().__init__(key='LC_axon_plastic',
                         max_firing_rate=max_firing_rate,
                         firing_threshold=firing_threshold,
                         slope=slope,
                         normalize=False)


########################
# JansenRit-type axons #
########################


class JansenRitAxon(SigmoidAxon):
    """Sigmoid axon with parameters set according to [1]_.

    See Also
    --------
    :class:`SigmoidAxon`: Detailed description of parameters.
    :class:`Axon`: Detailed description of attributes and methods.

    References
    ----------
    .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
       of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

    """

    def __init__(self,
                 max_firing_rate: float = 5.,
                 firing_threshold: float = 0.006,
                 slope: float = 560.,
                 tf_graph: Optional[tf.Graph] = None,
                 key: str = 'JR_axon'
                 ) -> None:
        """Instantiates sigmoid axon with Jansen & Rit's parameters.
        """

        super().__init__(max_firing_rate=max_firing_rate,
                         firing_threshold=firing_threshold,
                         slope=slope,
                         tf_graph=tf_graph,
                         key=key)


class MoranAxon(PlasticSigmoidAxon):
    """Sigmoidal axon as defined in [1]_.

    See Also
    --------
    :class:`PlasticSigmoidAxon`: Detailed description of parameters.
    :class:`Axon`: Detailed description of attributes and methods.

    References
    ----------
    .. [1] R.J. Moran, S.J. Kiebel, K.E. Stephan, R.B. Reilly, J. Daunizeau & K.J. Friston, "A Neural Mass Model of
       Spectral Responses in Electrophysiology" NeuroImage, vol. 37, pp. 706-720, 2007.

    """

    def __init__(self,
                 max_firing_rate: float = 1.,
                 slope: float = 2000.,
                 firing_threshold: float = 0.001
                 ) -> None:
        """Instantiate moran axon.
        """

        super().__init__(max_firing_rate=max_firing_rate,
                         firing_threshold=firing_threshold,
                         slope=slope,
                         normalize=True,
                         key='Moran_axon')


#################
# bursting axon #
#################


class SuffczynskiAxon(BurstingAxon):
    """Bursting axon including low-threshold spikes as described in [1]_.
    
    Parameters
    ----------
    bin_size
        time window one bin of the axonal kernel is representing [unit = s].
    key
        See description of parameter `key` of :class:`Axon`.
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
        Firing threshold for activating normal firing [unit = V].
    activation_slope
        Slope of sigmoid representing the standard transfer function [unit = V].
    inactivation_threshold
        Firing threshold for de-activating LTS firing [unit = V].
    inactivation_slope
        Slope of sigmoid representing the LTS bursts [unit = V].
    
    See Also
    --------
    :class:`BurstingAxon`: ... for a detailed description of the parameters, methods and attributes.
    
    References
    ----------
    .. [1] P. Suffczynski, S. Kalitzin, G. Pfurtscheller & F.H. Lopes da Silva, "Computational model of thalamo-cortical
       networks: dynamical control of alpha rhythms in relation to focal attention" International Journal of
       Psychophysiology, vol. 43(1), pp. 25-40, 2001.

    """

    def __init__(self,
                 bin_size: float = 1e-3,
                 key: Optional[str] = None,
                 epsilon: float = 1e-10,
                 max_delay: Optional[float] = None,
                 resting_potential: float = -0.065,
                 tau_rise: float = 0.05,
                 tau_decay: float = 0.1,
                 max_firing_rate: float = 800.,
                 activation_threshold: float = -0.059,
                 activation_slope: float = 670.,
                 inactivation_threshold: float = -0.081,
                 inactivation_slope: float = 170.,
                 ) -> None:
        """Instantiates suffczynski axon.
        """

        # define kernel function attributes
        ###################################

        double_exp_args = {'tau_rise': tau_rise,
                           'tau_decay': tau_decay}
        sigmoid_args = {'inactivation_threshold': inactivation_threshold,
                        'inactivation_slope': inactivation_slope}

        # call super method
        ###################

        super().__init__(transfer_function=activation_sigmoid,
                         kernel_functions=[axon_exponential, inactivation_sigmoid],
                         bin_size=bin_size,
                         key=key,
                         epsilon=epsilon,
                         max_delay=max_delay,
                         max_firing_rate=max_firing_rate,
                         resting_potential=resting_potential,
                         kernel_function_args=[double_exp_args, sigmoid_args],
                         activation_threshold=activation_threshold,
                         activation_slope=activation_slope
                         )
