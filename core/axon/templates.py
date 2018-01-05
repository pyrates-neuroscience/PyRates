"""Templates for specific axon parametrizations.
"""

from core.axon import SigmoidAxon, Axon  # type: ignore
import numpy as np

__author__ = "Daniel F. Rose, Richard Gast"
__status__ = "Development"


class JansenRitAxon(SigmoidAxon):
    """Sigmoid axon with parameters set according to [1]_.

    Parameters
    ----------
    max_firing_rate
        Default = 5.0 Hz. See documentation of parameter 'max_firing_rate' of :class:`SigmoidAxon`.
    membrane_potential_threshold
        Default = -0.06 V. See documentation of parameter 'membrane_potential_threshold' of :class:`SigmoidAxon`.
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
                 sigmoid_steepness: float = 555.56) -> None:
        """Instantiates sigmoid axon with Knoesche's parameters.
        """

        super().__init__(max_firing_rate=max_firing_rate,
                         membrane_potential_threshold=membrane_potential_threshold,
                         sigmoid_steepness=sigmoid_steepness,
                         axon_type='Knoesche')


class MoranAxon(Axon):
    """Sigmoidal axon as defined in [1]_.

    Parameters
    ----------
    max_firing_rate
    sigmoid_steepness
    membrane_potential_threshold
    adaption

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
                 adaption: float = 0.) -> None:

        ###########################################
        # sigmoidal transfer function (de-meaned) #
        ###########################################

        def sigmoid(membrane_potential, max_firing_rate, membrane_potential_threshold, sigmoid_steepness, adaption):
            return max_firing_rate / (1 + np.exp(sigmoid_steepness *
                                                 (membrane_potential_threshold - membrane_potential + adaption))) - \
                   max_firing_rate / (1 + np.exp(sigmoid_steepness * membrane_potential_threshold))

        ###################
        # call super init #
        ###################

        super().__init__(sigmoid,  # type: ignore
                         'Moran_Axon',
                         max_firing_rate=max_firing_rate,
                         membrane_potential_threshold=membrane_potential_threshold,
                         sigmoid_steepness=sigmoid_steepness,
                         adaption=adaption)
