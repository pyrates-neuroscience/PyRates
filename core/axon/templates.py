"""Templates for specific axon parametrizations.
"""

from core.axon import SigmoidAxon, Axon
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
        Default = -0.069 V. See documentation of parameter 'membrane_potential_threshold' of :class:`SigmoidAxon`.
    sigmoid_steepness
        Default = 555.56 Hz. See documentation of parameter 'sigmoid_steepness' of :class:`SigmoidAxon`.

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
                 membrane_potential_threshold: float = -0.069,
                 sigmoid_steepness: float = 555.56) -> None:
        """Instantiates sigmoid axon with Jansen & Rit's parameters.
        """

        super().__init__(max_firing_rate=max_firing_rate,
                         membrane_potential_threshold=membrane_potential_threshold,
                         sigmoid_steepness=sigmoid_steepness,
                         axon_type='JansenRit')


class MoranAxon(Axon):
    """
    """

    def __init__(self,
                 max_firing_rate: float = 1.,
                 sigmoid_steepness: float = 200.,
                 membrane_potential_threshold: float = 0.001):

        def sigmoid(membrane_potential, max_firing_rate, membrane_potential_threshold, sigmoid_steepness):
            return max_firing_rate / (1 + np.exp(sigmoid_steepness *
                                                 (membrane_potential_threshold - membrane_potential))) - \
                   max_firing_rate / (1 + np.exp(sigmoid_steepness * membrane_potential_threshold))

        super().__init__(sigmoid,
                         'Moran_Axon',
                         max_firing_rate=max_firing_rate,
                         membrane_potential_threshold=membrane_potential_threshold,
                         sigmoid_steepness=sigmoid_steepness)
