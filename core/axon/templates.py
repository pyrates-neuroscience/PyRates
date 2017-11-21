"""
"""
from core.axon import Axon

__author__ = "Daniel F. Rose, Richard Gast"
__status__ = "Development"


class KnoescheAxon(Axon):
    """
    Specific parametrization of generic axon, following the code of Thomas Knoesche
    """

    def __init__(self, max_firing_rate: float = 5.,
                 membrane_potential_threshold: float = -0.069,
                 sigmoid_steepness: float = 555.56) -> None:
        """
        Initializes basic axon with Thomas Knoesche's sigmoid parameters.

        :param max_firing_rate: scalar, determines maximum firing rate of axon [unit = 1/s] (default = 5).
        :param membrane_potential_threshold: scalar, determines value for which sigmoidal transfer function value is 0.5
               [unit = V] (default = -0.069).
        :param sigmoid_steepness: scalar, determines steepness of the sigmoidal transfer function [unit = 1/V]
               (default = 555.56).

        """

        super(KnoescheAxon, self).__init__(max_firing_rate=max_firing_rate,
                                           membrane_potential_threshold=membrane_potential_threshold,
                                           sigmoid_steepness=sigmoid_steepness,
                                           axon_type='Knoesche')


class JansenRitAxon(Axon):
    """
    Specific parametrization of generic axon, following Jansen & Rit (1995)
    """

    def __init__(self, max_firing_rate: float = 5.,
                 membrane_potential_threshold: float = -0.069,
                 sigmoid_steepness: float = 555.56) -> None:
        """
        Initializes basic axon with Jansen & Rit's sigmoid parameters.

        :param max_firing_rate: scalar, determines maximum firing rate of axon [unit = 1/s] (default = 5).
        :param membrane_potential_threshold: scalar, determines value for which sigmoidal transfer function value is 0.5
               [unit = V] (default = 0.06).
        :param sigmoid_steepness: scalar, determines steepness of the sigmoidal transfer function [unit = 1/V]
               (default = 555.56).

        """

        super(JansenRitAxon, self).__init__(max_firing_rate=max_firing_rate,
                                            membrane_potential_threshold=membrane_potential_threshold,
                                            sigmoid_steepness=sigmoid_steepness,
                                            axon_type='JansenRit')