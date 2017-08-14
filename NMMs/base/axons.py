"""
Includes basic axon class and pre-parametrized axon subclasses
"""

import numpy as np

__author__ = "Daniel F. Rose, Richard Gast"
__status__ = "Development"


class Axon:
    """
    Basic class for axons. The key function is to transform average membrane potentials (from the soma)
    into average firing rates.

    :var axon_type: character string, indicates type of the axon
    :var max_firing_rate: scalar, determines maximum firing rate of axon
    :var membrane_potential_threshold: scalar, determines value for which sigmoidal transfer function value is 0.5
    :var sigmoid_steepness: scalar, determines steepness of the sigmoidal transfer function

    """

    def __init__(self, max_firing_rate, membrane_potential_threshold, sigmoid_steepness, axon_type=None):
        """
        Initializes basic axon that transforms average membrane potential into average firing rate
        via a sigmoidal transfer function.

        :param max_firing_rate: scalar, determines maximum firing rate of axon [unit = 1/s]
        :param membrane_potential_threshold: scalar, determines value for which sigmoidal transfer function value is 0.5
               [unit = V].
        :param sigmoid_steepness: scalar, determines steepness of the sigmoidal transfer function [unit = 1/V]
        :param axon_type: if not None, will be treated as type of axon (should be character string)

        """

        ##########################
        # check input parameters #
        ##########################

        assert max_firing_rate >= 0
        assert type(membrane_potential_threshold) is float
        assert sigmoid_steepness > 0

        ##############################
        # initialize axon parameters #
        ##############################

        self.axon_type = 'custom' if axon_type is None else axon_type
        self.max_firing_rate = max_firing_rate
        self.membrane_potential_threshold = membrane_potential_threshold
        self.sigmoid_steepness = sigmoid_steepness

    def compute_firing_rate(self, membrane_potential):
        """
        Method that computes average firing rate based on sigmoidal transfer function with previously set parameters

        :param membrane_potential: scalar, resembles current average membrane potential that is to be transferred into
               an average firing rate [unit = V].

        :return: scalar, average firing rate at axon [unit = 1/s]

        """

        return self.max_firing_rate / (1 + np.exp(self.sigmoid_steepness * (self.membrane_potential_threshold - membrane_potential)))


class KnoescheAxon(Axon):
    """
    Specific parametrization of generic axon, following the code of Thomas Knoesche
    """

    def __init__(self, max_firing_rate=5., membrane_potential_threshold=-0.069, sigmoid_steepness=555.56):
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

    def __init__(self, max_firing_rate=5., membrane_potential_threshold=0.060, sigmoid_steepness=555.56):
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
