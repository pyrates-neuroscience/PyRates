"""
Includes basic axon class and pre-parametrized axon subclasses
"""

import numpy as np

__author__ = "Daniel F. Rose, Richard Gast"
__status__ = "Development"


class Axon(object):
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

        :param max_firing_rate: scalar, determines maximum firing rate of axon [unit = firing rate]
        :param membrane_potential_threshold: scalar, determines value for which sigmoidal transfer function value is 0.5
               [unit = mV].
        :param sigmoid_steepness: scalar, determines steepness of the sigmoidal transfer function [unit = 1/mV]
        :param axon_type: if not None, will be treated as type of axon (should be character string)

        """

        ##########################
        # check input parameters #
        ##########################

        assert max_firing_rate >= 0
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
               an average firing rate

        :return: scalar, average firing rate at axon

        """

        return self.max_firing_rate / (1 + np.exp(self.sigmoid_steepness * (self.membrane_potential_threshold - membrane_potential)))


class KnoescheAxon(Axon):
    """
    Specific parametrization of generic axon, following the code of Thomas Knoesche
    """

    def __init__(self, max_firing_rate=5, membrane_potential_threshold=-69.0, sigmoid_steepness=0.56):
        """
        Initializes basic axon with Thomas Knoesche's sigmoid parameters.

        :param max_firing_rate: scalar, determines maximum firing rate of axon [unit = firing rate] (default = 5).
        :param membrane_potential_threshold: scalar, determines value for which sigmoidal transfer function value is 0.5
               [unit = mV] (default = -69).
        :param sigmoid_steepness: scalar, determines steepness of the sigmoidal transfer function [unit = 1/mV]
               (default = 0.56).

        """

        super(KnoescheAxon, self).__init__(max_firing_rate, membrane_potential_threshold, sigmoid_steepness,
                                           axon_type='Knoesche')


class JansenRitAxon(Axon):
    """
    Specific parametrization of generic axon, following Jansen & Rit (1995)
    """

    def __init__(self, max_firing_rate=5, membrane_potential_threshold=6, sigmoid_steepness=0.56):
        """
        Initializes basic axon with Jansen & Rit's sigmoid parameters.

        :param max_firing_rate: scalar, determines maximum firing rate of axon [unit = firing rate] (default = 5).
        :param membrane_potential_threshold: scalar, determines value for which sigmoidal transfer function value is 0.5
               [unit = mV] (default = 6).
        :param sigmoid_steepness: scalar, determines steepness of the sigmoidal transfer function [unit = 1/mV]
               (default = 0.56).

        """

        super(JansenRitAxon, self).__init__(max_firing_rate, membrane_potential_threshold, sigmoid_steepness,
                                            axon_type='JansenRit')
