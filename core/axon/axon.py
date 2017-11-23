"""
Basic parametrized axon class that can compute average firing rates from an average membrane potential
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

__author__ = "Richard Gast, Daniel F. Rose"
__status__ = "Development"


class Axon(object):
    """
    Basic class that behaves like the axon hillok. The key function is to transform average membrane potentials
    (from the soma) into average firing rates.

    :var axon_type: character string, indicates type of the axon
    :var max_firing_rate: scalar, determines maximum firing rate of axon
    :var membrane_potential_threshold: scalar, determines value for which sigmoidal transfer function value is 0.5
    :var sigmoid_steepness: scalar, determines steepness of the sigmoidal transfer function

    """

    def __init__(self, max_firing_rate: float, membrane_potential_threshold: float, sigmoid_steepness: float,
                 axon_type: Optional[str] = None) -> None:
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
        assert type(membrane_potential_threshold) is float  # fixme: could be removed
        assert sigmoid_steepness > 0

        ##############################
        # initialize axon parameters #
        ##############################

        self.axon_type = 'custom' if axon_type is None else axon_type
        self.max_firing_rate = max_firing_rate
        self.membrane_potential_threshold = membrane_potential_threshold
        self.sigmoid_steepness = sigmoid_steepness

    def compute_firing_rate(self, membrane_potential: np.float64) -> np.float64:
        """
        Method that computes average firing rate based on sigmoidal transfer function with previously set parameters

        :param membrane_potential: scalar, resembles current average membrane potential that is to be transferred into
               an average firing rate [unit = V].

        :return: scalar, average firing rate at axon [unit = 1/s]

        """
        # fixme: Possibly restrict to exclusively np.float64 (or float)
        return self.max_firing_rate / (1 + np.exp(self.sigmoid_steepness *
                                                  (self.membrane_potential_threshold - membrane_potential)))

    def plot_transfer_function(self, membrane_potentials: Optional[np.ndarray] = None,
                               epsilon: float = 1e-4,
                               bin_size: float = 0.001,
                               create_plot: bool = True,
                               fig=None):
        """
        Creates figure of the sigmoidal function transforming membrane potentials into output firing rates.

        :param membrane_potentials: Can be vector of membrane potentials for which to plot the function value
               [unit = V] (default = None).
        :param epsilon: scalar, determines min/max function value to plot, if membrane_potentials were not passed.
               Min = 0 + epsilon, max = max_firing_rate - epsilon [unit = 1/s] (default = 1e-4).
        :param bin_size: scalar, determines the size of the steps between the membrane potentials at which the firing
               rate is evaluated. Only necessary if no membrane potentials are passed [unit = V] (default = 0.001).
        :param create_plot: If false, plot will not be shown (default = True).
        :param fig: figure handle that can be passed optionally (default = None).

        :return: figure handle

        """
        # TODO: annotate plotting function
        ##########################
        # check input parameters #
        ##########################

        assert membrane_potentials is None or type(membrane_potentials) is np.ndarray
        assert epsilon >= 0
        assert bin_size >= 0

        ##########################
        # calculate firing rates #
        ##########################

        if membrane_potentials is None:

            # start at membrane potential threshold and successively calculate the firing rate & reduce the membrane
            # potential until the firing rate is smaller or equal to epsilon
            membrane_potential = list()
            membrane_potential.append(self.membrane_potential_threshold)
            firing_rates = list()
            firing_rates.append(self.compute_firing_rate(membrane_potential[-1]))
            while firing_rates[-1] >= epsilon:
                membrane_potential.append(membrane_potential[-1] - bin_size)
                firing_rates.append(self.compute_firing_rate(membrane_potential[-1]))
            firing_rates_1 = np.array(firing_rates)
            firing_rates_1 = np.flipud(firing_rates_1)
            membrane_potential_1 = np.array(membrane_potential)
            membrane_potential_1 = np.flipud(membrane_potential_1)

            # start at membrane potential threshold and successively calculate the firing rate & increase the membrane
            # potential until the firing rate is greater or equal to max_firing_rate - epsilon
            membrane_potential = list()
            membrane_potential.append(self.membrane_potential_threshold + bin_size)
            firing_rates = list()
            firing_rates.append(self.compute_firing_rate(membrane_potential[-1]))
            while firing_rates[-1] <= self.max_firing_rate - epsilon:
                membrane_potential.append(membrane_potential[-1] + bin_size)
                firing_rates.append(self.compute_firing_rate(membrane_potential[-1]))
            firing_rates_2 = np.array(firing_rates)
            membrane_potential_2 = np.array(membrane_potential)

            # concatenate the resulting membrane potentials and firing rates
            firing_rates = np.concatenate((firing_rates_1, firing_rates_2))
            membrane_potentials = np.concatenate((membrane_potential_1, membrane_potential_2))

        else:
            # fixme: inconsistency in type of membrane_potentials
            firing_rates = self.compute_firing_rate(membrane_potentials)

        ##############################################
        # plot firing rates over membrane potentials #
        ##############################################

        if fig is None:
            fig = plt.figure('Axonal Transfer Function')

        plt.hold('on')
        plt.plot(membrane_potentials, firing_rates)
        plt.hold('off')
        plt.xlabel('membrane potential [V]')
        plt.ylabel('firing rate [Hz]')
        plt.title('Wave-To-Pulse Function')

        if create_plot:
            fig.show()

        return fig
