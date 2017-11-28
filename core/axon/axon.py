"""Basic axon class.

This module includes a basic axon parametrized class that can calculate average firing rates from average membrane
potentials. Its behavior approximates the average axon hillok of a homogenous neural population.

Requires
--------
matplotlib
numpy
typing

"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

__author__ = "Richard Gast, Daniel F. Rose"
__status__ = "Development"


class Axon(object):
    """Basic axon class. Represents average firing behavior of axon hillok of a neural population.

    Parameters
    ----------
    max_firing_rate
        Determines maximum firing rate of axon [unit = 1/s].
    membrane_potential_threshold
        Determines membrane potential for which output firing rate is half the maximum firing rate [unit = V].
    sigmoid_steepness
        Determines steepness of the sigmoidal transfer function mapping membrane potential to firing rate [unit = 1/V].
    axon_type
        Name of axon type.

    Attributes
    ----------
    max_firing_rate
        See parameter description.
    membrane_potential_threshold
        See parameter description.
    sigmoid_steepness
        See parameter description.
    axon_type
        See parameter description.

    Methods
    -------
    compute_firing_rate(membrane_potential)
        Calculates average firing rate from average membrane potential.
        See Also docstring of method itself.
    plot_transfer_function(membrane_potential=None, epsilon=1e-4, bin_size=1e-3, create_plot=True, fig=None)
        Creates plot of average firing rates over membrane potentials.
        See Also docstring of method itself.

    Notes
    -----
    Consider building generic, non-parametric base class, instead of the specific one described in [1]_

    References
    ----------
    .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
       of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

    """

    def __init__(self, max_firing_rate: float, membrane_potential_threshold: float, sigmoid_steepness: float,
                 axon_type: Optional[str] = None) -> None:

        ##########################
        # check input parameters #
        ##########################

        if max_firing_rate < 0:
            raise ValueError('Maximum firing rate cannot be negative.')

        if sigmoid_steepness < 0:
            raise ValueError('Sigmoid steepness cannot be negative.')

        ##############################
        # initialize axon parameters #
        ##############################

        self.axon_type = 'custom' if axon_type is None else axon_type
        self.max_firing_rate = max_firing_rate
        self.membrane_potential_threshold = membrane_potential_threshold
        self.sigmoid_steepness = sigmoid_steepness

    def compute_firing_rate(self, membrane_potential: float) -> float:
        """Computes average firing rate from membrane potential based on sigmoidal transfer function.

        Parameters
        ----------
        membrane_potential
            current average membrane potential of neural mass [unit = V].

        Returns
        -------
        float
            average firing rate at axon hillok [unit = 1/s]

        """

        # fixme: Possibly restrict to exclusively np.float64 (or float)

        return self.max_firing_rate / (1 + np.exp(self.sigmoid_steepness *
                                                  (self.membrane_potential_threshold - membrane_potential)))

    def plot_transfer_function(self, membrane_potentials: Optional[np.ndarray] = None,
                               epsilon: float = 1e-4,
                               bin_size: float = 0.001,
                               create_plot: bool = True,
                               fig=None):
        """Creates figure of the sigmoidal function transforming membrane potentials into output firing rates.

        Parameters
        ----------
        membrane_potentials
            Membrane potential values for which to plot transfer function value [unit = V] (default = None).
        epsilon
            Determines min/max function value to plot, if membrane_potentials is None.
            Min = 0 + epsilon, max = max_firing_rate - epsilon [unit = 1/s] (default = 1e-4).
        bin_size
            Determines size of steps between membrane potentials at which function is evaluated. Not necessary if
            membrane_potentials is None [unit = V] (default = 0.001).
        create_plot
            If false, plot will bot be shown (default = True).
        fig
            If passed, plot will be created in respective figure (default = None).

        Returns
        -------
        :obj:`figure handle`
            Handle of the newly created or updated figure.

        """

        ##########################
        # check input parameters #
        ##########################

        if epsilon < 0:
            raise ValueError('Epsilon cannot be negative. See parameter description for further information.')

        if bin_size < 0:
            raise ValueError('Bin size cannot be negative. See parameter description for further information.')

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

        # check whether new figure needs to be created
        if fig is None:
            fig = plt.figure('Wave-To-Pulse-Function')

        # plot firing rates over membrane potentials
        plt.hold('on')
        plt.plot(membrane_potentials, firing_rates)
        plt.hold('off')

        # set figure labels
        plt.xlabel('membrane potential [V]')
        plt.ylabel('firing rate [Hz]')
        plt.title('Axon Hillok Transfer Function')

        # show plot
        if create_plot:
            fig.show()

        return fig
