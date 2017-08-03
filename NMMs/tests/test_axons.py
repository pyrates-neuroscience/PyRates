"""
Includes a class with test functions for the axon class and a number of tests using that function. Should be run or
updated after each alteration of axons.py.
"""

from unittest import TestCase
import numpy as np
import sys
sys.path.append('/home/rgast/Documents/GitRepo/CBSMPG/NMMs/neural-mass-models/base/')
import axons as ax

__author__ = "Richard Gast"
__status__ = "Development"


#############################################
# function that calculates axon firing rate #
#############################################

def get_axon_firing_rate(membrane_potential, max_firing_rate=1., membrane_potential_threshold=0., sigmoid_steepness=1.):
    """
    Axon init that initializes axon instance with passed parameters.

    :param membrane_potential: array with 1 or 2 scalars representing the membrane potential that is to be
           transformed into a firing rate via the axon [unit = mV].
    :param max_firing_rate: tuple with 1 or 2 scalars determining maximum firing rate of axon [unit = firing rate].
    :param membrane_potential_threshold: tuple of 1 or 2 scalars determining value for which sigmoidal transfer
           function value is 0.5 [unit = mV].
    :param sigmoid_steepness: tuple of 1 or 2 scalars determining steepness of the sigmoidal transfer function
           [unit = mA].

    :return firing_rate: scalar or length 2 vector [unit = firing rate]

    """

    if type(max_firing_rate) is tuple:

        axon1 = ax.Axon(max_firing_rate[0], membrane_potential_threshold[0], sigmoid_steepness[0])
        axon2 = ax.Axon(max_firing_rate[1], membrane_potential_threshold[1], sigmoid_steepness[1])
        firing_rate = np.array([axon1.compute_firing_rate(membrane_potential),
                                axon2.compute_firing_rate(membrane_potential)])

    else:

        axon = ax.Axon(max_firing_rate, membrane_potential_threshold, sigmoid_steepness)
        firing_rate = axon.compute_firing_rate(membrane_potential)

    return firing_rate


#############################
# axon class test functions #
#############################

class TestAxon(TestCase):
    """
    Test class that includes test functions for the Axon class of axons.py.
    """

    def test_input(self):
        """
        Performs 3 tests:

            1. Tests whether firing rate at input equal to the membrane potential threshold equals 0.5 * max_firing_rate
            2. Tests whether firing rate at input greater than the membrane potential threshold, with positive maximum
               firing rate, is larger than firing rate at input equal to membrane potential threshold.
            3. Tests whether firing rate at input smaller than the  membrane potential threshold and positive maximum
               firing rate, is smaller than firing rate at input equal to membrane potential threshold.
        """

        ##################
        # set parameters #
        ##################

        max_firing_rate = 1.
        sigmoid_steepness = 1.
        membrane_potential_threshold = 0.

        ####################
        # get firing rates #
        ####################

        membrane_potential = membrane_potential_threshold
        firing_rate_1 = get_axon_firing_rate(membrane_potential, max_firing_rate, membrane_potential_threshold,
                                           sigmoid_steepness)

        membrane_potential = membrane_potential_threshold + 1.
        firing_rate_2 = get_axon_firing_rate(membrane_potential, max_firing_rate, membrane_potential_threshold,
                                             sigmoid_steepness)

        membrane_potential = membrane_potential_threshold - 1.
        firing_rate_3 = get_axon_firing_rate(membrane_potential, max_firing_rate, membrane_potential_threshold,
                                             sigmoid_steepness)
        #################
        # perform tests #
        #################

        self.assertEqual(firing_rate_1, 0.5*max_firing_rate)
        self.assertLess(firing_rate_1, firing_rate_2)
        self.assertGreater(firing_rate_1, firing_rate_3)

    def test_maximum_firing_rate(self):
        """
        Performs 3 tests:

            1. Tests whether a zero maximum firing rate leads to a zero firing rate output.
            2. Tests whether the greater of two positive maximum firing rates leads to the greater firing rate output
               given a positive sigmoid steepness.
            3. Tests whether a negative maximum firing rate leads to an assertion error or thrown exception

        """

        ##################
        # set parameters #
        ##################

        sigmoid_steepness = 1.
        membrane_potential_threshold = 0.
        membrane_potential = membrane_potential_threshold

        ####################
        # get firing rates #
        ####################

        max_firing_rate = 0.
        firing_rate_1 = get_axon_firing_rate(membrane_potential, max_firing_rate, membrane_potential_threshold,
                                             sigmoid_steepness)

        max_firing_rate = 1.
        firing_rate_2 = get_axon_firing_rate(membrane_potential, max_firing_rate, membrane_potential_threshold,
                                             sigmoid_steepness)

        max_firing_rate = 2.
        firing_rate_3 = get_axon_firing_rate(membrane_potential, max_firing_rate, membrane_potential_threshold,
                                             sigmoid_steepness)

        #################
        # perform tests #
        #################

        self.assertEqual(firing_rate_1, 0.)
        self.assertLess(firing_rate_2, firing_rate_3)
        self.assertRaises(AssertionError, get_axon_firing_rate, membrane_potential,
                          membrane_potential_threshold=membrane_potential_threshold,
                          sigmoid_steepness=sigmoid_steepness,
                          max_firing_rate=-1.)

    def test_membrane_potential_threshold(self):
        """
        Performs 1 test:

            1. Tests whether the smaller of two membrane potential thresholds leads to a higher firing rate output

        """

        ##################
        # set parameters #
        ##################

        sigmoid_steepness = 1.
        max_firing_rate = 1.
        membrane_potential = 0.

        ####################
        # get firing rates #
        ####################

        membrane_potential_threshold = 0.
        firing_rate_1 = get_axon_firing_rate(membrane_potential, max_firing_rate, membrane_potential_threshold,
                                             sigmoid_steepness)

        membrane_potential_threshold = -1.
        firing_rate_2 = get_axon_firing_rate(membrane_potential, max_firing_rate, membrane_potential_threshold,
                                             sigmoid_steepness)

        #################
        # perform tests #
        #################

        self.assertLess(firing_rate_1, firing_rate_2)

    def test_sigmoid_steepness(self):
        """
        Performs 3 tests:

            1. Tests whether the smaller of two sigmoid steepness values leads to a lower firing rate output given the
               same input (greater than membrane potential threshold)
            2. Tests whether a negative sigmoid steepness leads to an assertion error
            3. Tests whether a zero sigmoid steepness leads to an assertion error

        """

        ##################
        # set parameters #
        ##################

        max_firing_rate = 1.
        membrane_potential = 1.
        membrane_potential_threshold = 0.

        ####################
        # get firing rates #
        ####################

        sigmoid_steepness = 1.
        firing_rate_1 = get_axon_firing_rate(membrane_potential, max_firing_rate, membrane_potential_threshold,
                                             sigmoid_steepness)

        sigmoid_steepness = 2.
        firing_rate_2 = get_axon_firing_rate(membrane_potential, max_firing_rate, membrane_potential_threshold,
                                             sigmoid_steepness)

        #################
        # perform tests #
        #################

        self.assertLess(firing_rate_1, firing_rate_2)
        self.assertRaises(AssertionError, get_axon_firing_rate, membrane_potential,
                          max_firing_rate=max_firing_rate,
                          membrane_potential_threshold=membrane_potential_threshold,
                          sigmoid_steepness=-1.)
        self.assertRaises(AssertionError, get_axon_firing_rate, membrane_potential,
                          max_firing_rate=max_firing_rate,
                          membrane_potential_threshold=membrane_potential_threshold,
                          sigmoid_steepness=0.)

