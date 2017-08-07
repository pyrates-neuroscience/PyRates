"""
Includes a class with test functions for the axon class and a number of tests using that function. Should be run or
updated after each alteration of axons.py.
"""

import unittest
import numpy as np
import sys
sys.path.append('../base')
import nmm_network

__author__ = "Richard Gast & Konstantin Weise"
__status__ = "Test"

class TestNMM(unittest.TestCase):
    """
    Test class that includes test functions for the Axon class of axons.py.
    """

    def test_0_input(self):
        """

        """

        print('Running Jansen Rit NMM test ...')

        # set parameters
        population_labels = ['PC', 'EIN', 'IIN']
        connections = np.zeros((3,3,2))
        # AMPA connections (excitatory)
        connections[:, :, 0] = [[0, 0.8 * 135, 0], [135, 0, 0], [0.25 * 135, 0, 0]]
        # GABA connections (inhibitory)
        connections[:, :, 1] = [[0, 0, 0.25 * 135], [0, 0, 0], [0, 0, 0]]

        print('done!')


if __name__ == '__main__':
    unittest.main()