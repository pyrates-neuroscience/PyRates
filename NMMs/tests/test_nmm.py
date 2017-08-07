"""
Includes a class with test functions for the axon class and a number of tests using that function. Should be run or
updated after each alteration of axons.py.
"""

import unittest
import numpy as np
import nmm_network
from matplotlib.pyplot import *

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
        ################################################################################################################

        # simulations parameters
        simulation_time = 1     # s
        cutoff_time = 0        # s
        step_size = 5.0e-4      # s
        store_step = 1

        # populations
        population_labels = ['PC', 'EIN', 'IIN']
        N = len(population_labels)
        n_synapses = 2

        # synapses
        connections = np.zeros((N, N, n_synapses))

        # AMPA connections (excitatory)
        connections[:, :, 0] = [[0, 0.8 * 135, 0], [135, 0, 0], [0.25 * 135, 0, 0]]

        # GABA-A connections (inhibitory)
        synaptic_kernel_length = 100 # in time steps

        connections[:, :, 1] = [[0, 0, 0.25 * 135], [0, 0, 0], [0, 0, 0]]

        ampa_dict = {'efficacy': 1.273 * 3e-13,     # A
                     'tau_decay': 0.006,            # s
                     'tau_rise': 0.0006,            # s
                     'conductivity_based': False}

        gaba_a_dict = {'efficacy': 1.273 * -1e-12,  # A
                       'tau_decay': 0.02,           # s
                       'tau_rise': 0.0004,          # s
                       'conductivity_based': False}

        synapse_params = [ampa_dict, gaba_a_dict]

        # axon
        axon_dict = {'max_firing_rate' : 5,                   # 1
                     'membrane_potential_threshold' : -0.069, # V
                     'sigmoid_steepness' : 556}               # 1/V
        axon_params = [axon_dict for i in range(N)]

        distances = np.zeros((N, N))

        init_states = np.zeros((N, n_synapses))

        # synaptic inputs
        start_stim = 300.0*1e3  # s
        len_stim = 50.0*1e3     # s
        mag_stim = 300.0        # 1/s

        synaptic_inputs = np.zeros((int(simulation_time/step_size),N,n_synapses))
        synaptic_inputs[int(start_stim/step_size):int(start_stim/step_size+len_stim/step_size),:,0] = mag_stim

        # initialize
        ################################################################################################################
        nmm = nmm_network.NeuralMassModel(connections = connections,
                                          population_labels = population_labels,
                                          step_size = step_size,
                                          synaptic_kernel_length = synaptic_kernel_length,
                                          distances = distances,
                                          synapse_params = synapse_params,
                                          axon_params = axon_params,
                                          init_states = init_states)

        # run
        ################################################################################################################
        nmm.run(synaptic_inputs = synaptic_inputs,
                simulation_time = simulation_time,
                cutoff_time = cutoff_time,
                store_step = store_step)

        # plot
        ################################################################################################################
        figure()
        plot(np.squeeze(nmm.states))
        show()

        print('done!')

if __name__ == '__main__':
    unittest.main()