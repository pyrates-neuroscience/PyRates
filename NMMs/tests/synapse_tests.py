"""
Includes test functions for the synapse class and a number of tests using these functions. Should be run or updated
after each alteration of synapses.py.
"""

from unittest import TestCase
import numpy as np
import sys
sys.path.append('../base')
import synapses as syn


__author__ = "Richard Gast"
__status__ = "Development"

# TODO: Implement population class test functions
# TODO: Implement population class tests based on population parametrizations in literature
# TODO: Implement nmm network class test functions
# TODO: Implement nmm network class tests based on established neural mass models like Jansen & Rit (1995)


################################
# synapse class test functions #
################################

class TestSynapse(TestCase):
    """
    Test class that includes test functions for the Synapse class of synapses.py.
    """



#######################
# synapse class tests #
#######################

def synaptic_kernel_value_test(t, efficiency=1., tau_decay=0.01, tau_rise=0.001, conductivity_based=False,
                               synaptic_kernel_target=None):
    """
    Test function that tests properties of the synaptic kernel function.

    :param t: scalar or vector, resembles time(s) at which synaptic kernel should be evaluated [unit = s].
    :param efficiency: scalar, determines synaptic efficiency [unit = mA or mS/m].
    :param tau_decay: scalar, determines how fast synaptic kernel decays [unit = s].
    :param tau_rise: scalar, determines how fast synaptic kernel rises [unit = s].
    :param conductivity_based: if true, synaptic kernel unit will be conductivity (mS/m), else current (mA).
    :param synaptic_kernel_target: scalar or vector of length of t, resembles target synaptic kernel values for the
           passed times t at which to evaluate the synaptic kernel [unit = mA or mS/m]

    :return kernel_value or kernel_value == synaptic_kernel_target

    """

    ######################
    # initialize synapse #
    ######################

    step_size = 0.001
    kernel_threshold = 1e-4
    synapse_instance = syn.Synapse(efficiency, tau_decay, tau_rise, step_size, kernel_threshold, conductivity_based)

    #########################################
    # evaluate membrane potential at time t #
    #########################################

    kernel_value = synapse_instance.evaluate_kernel(t)

    return kernel_value if not synaptic_kernel_target else kernel_value == synaptic_kernel_target


def synaptic_kernel_test(step_size, kernel_threshold, efficiency=1., tau_decay=0.01, tau_rise=0.001,
                         conductivity_based=False):
    """
    Test function that tests properties of the synaptic kernel function.

    :param step_size: scalar, indicates size of the time-steps between kernel bins [unit = s].
    :param kernel_threshold: scalar, indicates threshold for which kernel values will be assumed to be non-different
           from zero [unit = mA or mS/m].
    :param efficiency: scalar, determines synaptic efficiency [unit = mA or mS/m].
    :param tau_decay: scalar, determines how fast synaptic kernel decays [unit = s].
    :param tau_rise: scalar, determines how fast synaptic kernel rises [unit = s].
    :param conductivity_based: if true, synaptic kernel unit will be conductivity (mS/m), else current (mA).

    :return: kernel

    """

    ######################
    # initialize synapse #
    ######################

    synapse_instance = syn.Synapse(efficiency, tau_decay, tau_rise, step_size, kernel_threshold, conductivity_based)

    #########################
    # build synaptic kernel #
    #########################

    synapse_instance.build_kernel()

    return synapse_instance.synaptic_kernel


def synaptic_current_test(x, step_size=0.0001, kernel_threshold=1e-4, efficiency=1., tau_decay=0.01, tau_rise=0.001,
                          conductivity_based=False, reversal_potential=-70.0, membrane_potential=None,
                          synaptic_current_target=None):
    """
    Test function that tests the synaptic current evaluated by a certain synapse instance parametrization.

    :param x: vector, resembles synaptic inputs for which synaptic current should be evaluated [unit = firing rate].
    :param step_size: scalar, indicates size of the time-steps between kernel bins [unit = s] (default = 0.0001).
    :param kernel_threshold: scalar, indicates threshold for which kernel values will be assumed to be non-different
           from zero [unit = mA or mS/m] (default = 1e-4).
    :param efficiency: scalar, determines synaptic efficiency [unit = mA or mS/m] (default = 1.).
    :param tau_decay: scalar, determines how fast synaptic kernel decays [unit = s] (default = 0.01).
    :param tau_rise: scalar, determines how fast synaptic kernel rises [unit = s] (default = 0.001).
    :param conductivity_based: if true, synaptic kernel unit will be conductivity (mS/m), else current (mA)
           (default = False).
    :param reversal_potential: scalar, determines the reversal potential of the synapse, i.e. the membrane potential at
           which no currents would flow given zero input [unit = mV] (default = -70).
    :param membrane_potential: scalar, indicates the current membrane potential for which to evaluate the synaptic
           current. Only relevant if synapse is conductivity based [unit = mV] (default = None).
    :param synaptic_current_target: vector of length of x, resembles target synaptic current values for the
           passed input x at which to evaluate the synaptic kernel [unit = mA]

    :return synaptic_value or synaptic_current == synaptic_current_target

    """

    ######################
    # initialize synapse #
    ######################

    synapse_instance = syn.Synapse(efficiency, tau_decay, tau_rise, step_size, kernel_threshold, conductivity_based,
                                   reversal_potential)

    #########################
    # build synaptic kernel #
    #########################

    synapse_instance.build_kernel()

    ##############################
    # calculate synaptic current #
    ##############################

    synaptic_current = synapse_instance.get_synaptic_current(x, membrane_potential)

    return synaptic_current if not synaptic_current_target else (synaptic_current == synaptic_current_target)


##################################
# synaptic kernel function tests #
##################################

# test whether output is zero for equal rise and decay delay
tau_rise = 0.01
tau_decay = 0.01
t = np.arange(0, 0.1, 0.001)
assert all(synaptic_kernel_value_test(t, tau_rise=tau_rise, tau_decay=tau_decay)) == 0

# test whether synaptic kernel values are positive for tau_decay > tau_rise
tau_rise = 0.001
assert all(synaptic_kernel_value_test(t, tau_rise=tau_rise, tau_decay=tau_decay)) >= 0

# test whether synaptic kernel values are negative for tau_decay < tau_rise
tau_decay = 0.0001
assert all(synaptic_kernel_value_test(t, tau_rise=tau_rise, tau_decay=tau_decay)) <= 0

# test whether synaptic kernel values are positive for positive synaptic efficiency
efficiency = 1.
assert all(synaptic_kernel_value_test(t, efficiency=efficiency)) >= 0

# test whether synaptic kernel values are negative for negative synaptic efficiency
efficiency = -1.
assert all(synaptic_kernel_value_test(t, efficiency=efficiency)) <= 0

#########################
# synaptic kernel tests #
#########################

# tests whether higher time resolution leads to longer kernel
kernel_threshold_1 = 1e-4
step_size_1 = 0.001
step_size_2 = 0.0001
assert len(synaptic_kernel_test(step_size_1, kernel_threshold_1)) < \
       len(synaptic_kernel_test(step_size_2, kernel_threshold_1))

# tests whether lower kernel value threshold leads to longer kernel
kernel_threshold_2 = 1e-5
assert len(synaptic_kernel_test(step_size_1, kernel_threshold_1)) < \
       len(synaptic_kernel_test(step_size_1, kernel_threshold_2))

# tests whether kernel has a single extremum or not
kernel_diff = np.diff(synaptic_kernel_test(step_size_1, kernel_threshold_1))
kernel_rise = np.where(kernel_diff >= 0)
kernel_decay = np.where(kernel_diff <= 0)
assert (np.diff(kernel_rise) == 1).all() and (np.diff(kernel_decay) == 1).all()

##########################
# synaptic current tests #
##########################

