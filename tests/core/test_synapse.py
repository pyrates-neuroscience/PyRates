""" Test suite for basic synapse class functionality
"""

import numpy as np
import pytest

from core.synapse import Synapse

###########
# Utility #
###########


def exponential_decay(time_point, scaling=1):
    return np.exp(-time_point*scaling)


def setup_module():
    print("\n")
    print("===============================")
    print("| Test Suite 2 : Synapse Class |")
    print("===============================")


#########
# Tests #
#########

def test_2_1_synapse_init():
    """Testing initialization of Synapse class:


    See Also
    --------
    :class:`Synapse`: Detailed documentation of synapse attributes and methods.
    """

    # test minimal minimal call example
    ###################################

    synapse = Synapse(kernel_function=exponential_decay,
                      efficacy=1, bin_size=1)
    assert isinstance(synapse, Synapse)

    # test expected exceptions
    ##########################

    with pytest.raises(ValueError):
        Synapse(kernel_function=exponential_decay, efficacy=1, bin_size=0)
    with pytest.raises(ValueError):
        Synapse(kernel_function=exponential_decay, efficacy=1, bin_size=-1)
    with pytest.raises(ValueError):
        Synapse(kernel_function=exponential_decay, efficacy=1, bin_size=1, max_delay=0.)
    with pytest.raises(ValueError):
        Synapse(kernel_function=exponential_decay, efficacy=1, bin_size=1, max_delay=-1)
    with pytest.raises(ValueError):
        Synapse(kernel_function=exponential_decay, efficacy=1, bin_size=1, epsilon=0)
    with pytest.raises(ValueError):
        Synapse(kernel_function=exponential_decay, efficacy=1, bin_size=1, epsilon=-1)

    # test synapse_type
    ###################

    synapse = Synapse(kernel_function=exponential_decay,
                      efficacy=1, bin_size=1, synapse_type="something_odd")
    assert synapse.synapse_type == "something_odd"

    synapse = Synapse(kernel_function=exponential_decay, efficacy=1, bin_size=1)
    assert synapse.synapse_type == "excitatory_current"

    synapse = Synapse(kernel_function=exponential_decay, efficacy=-1, bin_size=1)
    assert synapse.synapse_type == "inhibitory_current"

    synapse = Synapse(kernel_function=exponential_decay, efficacy=-1, bin_size=1, conductivity_based=True)
    assert synapse.synapse_type == "inhibitory_conductance"

    synapse = Synapse(kernel_function=exponential_decay, efficacy=1, bin_size=1, conductivity_based=True)
    assert synapse.synapse_type == "excitatory_conductance"

    # test kernel_scaling
    #####################

    synapse = Synapse(kernel_function=exponential_decay, efficacy=1, bin_size=1, conductivity_based=True)
    assert synapse.kernel_scaling(-0.075) == 0

    synapse = Synapse(kernel_function=exponential_decay, efficacy=1, bin_size=1)
    assert synapse.kernel_scaling(1000) == 1


# noinspection PyTypeChecker
def test_2_2_synapse_build_kernel():
    """Testing build_kernel and evaluate_kernel function of Synapse class


    See Also
    --------
    :class:`Synapse`: Detailed documentation of synapse attributes and methods.
    """

    # test minimal minimal call example
    ###################################

    synapse = Synapse(kernel_function=exponential_decay, efficacy=1, bin_size=1)

    assert synapse.synaptic_kernel[0] <= synapse.epsilon
    assert synapse.synaptic_kernel[1] >= synapse.epsilon

    # test with max_delay
    #####################

    max_delay = 10
    synapse = Synapse(kernel_function=exponential_decay, efficacy=1, bin_size=1, max_delay=max_delay)
    assert synapse.synaptic_kernel[0] == synapse.kernel_function(max_delay)

    last_point = max_delay
    while last_point > 0.+0.5*synapse.bin_size:
        last_point -= synapse.bin_size
    assert synapse.synaptic_kernel[-1] == synapse.kernel_function(last_point)

    # test with correct argument to kernel_function
    ###############################################

    synapse = Synapse(kernel_function=exponential_decay, efficacy=1, bin_size=1, scaling=2)
    assert synapse.kernel_function_args["scaling"] == 2

    # test evaluate_kernel
    ######################

    assert synapse.evaluate_kernel() == 1  # e^0 = 1
    assert synapse.evaluate_kernel(1) == np.exp(-2)
    list_ = [1, 2, 3]
    array = np.array(list_)
    assert np.array_equal(synapse.evaluate_kernel(array), np.exp(-array*2))
    # should never do this with a list
    with pytest.raises(TypeError):
        synapse.evaluate_kernel(list_)
        # note: this actually tests the exponential_decay function, but it would fail in most other cases as well

    # test with wrong argument to kernel_function
    #############################################

    with pytest.raises(TypeError):
        _ = Synapse(kernel_function=exponential_decay, efficacy=1, bin_size=1, wrong_keyword=2)


def test_2_3_synapse_get_synaptic_current():
    """Testing build_kernel function of Synapse class


    See Also
    --------
    :class:`Synapse`: Detailed documentation of synapse attributes and methods.

    TODO: Look at original equations and think of some test cases
    to test:
      - kernel multiplication and integration (convolution?)
      - kernel_scaling with conductivity-based synapse
      - depression
    """
    pass


# noinspection PyTypeChecker
def test_2_4_ampa_current_synapse():
    """Tests whether synapse with standard AMPA parametrization from Thomas Knoesche (corresponding to AMPA synapse
     in [1]_) shows expected output for various firing rate inputs.

    See Also
    --------
    :class:`DoubleExponentialSynapse`: Detailed documentation of synapse parameters.
    :class:`Synapse`: Detailed documentation of synapse attributes and methods.

    References
    ----------
    .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
       of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

    """

    from core.synapse import AMPACurrentSynapse

    # synapse parameters
    ####################

    # efficacy = 1.273 * 3e-13  # unit = A
    # tau_decay = 0.006  # unit = s
    # tau_rise = 0.0006  # unit = s
    step_size = 5.e-4  # unit = s
    synaptic_kernel_length = 0.05  # unit = s
    # conductivity_based = False

    # initialize synapse
    ####################

    synapse = AMPACurrentSynapse(bin_size=step_size,
                                 max_delay=synaptic_kernel_length)

    # define firing rate inputs
    ###########################
    firing_rates_1 = np.zeros_like(synapse.synaptic_kernel)
    firing_rates_2 = np.ones_like(synapse.synaptic_kernel) * 300.0
    firing_rates_3 = np.zeros_like(synapse.synaptic_kernel)
    firing_rates_3[40:70] = 300.0

    # calculate synaptic currents
    #############################

    for i in range(len(firing_rates_1)):
        synapse.pass_input(firing_rates_1[i])
        synapse.get_synaptic_current()
    synaptic_current_1 = synapse.get_synaptic_current()
    synapse.clear()
    for i in range(len(firing_rates_2)):
        synapse.pass_input(firing_rates_2[i])
        synapse.get_synaptic_current()
    synaptic_current_2 = synapse.get_synaptic_current()
    synapse.clear()

    # get synaptic current at each incoming firing rate of firing_rates_3
    idx = np.arange(1, len(firing_rates_3))
    synaptic_current_3 = np.zeros(len(idx))
    for i in range(len(idx)):
        synapse.pass_input(firing_rates_3[i])
        synaptic_current_3[i] = synapse.get_synaptic_current()

    # perform unit tests
    ####################

    # test whether zero input to AMPA synapse leads to zero synaptic current.
    assert synaptic_current_1 == 0.

    # test whether increased input to AMPA synapse leads to increased synaptic current.
    assert synaptic_current_2 > synaptic_current_1

    # test whether synaptic current response to step-function input has a single maximum.
    pairwise_difference = np.diff(synaptic_current_3)
    response_rise = np.where(pairwise_difference > 0.)
    response_decay = np.where(pairwise_difference < 0.)
    assert (np.diff(response_rise) == 1).all()
    assert (np.diff(response_decay) == 1).all()


# noinspection PyTypeChecker
def test_2_5_gabaa_current_synapse():
    """Tests whether synapse with standard GABAA parametrization from Thomas Knoesche (corresponding to GABAA
    synapse in [1]_) shows expected output for various firing rate inputs.

    See Also
    --------
    :class:`DoubleExponentialSynapse`: Detailed documentation of synapse parameters.
    :class:`Synapse`: Detailed documentation of synapse attributes and methods.

    References
    ----------
    .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
       of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

    """

    from core.synapse import GABAACurrentSynapse

    # synapse parameters
    ####################

    # efficacy = 1.273 * -1e-12  # unit = A
    # tau_decay = 0.02  # unit = s
    # tau_rise = 0.0004  # unit = s
    step_size = 5.e-4  # unit = s
    # epsilon = 1e-13  # unit = V or A or [conductivity]
    # max_delay = None  # unit = s
    # conductivity_based = False

    # initialize synapse
    ####################

    synapse = GABAACurrentSynapse(bin_size=step_size)

    # define firing rate inputs
    ###########################
    firing_rates_1 = np.zeros_like(synapse.synaptic_kernel)
    firing_rates_2 = np.ones_like(synapse.synaptic_kernel) * 300.0
    firing_rates_3 = np.zeros_like(synapse.synaptic_kernel)
    firing_rates_3[40:70] = 300.0

    # calculate synaptic currents
    #############################
    for i in range(len(firing_rates_1)):
        synapse.pass_input(firing_rates_1[i])
        synapse.get_synaptic_current()
    synaptic_current_1 = synapse.get_synaptic_current()
    synapse.clear()
    for i in range(len(firing_rates_2)):
        synapse.pass_input(firing_rates_2[i])
        synapse.get_synaptic_current()
    synaptic_current_2 = synapse.get_synaptic_current()
    synapse.clear()

    # get synaptic current at each incoming firing rate of firing_rates_3
    idx = np.arange(1, len(firing_rates_3))
    synaptic_current_3 = np.zeros(len(idx))
    for i in range(len(idx)):
        synapse.pass_input(firing_rates_3[i])
        synaptic_current_3[i] = synapse.get_synaptic_current()

    # perform unit tests
    ####################

    # test whether zero input to GABAA synapse leads to zero synaptic current.
    assert synaptic_current_1 == 0.

    # test whether increased input to GABAA synapse leads to decreased synaptic current.
    assert synaptic_current_2 < synaptic_current_1

    # test whether synaptic current response to step-function input has a single minimum.
    pairwise_difference = np.diff(synaptic_current_3)
    response_rise = np.where(pairwise_difference > 0.)
    response_decay = np.where(pairwise_difference < 0.)
    assert (np.diff(response_rise) == 1).all()
    assert (np.diff(response_decay) == 1).all()


# noinspection PyTypeChecker
def test_2_6_ampa_conductivity_synapse():
    """Tests whether synapse with parametrization from Thomas Knoesche corresponding to conductivity based AMPA
    synapse shows expected output for various firing rate inputs.

    See Also
    --------
    :class:`DoubleExponentialSynapse`: Detailed documentation of synapse parameters.
    :class:`Synapse`: Detailed documentation of synapse attributes and methods.

    """

    from core.synapse import AMPAConductanceSynapse

    # synapse parameters
    ####################

    # efficacy = 1.273 * 7.2e-10  # unit = S
    # tau_decay = 0.0015  # unit = s
    # tau_rise = 0.000009  # unit = s
    step_size = 5.e-4  # unit = s
    # synaptic_kernel_length = 0.05  # unit = s
    # conductivity_based = True

    # initialize synapse
    ####################

    synapse = AMPAConductanceSynapse(bin_size=step_size)

    # define firing rate inputs and membrane potential
    ##################################################

    firing_rates_1 = np.zeros_like(synapse.synaptic_kernel)
    firing_rates_2 = np.ones_like(synapse.synaptic_kernel) * 300.0
    firing_rates_3 = np.zeros_like(synapse.synaptic_kernel)
    firing_rates_3[40:70] = 300.0

    membrane_potential = -0.09

    # calculate synaptic currents
    #############################
    for i in range(len(firing_rates_1)):
        synapse.pass_input(firing_rates_1[i])
        synapse.get_synaptic_current()
    synaptic_current_1 = synapse.get_synaptic_current(membrane_potential=membrane_potential)
    synapse.clear()
    for i in range(len(firing_rates_2)):
        synapse.pass_input(firing_rates_2[i])
        synapse.get_synaptic_current()
    synaptic_current_2 = synapse.get_synaptic_current(membrane_potential=membrane_potential)
    synapse.clear()

    # get synaptic current at each incoming firing rate of firing_rates_3
    idx = np.arange(1, len(firing_rates_3))
    synaptic_current_3 = np.zeros(len(idx))
    for i in range(len(idx)):
        synapse.pass_input(firing_rates_3[i])
        synaptic_current_3[i] = synapse.get_synaptic_current(membrane_potential=membrane_potential)

    # perform unit tests
    ####################

    # test whether zero input to AMPA conductance synapse leads to zero synaptic current.
    assert synaptic_current_1 == 0.

    # test whether increased input to AMPA conductance synapse leads to increased synaptic current.
    assert synaptic_current_2 > synaptic_current_1

    # test whether synaptic current response to step-function input has a single minimum.
    pairwise_difference = np.diff(synaptic_current_3)
    response_rise = np.where(pairwise_difference > 0.)
    response_decay = np.where(pairwise_difference < 0.)
    assert (np.diff(response_rise) == 1).all()
    assert (np.diff(response_decay) == 1).all()
