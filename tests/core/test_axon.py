""" Elementary tests for axon classes
"""

import numpy as np
import pytest

###########
# Utility #
###########


def multiply(membrane_potential, factor=0):
    return membrane_potential * factor


def setup_module():
    print("\n")
    print("=============================")
    print("| Test Suite 1 : Axon Class |")
    print("=============================")


#########
# Tests #
#########

def test_1_1_axon_transfer_function():
    """Testing basic functionality of Axon class:
    The compute_firing_rate function is tested with a 2x2 design:
    - with and without transfer_function_args as captured in the init
    - called with a scalar or with an array"""

    from pyrates.axon import Axon

    scalar = 5
    array = np.arange(0, 5, 0.1)

    # Testing without transfer_function_args
    ########################################

    zero_axon = Axon(transfer_function=multiply)

    assert zero_axon.compute_firing_rate(scalar) == 0
    assert np.array_equal(zero_axon.compute_firing_rate(array), array * 0)

    # Testing with transfer_function_args given as keyword argument
    ###############################################################

    double_axon = Axon(transfer_function=multiply,
                       factor=2.)
    assert double_axon.compute_firing_rate(scalar) == 10
    assert np.array_equal(double_axon.compute_firing_rate(array), array * 2)

    # Erroneous behaviour with list instead of array
    ################################################

    list_ = list(array)
    with pytest.raises(TypeError):
        double_axon.compute_firing_rate(list_)  # PyCharm correctly warns about this

    integer_axon = Axon(transfer_function=multiply,
                        factor=10)

    result_list = integer_axon.compute_firing_rate(list_)
    result_array = integer_axon.compute_firing_rate(array)

    assert not len(result_list) == len(result_array)


def test_1_2_jr_sigmoid_axon():
    """Tests whether axon with standard parametrization from [1]_ shows expected output to membrane potential input.

    See Also
    --------
    :class:`SigmoidAxon`: Detailed documentation of axon parameters.
    :class:`Axon`: Detailed documentation of axon attributes and methods.

    References
    ----------
    .. [1] B.H. Jansen & V.G. Rit, "Electroencephalogram and visual evoked potential generation in a mathematical model
       of coupled cortical columns." Biological Cybernetics, vol. 73(4), pp. 357-366, 1995.

    """

    # axon parameters
    #################

    max_firing_rate = 5.  # unit = 1
    membrane_potential_threshold = -0.069  # unit = V
    sigmoid_steepness = 555.56  # unit = 1/V

    # initialize axon
    #################

    from pyrates.axon import SigmoidAxon
    axon = SigmoidAxon(max_firing_rate=max_firing_rate,
                       firing_threshold=membrane_potential_threshold,
                       slope=sigmoid_steepness)

    # define inputs (unit = V)
    ##########################

    membrane_potential_1 = membrane_potential_threshold
    membrane_potential_2 = membrane_potential_threshold - 0.01
    membrane_potential_3 = membrane_potential_threshold + 0.01
    membrane_potential_4 = membrane_potential_threshold - 0.1
    membrane_potential_5 = membrane_potential_threshold + 0.1

    # get firing rates
    ##################

    firing_rate_1 = axon.compute_firing_rate(membrane_potential_1)
    firing_rate_2 = axon.compute_firing_rate(membrane_potential_2)
    firing_rate_3 = axon.compute_firing_rate(membrane_potential_3)
    firing_rate_4 = axon.compute_firing_rate(membrane_potential_4)
    firing_rate_5 = axon.compute_firing_rate(membrane_potential_5)

    # perform unit tests
    ####################

    # print('-----------------')
    # print('| Test I - Axon |')
    # print('-----------------')

    # print('I.1 test whether output firing rate at membrane potential threshold is indeed 0.5 scaled by the max '
    #       'firing rate.')
    assert firing_rate_1 == 0.5 * max_firing_rate
    # print('I.1 done!')

    # print('I.2 test whether output firing rate gets smaller for lower membrane potential and the other way around.')
    assert firing_rate_2 < firing_rate_1
    assert firing_rate_3 > firing_rate_1
    # print('I.2 done!')

    # print('I.3 test whether equal amounts of hyperpolarization and depolarization lead to equal changes in membrane'
    #       ' potential.')
    assert np.abs(firing_rate_1 - firing_rate_2) == pytest.approx(np.abs(firing_rate_1 - firing_rate_3), 1e-4)
    # print('I.3 done!')

    # print('I.4 test whether extreme depolarization leads to almost zero firing rate.')
    assert firing_rate_4 == pytest.approx(0., 1e-2)
    # print('I.4 done!')

    # print('I.5 test whether extreme hyperpolarization leads to almost max firing rate')
    assert firing_rate_5 == max_firing_rate
    # this was supposed to be within 2 digits correct. It is in fact exactly equal (within numerical accuracy).
    # print('I.5 done!')







