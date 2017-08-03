"""
Here, we test the symbolic math functionality.
"""
# TODO: Test for symbolic display
# TODO: Test for numpy conversion of symbolic equations


# import sympy as sp
import numpy as np
from matplotlib import pyplot as plt

from base.synapses import AMPACurrentSynapse, GABAACurrentSynapse

__author__ = "Daniel F. Rose"
__status__ = "Development"


# def test_symbolic_display():
#     some_synapse = SomeSynapse()
#     equation = some_synapse.kernel_symbolic
#     assert isinstance(equation, sp.Eq)
#     sp.pprint(equation)


def test_numpy_representation_of_symbols():
    time_axis = np.linspace(0, 0.05, 1000)
    some_synapse = AMPACurrentSynapse(time_axis)
    numeric_kernel = some_synapse.kernel.values
    assert isinstance(numeric_kernel, np.ndarray)


# TODO: include test for kernel functionality


if __name__ == "__main__":
    test_numpy_representation_of_symbols()

    # Plot kernels
    time_axis = np.linspace(0, 0.05, 1000)

    some_synapse = AMPACurrentSynapse(time_axis)
    numeric_kernel = some_synapse.kernel.values
    plt.plot(time_axis, numeric_kernel)

    some_synapse = GABAACurrentSynapse(time_axis)
    numeric_kernel = some_synapse.kernel.values
    plt.plot(time_axis, numeric_kernel)

    plt.show()
