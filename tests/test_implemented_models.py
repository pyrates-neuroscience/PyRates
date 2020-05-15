"""Test suite for accurate behavior of models provided under pyrates.model_templates"""

# external imports
from typing import Union

import numpy as np
import pytest

# pyrates internal imports
from pyrates.ir import CircuitIR

# meta infos
__author__ = "Richard Gast"
__status__ = "Development"


###########
# Utility #
###########


def setup_module():
    print("\n")
    print("================================")
    print("| Test Suite 3 : SimNet Module |")
    print("================================")


def nmrse(x: np.ndarray,
          y: np.ndarray
          ) -> Union[float, np.ndarray]:
    """Calculates the normalized root mean squared error of two vectors of equal length.
    Parameters
    ----------
    x,y
        Arrays to calculate the nmrse between.
    Returns
    -------
    float
        Normalized root mean squared error.
    """

    max_val = np.max((np.max(x, axis=0), np.max(y, axis=0)))
    min_val = np.min((np.min(x, axis=0), np.min(y, axis=0)))

    diff = x - y

    return np.sqrt(np.sum(diff ** 2, axis=0)) / (max_val - min_val)


#########
# Tests #
#########


def test_3_1_jansenrit():
    """Testing accuracy of Jansen-Rit model implementations:
    """

    T = 1.5
    dt = 1e-4
    dts = 1e-2

    # compare single operator JRC implementation with multi-node JRC implementation
    ###############################################################################

    # single operator JRC
    jrc1 = CircuitIR.from_yaml("model_templates.jansen_rit.simple_jansenrit.JRC_simple")
    jrc1 = jrc1.compile(backend='numpy', step_size=dt, solver='scipy')
    r1 = jrc1.run(T, outputs={'EIN': 'JRC/JRC_op/PSP_ein'}, sampling_step_size=dts)
    jrc1.clear()

    # multi-node JRC
    jrc2 = CircuitIR.from_yaml("model_templates.jansen_rit.simple_jansenrit.JRC")
    jrc2 = jrc2.compile(backend='numpy', step_size=dt, solver='scipy')
    r2 = jrc2.run(T, outputs={'EIN': 'EIN/RPO_e/PSP'}, sampling_step_size=dts)
    jrc2.clear()

    assert np.mean(r1.values.flatten() - r2.values.flatten()) == pytest.approx(0., rel=1e-4, abs=1e-4)


def test_3_2_montbrio():
    """Testing accuracy of mean-field representation of QIF population.
    """

    T = 80.0
    dt = 1e-3
    dts = 1e-2

    in_start = int(np.round(30.0/dt))
    in_dur = int(np.round(20.0/dt))
    inp = np.zeros((int(np.round(T/dt)),))
    inp[in_start:in_start+in_dur] = 5.0

    # assess correct behavior of the model around the bi-stable regime
    ##################################################################

    # set up circuit
    m1 = CircuitIR.from_yaml("model_templates.montbrio.simple_montbrio.QIF_exc")
    m1 = m1.compile(vectorization=True, backend='numpy', solver='scipy', step_size=dt)

    # perform simulation
    r1 = m1.run(T, sampling_step_size=dts, inputs={"pop_e/Op_e/inp": inp}, outputs={"r": "pop_e/Op_e/r"})

    # test firing rate relationships at pre-defined times
    times = [25.0, 49.0, 79.0]
    indices = [np.argmin(np.abs(t - r1.index)) for t in times]
    assert r1.iloc[indices[0], 0] < r1.iloc[indices[1], 0]
    assert r1.iloc[indices[0], 0] < r1.iloc[indices[2], 0]
    assert r1.iloc[indices[1], 0] > r1.iloc[indices[2], 0]


def test_3_3_wilson_cowan():
    """Test accuracy of wilson cowan neural mass model implementation.
    """

    # assess correct response of model to input
    ###########################################

    T = 80.0
    dt = 1e-3
    dts = 1e-2

    in_start = int(np.round(30.0 / dt))
    in_dur = int(np.round(20.0 / dt))
    inp = np.zeros((int(np.round(T / dt)),))
    inp[in_start:in_start + in_dur] = 5.0

    # set up circuit
    wc1 = CircuitIR.from_yaml("model_templates.wilson_cowan.simple_wilsoncowan.WC_simple")
    wc1 = wc1.compile(vectorization=True, backend='numpy', solver='scipy', step_size=dt)

    # perform simulation
    r1 = wc1.run(T, sampling_step_size=dts, inputs={"E/Op_rate/I_ext": inp}, outputs={"R_e": "E/Op_rate/r"})
    wc1.clear()

    # test firing rate relationships at pre-defined times
    times = [29.0, 49.0, 79.0]
    indices = [np.argmin(np.abs(t - r1.index)) for t in times]
    assert r1.iloc[indices[0], 0] < r1.iloc[indices[1], 0]
    assert r1.iloc[indices[0], 0] - r1.iloc[indices[2], 0] == pytest.approx(0., rel=1e-4, abs=1e-4)

    # repeat for model with short-term plasticity
    #############################################

    T = 220.0
    dt = 5e-3
    dts = 1e-2

    in_start = int(np.round(30.0 / dt))
    in_dur = int(np.round(20.0 / dt))
    inp = np.zeros((int(np.round(T / dt)),))
    inp[in_start:in_start + in_dur] = 5.0

    # set up circuit
    wc2 = CircuitIR.from_yaml("model_templates.wilson_cowan.simple_wilsoncowan.WC_stp")
    wc2 = wc2.compile(vectorization=True, backend='numpy', solver='scipy', step_size=dt)

    # perform simulation
    r2 = wc2.run(T, sampling_step_size=dts, inputs={"E/E_op/I_ext": inp}, outputs={"V_e": "E/E_op/v"})
    wc2.clear()

    # test firing rate relationships at pre-defined times
    times = [29.0, 49.0, 219.0]
    indices = [np.argmin(np.abs(t - r2.index)) for t in times]
    assert r2.iloc[indices[0], 0] < r2.iloc[indices[1], 0]
    assert r2.iloc[indices[0], 0] - r2.iloc[indices[2], 0] == pytest.approx(0., rel=1e-3, abs=1e-3)
