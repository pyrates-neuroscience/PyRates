"""Test suite for accurate behavior of models provided under pyrates.model_templates"""

# external _imports
from typing import Union
import numpy as np
import pytest

# pyrates internal _imports
from pyrates import simulate

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


# define backends for which to run the tests
backends = ['default']

# define test accuracy
accuracy = 1e-4

#########
# Tests #
#########


def test_3_1_jansenrit():
    """Testing accuracy of Jansen-Rit model implementations:
    """

    T = 1.5
    dt = 1e-4
    dts = 1e-2

    for b in backends:

        # compare single operator JRC implementation with multi-node JRC implementation
        ###############################################################################

        # single operator JRC
        r1 = simulate("model_templates.jansen_rit.simple_jansenrit.JRC_simple", simulation_time=T,
                      outputs={'EIN': 'JRC/JRC_op/PSP_ein'}, backend=b, step_size=dt, solver='scipy',
                      sampling_step_size=dts, clear=True, file_name='jrc1')

        # multi-node JRC
        r2 = simulate("model_templates.jansen_rit.simple_jansenrit.JRC", simulation_time=T,
                      outputs={'EIN': 'EIN/RPO_e/PSP'}, backend=b, step_size=dt, solver='scipy',
                      sampling_step_size=dts, clear=True, file_name='jrc2')

        assert np.mean(r1.values.flatten() - r2.values.flatten()) == pytest.approx(0., rel=accuracy, abs=accuracy)


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

    for b in backends:

        # assess correct behavior of the model around the bi-stable regime
        ##################################################################

        # perform simulation
        r1 = simulate("model_templates.montbrio.simple_montbrio.QIF_exc", simulation_time=T, sampling_step_size=dts,
                      inputs={"p/Op_e/inp": inp}, outputs={"r": "p/Op_e/r"}, method='RK23', backend=b, solver='scipy',
                      step_size=dt, clear=True, file_name='m1')

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
    inp = np.round(inp, decimals=4)

    for b in backends:

        # perform simulation
        r1 = simulate("model_templates.wilson_cowan.simple_wilsoncowan.WC_simple", simulation_time=T,
                      sampling_step_size=dts, inputs={"E/Op_rate/I_ext": inp}, outputs={"R_e": "E/Op_rate/r"},
                      backend=b, solver='scipy', step_size=dt, clear=True,
                      file_name='wc1', method='RK23')

        # test firing rate relationships at pre-defined times
        times = [29.0, 49.0, 79.0]
        indices = [np.argmin(np.abs(t - r1.index)) for t in times]
        assert r1.iloc[indices[0], 0] < r1.iloc[indices[1], 0]
        assert r1.iloc[indices[0], 0] - r1.iloc[indices[2], 0] == pytest.approx(0., rel=accuracy, abs=accuracy)

        # repeat for model with short-term plasticity
        #############################################

        T = 320.0
        dt = 5e-3
        dts = 1e-2

        in_start = int(np.round(130.0 / dt))
        in_dur = int(np.round(20.0 / dt))
        inp = np.zeros((int(np.round(T / dt)),))
        inp[in_start:in_start + in_dur] = 1.0

        # perform simulation
        r2 = simulate("model_templates.wilson_cowan.simple_wilsoncowan.WC_stp", simulation_time=T,
                      sampling_step_size=dts, inputs={"E/E_op/I_ext": inp}, outputs={"V_e": "E/E_op/v"},
                      backend=b, solver='scipy', step_size=dt, clear=True, method='RK23', atol=1e-6, rtol=1e-6,
                      file_name='wc2')

        # test firing rate relationships at pre-defined times
        times = [129.0, 149.0, 319.0]
        indices = [np.argmin(np.abs(t - r2.index)) for t in times]
        assert r2.iloc[indices[0], 0] < r2.iloc[indices[1], 0]
        assert r2.iloc[indices[0], 0] - r2.iloc[indices[2], 0] == pytest.approx(0., rel=accuracy, abs=accuracy)


def test_3_4_kuramoto():
    """Tests accurate behavior of kuramoto oscillator model.
    """

    T = 6.0
    dt = 1e-4
    dts = 1e-2

    for b in backends:

        # assess correct response of single base oscillator
        ###################################################

        # perform simulation
        r1 = simulate("model_templates.kuramoto.simple_kuramoto.KM_single", simulation_time=T, sampling_step_size=dts,
                      outputs={"theta": "p1/Op_base/theta"}, backend=b, solver='scipy', step_size=dt, clear=True,
                      method='RK23', file_name='km1')

        # test linear oscillator properties
        omega = 10.0
        results = np.sin(r1.values[:, 0]*2*np.pi)
        target = np.sin(omega*2.0*np.pi*r1.index.values)
        assert results - target == pytest.approx(0., rel=1e-2, abs=1e-2)

        # assess correct response of two coupled oscillators
        ####################################################

        # perform simulation
        r2 = simulate("model_templates.kuramoto.simple_kuramoto.KMN", simulation_time=T, sampling_step_size=dts,
                      outputs={"theta1": "p1/Op_base/theta", "theta2": "p2/Op_base/theta"},
                      backend=b, solver='scipy', step_size=dt, clear=True, file_name='km2')

        # test whether oscillator 2 showed a faster phase development than oscillator 1
        assert r2['theta1'].iloc[-1] < r2['theta2'].iloc[-1]

        # repeat test 2 for two coupled noisy oscillators
        #################################################

        inp1 = np.random.randn(int(np.round(T / dt))) * 0.5
        inp2 = np.random.randn(int(np.round(T / dt))) * 0.1

        # perform simulation
        r3 = simulate("model_templates.kuramoto.simple_kuramoto.KMN_noise", simulation_time=T, sampling_step_size=dts,
                      outputs={"theta1": "p1/Op_noise/theta", "theta2": "p2/Op_noise/theta"},
                      inputs={"p1/Op_noise/xi": inp1, "p2/Op_noise/xi": inp2}, backend=b, solver='scipy',
                      step_size=dt, clear=True, method='RK23', file_name='km3')

        # test whether oscillator 2 showed a faster phase development than oscillator 1
        assert r3['theta1'].iloc[-1] < r3['theta2'].iloc[-1]
