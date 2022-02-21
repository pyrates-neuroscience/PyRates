"""Test suite for accurate behavior of models provided under pyrates.model_templates"""

# external _imports
from typing import Union
import numpy as np
import pytest

# pyrates internal _imports
from pyrates import integrate

# meta infos
__author__ = "Richard Gast"
__status__ = "Development"


###########
# Utility #
###########

# define backends for which to run the tests
backends = ['torch', 'default', 'tensorflow']
vectorization = [True, False, True]
backend_kwargs = [{}, {}, {}]

# define test accuracy
accuracy = 1e-4


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

    for b, v, kwargs in zip(backends, vectorization, backend_kwargs):

        # compare single operator JRC implementation with multi-node JRC implementation
        ###############################################################################

        # single operator JRC
        r1 = integrate("model_templates.neural_mass_models.jansenrit.JRC2", simulation_time=T,
                       outputs={'EIN': 'jrc/jrc_op/V_ein'}, backend=b, step_size=dt, solver='scipy',
                       sampling_step_size=dts, clear=True, file_name='jrc1', vectorize=v, **kwargs)

        # multi-node JRC
        r2 = integrate("model_templates.neural_mass_models.jansenrit.JRC", simulation_time=T,
                       outputs={'EIN': 'ein/rpo_e/V'}, backend=b, step_size=dt, solver='scipy',
                       sampling_step_size=dts, clear=True, file_name='jrc2', vectorize=v, **kwargs)

        assert np.mean(r1.values.flatten() - r2.values.flatten()) == pytest.approx(0., rel=accuracy, abs=accuracy)


def test_3_2_qif():
    """Testing accuracy of mean-field representation of QIF population.
    """

    T = 80.0
    dt = 1e-3
    dts = 1e-2

    in_start = int(np.round(30.0/dt))
    in_dur = int(np.round(20.0/dt))
    inp = np.zeros((int(np.round(T/dt)),))
    inp[in_start:in_start+in_dur] = 5.0

    for b, kwargs in zip(backends, backend_kwargs):

        # compare qif population dynamics with and without plasticity
        #############################################################

        # basic qif population
        r1 = integrate("model_templates.neural_mass_models.qif.qif", simulation_time=T, sampling_step_size=dts,
                       inputs={"p/qif_op/I_ext": inp}, outputs={"r": "p/qif_op/r"}, method='RK45', backend=b,
                       solver='scipy', step_size=dt, clear=True, file_name='m1', vectorize=False, **kwargs)

        # qif population with spike-frequency adaptation
        r2 = integrate("model_templates.neural_mass_models.qif.qif_sfa", simulation_time=T, sampling_step_size=dts,
                       inputs={"p/qif_sfa_op/I_ext": inp}, outputs={"r": "p/qif_sfa_op/r"}, method='RK45', backend=b,
                       solver='scipy', step_size=dt, clear=True, file_name='m2', vectorize=False, **kwargs)

        # qif population with synaptic depression
        r3 = integrate("model_templates.neural_mass_models.qif.qif_sd", simulation_time=T, sampling_step_size=dts,
                       inputs={"p/qif_op/I_ext": inp}, outputs={"r": "p/qif_op/r"}, method='RK45', backend=b,
                       solver='scipy', step_size=dt, clear=True, file_name='m3', vectorize=False, **kwargs)

        # test firing rate relationships at pre-defined times
        time = 49.0
        idx = np.argmin(np.abs(time - r1.index))
        assert r1.iloc[idx, 0] > r2.iloc[idx, 0] > r3.iloc[idx, 0]


def test_3_3_wilson_cowan():
    """Test accuracy of wilson cowan neural mass model implementation.
    """

    # assess correct response of model to input
    ###########################################

    T = 120.0
    dt = 5e-4
    dts = 1e-2

    in_start = int(np.round(50.0 / dt))
    in_dur = int(np.round(30.0 / dt))
    inp = np.zeros((int(np.round(T / dt)),))
    inp[in_start:in_start + in_dur] = 1.0

    for b, v, kwargs in zip(backends, vectorization, backend_kwargs):

        # standard wilson-cowan model
        r1 = integrate("model_templates.neural_mass_models.wilsoncowan.WC", simulation_time=T,
                       sampling_step_size=dts, inputs={"e/se_op/r_ext": inp}, outputs={"R_e": "e/rate_op/r"},
                       backend=b, solver='scipy', step_size=dt, clear=True, file_name='wc1', vectorize=v, **kwargs)

        # wilson-cowan model with synaptic short-term plasticity
        r2 = integrate("model_templates.neural_mass_models.wilsoncowan.WC_stp", simulation_time=T,
                       sampling_step_size=dts, inputs={"e/se_op/r_ext": inp}, outputs={"R_e": "e/rate_op/r"},
                       backend=b, solver='scipy', step_size=dt, clear=True, file_name='wc2', vectorize=v, **kwargs)

        # test firing rate relationships at pre-defined times
        times = [75.0, 119.0]
        indices = [np.argmin(np.abs(t - r2.index)) for t in times]
        for idx in indices:
            assert r2.iloc[idx, 0] > r1.iloc[idx, 0]


def test_3_4_kuramoto():
    """Tests accurate behavior of kuramoto oscillator model.
    """

    T = 6.0
    dt = 1e-4
    dts = 1e-2

    in_start = int(np.round(0.0 / dt))
    in_dur = int(np.round(0.1 / dt))
    inp = np.zeros((int(np.round(T / dt)),))
    inp[in_start:in_start + in_dur] = 1.0

    for b, v, kwargs in zip(backends, vectorization, backend_kwargs):

        # assess correct response of single base oscillator
        ###################################################

        # perform simulation
        r1 = integrate("model_templates.oscillators.kuramoto.kmo", simulation_time=T, sampling_step_size=dts,
                       outputs={"theta": "p/phase_op/theta"}, backend=b, solver='scipy', step_size=dt, clear=True,
                       method='RK45', file_name='km1', vectorize=False, **kwargs)

        # test linear oscillator properties
        omega = 10.0
        results = np.sin(r1.values[:, 0]*2*np.pi)
        target = np.sin(omega*2.0*np.pi*r1.index.values)
        assert results - target == pytest.approx(0., rel=1e-2, abs=1e-2)

        # assess correct response of two coupled oscillators
        ####################################################

        # perform simulation
        r2 = integrate("model_templates.oscillators.kuramoto.kmo_2coupled", simulation_time=T, sampling_step_size=dts,
                       outputs={"theta1": "p1/phase_op/theta", "theta2": "p2/phase_op/theta"}, backend=b, solver='scipy',
                       inputs={"p1/phase_op/ext_in": inp}, step_size=dt, clear=True, file_name='km2', vectorize=v,
                       method='RK45', **kwargs)

        # test whether the oscillators expressed de-phasing
        init = int(in_dur*dt/dts)
        diff_init = r2['theta1'].iloc[init] - r2['theta2'].iloc[init]
        diff_end = r2['theta1'].iloc[-1] - r2['theta2'].iloc[-1]
        assert diff_end > diff_init
