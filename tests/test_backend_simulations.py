"""Test suite for basic backend simulation functionalities.
"""

# external _imports
from typing import Union
import numpy as np
import pytest

# pyrates internal _imports
from pyrates import integrate

# meta infos
__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"

###########
# Utility #
###########

# set backends to run the tests for
backends = ['default']

# set accuracy for all tests
accuracy = 1e-4


def setup_module():
    print("\n")
    print("==================================================")
    print("| Test Suite: Backend Simulation Functionalities |")
    print("==================================================")


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


def test_2_1_operator():
    """Testing operator functionality of compute graph class:

    See Also
    --------
    :method:`add_operator`: Detailed documentation of method for adding operations to instance of `ComputeGraph`.
    """

    # simulation parameters
    dt = 1e-1
    sim_time = 10.0
    sim_steps = int(sim_time / dt)
    inp = np.zeros((sim_steps,)) + 0.5

    for b in backends:

        # test correct numerical evaluation of operator with two coupled simple, linear equations
        #########################################################################################

        # simulate operator behavior
        results = integrate("model_templates.test_resources.test_backend.net0", simulation_time=sim_time, step_size=dt,
                            outputs={'a': 'pop0/op0/a'}, vectorize=False, backend=b, clear=True, file_name='net0')

        # generate target values
        update0_1 = lambda x: x * 0.5
        update0_0 = lambda x: x + 2.0
        targets = np.zeros((sim_steps, 2), dtype=np.float64)
        for i in range(sim_steps-1):
            targets[i + 1, 0] = targets[i, 0] + dt * update0_0(targets[i, 1])
            targets[i + 1, 1] = targets[i, 1] + dt * update0_1(targets[i, 0])

        diff = results['a'].values[:] - targets[:, 1]
        assert np.mean(np.abs(diff)) == pytest.approx(0., rel=accuracy, abs=accuracy)

        # test correct numerical evaluation of operator with a single differential equation and external input
        ######################################################################################################

        # simulate operator behavior
        results = integrate("model_templates.test_resources.test_backend.net1", simulation_time=sim_time, step_size=dt,
                            inputs={'pop0/op1/u': inp}, outputs={'a': 'pop0/op1/a'}, vectorize=False, backend=b,
                            clear=True, file_name='net1')

        # calculate operator behavior from hand
        update1 = lambda x, y: x + dt * (y - x)
        targets = np.zeros((sim_steps, 1), dtype=np.float32)
        for i in range(sim_steps-1):
            targets[i + 1] = update1(targets[i], inp[i])

        diff = results['a'].values[:] - targets[:, 0]
        assert np.mean(np.abs(diff)) == pytest.approx(0., rel=accuracy, abs=accuracy)

        # test correct numerical evaluation of operator with two coupled equations (1 ODE, 1 non-DE eq.)
        ################################################################################################

        results = integrate("model_templates.test_resources.test_backend.net2", simulation_time=sim_time,
                            outputs={'a': 'pop0/op2/a'}, step_size=dt, vectorize=False, backend=b,
                            clear=True, file_name='net_2')

        # calculate operator behavior from hand
        update2 = lambda x: 1. / (1. + np.exp(-x))
        targets = np.zeros((sim_steps, 2), dtype=np.float32)
        for i in range(sim_steps-1):
            targets[i + 1, 1] = update2(targets[i, 0])
            targets[i + 1, 0] = update1(targets[i, 0], targets[i + 1, 1])

        diff = results['a'].values[:] - targets[:, 0]
        assert np.mean(np.abs(diff)) == pytest.approx(0., rel=accuracy, abs=accuracy)

        # test correct numerical evaluation of operator with a two coupled DEs
        ######################################################################

        results = integrate("model_templates.test_resources.test_backend.net3", simulation_time=sim_time,
                            outputs={'b': 'pop0/op3/b'}, inputs={'pop0/op3/u': inp}, out_dir="/tmp/log",
                            step_size=dt, vectorize=True, backend=b, clear=True, file_name='net_3')

        # calculate operator behavior from hand
        update3_0 = lambda a, b, u: a + dt * (-10. * a + b ** 2 + u)
        update3_1 = lambda b, a: b + dt * 0.1 * a
        targets = np.zeros((sim_steps, 2), dtype=np.float32)
        for i in range(sim_steps-1):
            targets[i + 1, 0] = update3_0(targets[i, 0], targets[i, 1], inp[i])
            targets[i + 1, 1] = update3_1(targets[i, 1], targets[i, 0])

        diff = results['b'].values[:] - targets[:, 1]
        assert np.mean(np.abs(diff)) == pytest.approx(0., rel=accuracy, abs=accuracy)


def test_2_2_node():
    """Testing node functionality of compute graph class.

    See Also
    --------
    :method:`add_node`: Detailed documentation of method for adding nodes to instance of `ComputeGraph`.
    """

    dt = 1e-1
    sim_time = 10.
    sim_steps = int(np.round(sim_time/dt))

    for b in backends:

        # test correct numerical evaluation of node with 2 operators, where op1 projects to op2
        #######################################################################################

        # simulate node behavior
        results = integrate("model_templates.test_resources.test_backend.net4", simulation_time=sim_time,
                            outputs={'a': 'pop0/op1/a'}, step_size=dt, vectorize=True, backend=b, clear=True,
                            file_name='net4')

        # calculate node behavior from hand
        update0 = lambda x: x + dt * 2.
        update1 = lambda x, y: x + dt * (y - x)
        targets = np.zeros((sim_steps, 2), dtype=np.float32)
        for i in range(sim_steps-1):
            targets[i + 1, 0] = update0(targets[i, 0])
            targets[i + 1, 1] = update1(targets[i, 1], targets[i, 0])

        diff = results['a'].values[:] - targets[:, 1]
        assert np.mean(np.abs(diff)) == pytest.approx(0., rel=accuracy, abs=accuracy)

        # test correct numerical evaluation of node with 2 independent operators
        ########################################################################

        # simulate node behavior
        results = integrate("model_templates.test_resources.test_backend.net5", simulation_time=sim_time,
                            outputs={'a': 'pop0/op5/a'}, step_size=dt, vectorize=True, backend=b, clear=True,
                            file_name='net5')

        # calculate node behavior from hand
        targets = np.zeros((sim_steps, 2), dtype=np.float32)
        for i in range(sim_steps-1):
            targets[i + 1, 0] = update0(targets[i, 0])
            targets[i + 1, 1] = update1(targets[i, 1], 0.)

        diff = results['a'].values[:] - targets[:, 1]
        assert np.mean(np.abs(diff)) == pytest.approx(0., rel=accuracy, abs=accuracy)

        # test correct numerical evaluation of node with 2 independent operators projecting to the same target operator
        ###############################################################################################################

        results = integrate("model_templates.test_resources.test_backend.net6", simulation_time=sim_time,
                            outputs={'a': 'pop0/op1/a'}, step_size=dt, vectorize=True, backend=b, clear=True,
                            file_name='net6')

        # calculate node behavior from hand
        targets = np.zeros((sim_steps, 3), dtype=np.float32)
        update2 = lambda x: x + dt * (4. + np.tanh(0.5))
        for i in range(sim_steps-1):
            targets[i + 1, 0] = update0(targets[i, 0])
            targets[i + 1, 1] = update2(targets[i, 1])
            targets[i + 1, 2] = update1(targets[i, 2], targets[i, 0] + targets[i, 1])

        diff = results['a'].values[:] - targets[:, 2]
        assert np.mean(np.abs(diff)) == pytest.approx(0., rel=accuracy, abs=accuracy)

        # test correct numerical evaluation of node with 1 source operator projecting to 2 independent targets
        ######################################################################################################

        results = integrate("model_templates.test_resources.test_backend.net7", simulation_time=sim_time,
                            outputs={'a': 'pop0/op1/a', 'b': 'pop0/op3/b'}, step_size=dt, vectorize=True,
                            backend=b, clear=True, file_name='net7')

        # calculate node behavior from hand
        targets = np.zeros((sim_steps, 4), dtype=np.float32)
        update3 = lambda a, b, u: a + dt * (-10. * a + b ** 2 + u)
        update4 = lambda x, y: x + dt * 0.1 * y
        for i in range(sim_steps-1):
            targets[i + 1, 0] = update0(targets[i, 0])
            targets[i + 1, 1] = update1(targets[i, 1], targets[i, 0])
            targets[i + 1, 2] = update3(targets[i, 2], targets[i, 3], targets[i, 0])
            targets[i + 1, 3] = update4(targets[i, 3], targets[i, 2])

        diff = np.mean(np.abs(results['a'].values[:] - targets[:, 1])) + \
               np.mean(np.abs(results['b'].values[:] - targets[:, 3]))
        assert diff == pytest.approx(0., rel=accuracy, abs=accuracy)


def test_2_3_edge():
    """Testing edge functionality of compute graph class.

    See Also
    --------
    :method:`add_edge`: Detailed documentation of add_edge method of `ComputeGraph`class.

    """

    dt = 1e-1
    sim_time = 10.
    sim_steps = int(np.round(sim_time/dt))
    inp = np.zeros((sim_steps, 1)) + 0.5

    final_results_comparison = []
    for b in backends:

        # test correct numerical evaluation of graph with 1 source projecting unidirectional to 2 target nodes
        ######################################################################################################

        # calculate edge behavior from hand
        update0 = lambda x, y: x + dt * y * 0.5
        update1 = lambda x, y: x + dt * (y + 2.0)
        update2 = lambda x, y: x + dt * (y - x)
        targets = np.zeros((sim_steps, 4), dtype=np.float32)
        for i in range(sim_steps-1):
            targets[i + 1, 0] = update0(targets[i, 0], targets[i, 1])
            targets[i + 1, 1] = update1(targets[i, 1], targets[i, 0])
            targets[i + 1, 2] = update2(targets[i, 2], targets[i, 0] * 2.0)
            targets[i + 1, 3] = update2(targets[i, 3], targets[i, 0] * 0.5)

        # simulate edge behavior
        results = integrate("model_templates.test_resources.test_backend.net8", simulation_time=sim_time,
                            outputs={'a': 'pop1/op1/a', 'b': 'pop2/op1/a'}, step_size=dt, vectorize=False,
                            backend=b, clear=True, file_name='net8')

        diff = np.mean(np.abs(results['a'].values[:] - targets[:, 2])) + \
               np.mean(np.abs(results['b'].values[:] - targets[:, 3]))
        assert diff == pytest.approx(0., rel=accuracy, abs=accuracy)

        # test correct numerical evaluation of graph with 2 bidirectionaly coupled nodes
        ################################################################################

        results = integrate("model_templates.test_resources.test_backend.net9", simulation_time=sim_time,
                            outputs={'a': 'pop0/op1/a', 'b': 'pop1/op7/a'}, inputs={'pop1/op7/inp': inp},
                            step_size=dt, vectorize=False, backend=b, clear=True, file_name='net9')

        # calculate edge behavior from hand
        update3 = lambda x, y, z: x + dt * (y + z - x)
        targets = np.zeros((sim_steps, 2), dtype=np.float32)
        for i in range(sim_steps-1):
            targets[i + 1, 0] = update2(targets[i, 0], targets[i, 1] * 0.5)
            targets[i + 1, 1] = update3(targets[i, 1], targets[i, 0] * 2.0, inp[i])

        diff = np.mean(np.abs(results['a'].values[:] - targets[:, 0])) + \
               np.mean(np.abs(results['b'].values[:] - targets[:, 1]))
        assert diff == pytest.approx(0., rel=accuracy, abs=accuracy)

        # test correct numerical evaluation of graph with 2 bidirectionally delay-coupled nodes
        #######################################################################################

        results = integrate("model_templates.test_resources.test_backend.net10", simulation_time=sim_time,
                            outputs={'a': 'pop0/op8/a', 'b': 'pop1/op8/a'}, step_size=dt, vectorize=False,
                            backend=b, clear=True, file_name='net10')

        # calculate edge behavior from hand
        delay0 = int(0.5 / dt)
        delay1 = int(1. / dt)
        targets = np.zeros((sim_steps, 2), dtype=np.float32)
        update4 = lambda x, y: x + dt * (2.0 + y)
        for i in range(sim_steps-1):
            inp0 = 0. if i < delay0 else targets[i - delay0, 1]
            inp1 = 0. if i < delay1 else targets[i - delay1, 0]
            targets[i + 1, 0] = update4(targets[i, 0], inp0 * 0.5)
            targets[i + 1, 1] = update4(targets[i, 1], inp1 * 2.0)

        diff = np.mean(np.abs(results['a'].values[:] - targets[:, 0])) + \
               np.mean(np.abs(results['b'].values[:] - targets[:, 1]))
        assert diff == pytest.approx(0., rel=accuracy, abs=accuracy)

        # test correct numerical evaluation of graph with delay distributions
        #####################################################################

        results = integrate("model_templates.test_resources.test_backend.net13", simulation_time=sim_time,
                            outputs={'a1': 'p1/op9/a', 'a2': 'p2/op10/a'}, inputs={'p1/op9/I_ext': inp},
                            vectorize=False, step_size=dt, backend=b, solver='euler', clear=True,
                            file_name='net11')
        final_results_comparison.append(results.values)

    if len(final_results_comparison) > 1:
        r0 = final_results_comparison[0]
        final_comparison = np.mean([r0 - r for r in final_results_comparison[1:]])
        assert final_comparison == pytest.approx(0.0, rel=accuracy, abs=accuracy)


def test_2_4_solver():
    """Testing different numerical solvers of pyrates.

    See Also
    --------
    :method:`_solve`: Detailed documentation of how to numerical integration is performed by the `NumpyBackend`.
    :method:`run`: Detailed documentation of the method that needs to be called to solve differential equations in the
    `NumpyBackend`.
    """

    # define input
    dt = 1e-4
    dts = 1e-1
    sim_time = 20.
    sim_steps = int(np.round(sim_time / dt, decimals=0))
    inp = np.zeros((sim_steps, 1)) + 0.5

    for b in backends:

        # standard euler solver (trusted)
        r = integrate("model_templates.test_resources.test_backend.net13", simulation_time=sim_time,
                      outputs={'a1': 'p1/op9/a', 'a2': 'p2/op10/a'}, inputs={'p1/op9/I_ext': inp},
                      vectorize=False, step_size=dt, backend=b, solver='euler', clear=True, file_name='euler_solver',
                      sampling_step_size=dts)

        # scipy solver (tested)
        r2 = integrate("model_templates.test_resources.test_backend.net13", simulation_time=sim_time,
                       outputs={'a1': 'p1/op9/a', 'a2': 'p2/op10/a'}, inputs={'p1/op9/I_ext': inp}, method='RK23',
                       vectorize=False, step_size=dt, backend=b, solver='scipy', clear=True, file_name='scipy_solver',
                       sampling_step_size=dts)

        # Heun's method (tested)
        r3 = integrate("model_templates.test_resources.test_backend.net13", simulation_time=sim_time,
                       outputs={'a1': 'p1/op9/a', 'a2': 'p2/op10/a'}, inputs={'p1/op9/I_ext': inp},
                       vectorize=False, step_size=dt, backend=b, solver='heun', clear=True, file_name='heun_solver',
                       sampling_step_size=dts)

        assert np.mean(r.loc[:, 'a2'].values - r2.loc[:, 'a2'].values) == pytest.approx(0., rel=accuracy, abs=accuracy)
        assert np.mean(r.loc[:, 'a2'].values - r3.loc[:, 'a2'].values) == pytest.approx(0., rel=accuracy, abs=accuracy)


def test_2_5_inputs_outputs():
    """Tests the input-output interface of the run method in circuits of different hierarchical depth.

    See Also
    -------
    :method:`CircuitIR.run` detailed documentation of how to use the arguments `inputs` and `outputs`.

    """

    dt = 1e-3
    dts = 1e-1
    sim_time = 10.
    sim_steps = int(np.round(sim_time / dt, decimals=0))
    inp = np.zeros((sim_steps, 1)) + 0.5

    for b in backends:

        # define inputs and outputs for each population separately
        ##########################################################

        # perform simulation
        r1 = integrate("model_templates.test_resources.test_backend.net13", simulation_time=sim_time,
                       outputs={'a1': 'p1/op9/a'}, inputs={'p1/op9/I_ext': inp}, vectorize=True, step_size=dt,
                       backend=b, solver='euler', clear=True, file_name='inout_1', sampling_step_size=dts)

        # define input and output for both populations simultaneously
        #############################################################

        # perform simulation
        r2 = integrate("model_templates.test_resources.test_backend.net13", simulation_time=sim_time,
                       outputs=['all/op9/a'], inputs={'all/op9/I_ext': inp}, vectorize=True, step_size=dt, backend=b,
                       solver='euler', clear=True, file_name='inout_2', sampling_step_size=dts)

        assert np.mean(r1.values.flatten() - r2.values.flatten()) == pytest.approx(0., rel=accuracy, abs=accuracy)

        # repeat in a network with 2 hierarchical levels of node organization
        #####################################################################

        # define input
        inp2 = np.zeros((sim_steps, 1)) + 0.1

        # perform simulation
        r1 = integrate("model_templates.test_resources.test_backend.net14", simulation_time=sim_time, vectorize=True,
                       step_size=dt, backend=b, solver='euler', clear=True, sampling_step_size=dts,
                       outputs={'a1': 'c1/p1/op9/a', 'a2': 'c1/p2/op10/a', 'a3': 'c2/p1/op9/a', 'a4': 'c2/p2/op10/a'},
                       inputs={'c1/p1/op9/I_ext': inp, 'c1/p2/op10/I_ext': inp2, 'c2/p1/op9/I_ext': inp,
                               'c2/p2/op10/I_ext': inp2}, file_name='inout_3')

        # perform simulation
        r2 = integrate("model_templates.test_resources.test_backend.net14", simulation_time=sim_time,
                       outputs={'a1': 'all/all/op9/a', 'a2': 'all/all/op10/a'},
                       inputs={'all/all/op9/I_ext': inp, 'all/all/op10/I_ext': inp2},
                       vectorize=True, step_size=dt, backend=b, solver='euler', clear=True, file_name='inout_4',
                       sampling_step_size=dts)

        assert np.mean(r1.values.flatten() - r2.values.flatten()) == pytest.approx(0., rel=accuracy, abs=accuracy)


def test_2_6_vectorization():
    """Tests whether a Jansen-Rit-based circuit with and without vectorization of mathematical operations yields
    identical results.

    See Also
    --------
    :method:`CircuitTemplate.run` for a documentation of the keyword argument `vectorize`.
    """

    dt = 1e-4
    dts = 1e-2
    T = 1.0
    inp = np.zeros((int(np.round(T/dt)),)) + 220.0

    for i, b in enumerate(backends):

        # simulation without vectorization of the network equations
        r1 = integrate("model_templates.neural_mass_models.jansenrit.JRC_2delaycoupled", vectorize=False,
                       inputs={"jrc2/pc/rpo_e_in/u": inp}, outputs={"r": "jrc1/ein/rpo_e/v"}, backend=b,
                       solver='euler', step_size=dt, clear=True, simulation_time=T, sampling_step_size=dts,
                       file_name=f'novec{i + 1}')

        # simulation with vectorized network equations
        r2 = integrate("model_templates.neural_mass_models.jansenrit.JRC_2delaycoupled", vectorize=True,
                       inputs={"jrc2/pc/rpo_e_in/u": inp}, outputs={"r": "jrc1/ein/rpo_e/v"}, backend=b,
                       solver='euler', step_size=dt, clear=True, simulation_time=T, sampling_step_size=dts,
                       file_name=f'vec{i + 1}')

        assert np.mean(r1.values - r2.values) == pytest.approx(0.0, rel=accuracy, abs=accuracy)


def test_2_7_backends():
    """Tests the whether different backends produce comparable results when simulating the dynamics of different models.

    See Also
    -------
    :method:`CircuitIR.__init__` for documentation of the available backend options.
    """

    dt = 5e-4
    dts = 1e-3
    T = 10.

    r0 = integrate("model_templates.neural_mass_models.qif.qif_sfa", simulation_time=T, sampling_step_size=dts,
                   inputs=None, outputs={"r": "p/qif_sfa_op/r"}, solver='euler', step_size=dt, clear=True,
                   file_name='m0', vectorize=False)

    for i, b in enumerate(backends):

        if b != 'default':

            r = integrate("model_templates.neural_mass_models.qif.qif_sfa",
                          inputs=None, outputs={"r": "p/qif_sfa_op/r"}, backend=b, solver='euler', step_size=dt,
                          clear=True, simulation_time=T, sampling_step_size=dts, file_name=f'm{i+1}', vectorize=False)

            assert np.mean(r0.values - r.values) == pytest.approx(0.0, rel=accuracy, abs=accuracy)
