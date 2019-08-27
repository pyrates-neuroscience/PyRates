"""Test suite for basic operator module functionality.
"""

# external imports
from typing import Union

import numpy as np
import pytest

# pyrates internal imports
from pyrates.backend import ComputeGraph
from pyrates.frontend import CircuitTemplate

# meta infos
__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


###########
# Utility #
###########


def setup_module():
    print("\n")
    print("================================")
    print("| Test Suite 2 : SimNet Module |")
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


def test_2_1_operator():
    """Testing operator functionality of compute graph class:

    See Also
    --------
    :method:`add_operator`: Detailed documentation of method for adding operations to instance of `ComputeGraph`.
    """

    backends = ["numpy", "tensorflow"]

    for b in backends:

        # test correct numerical evaluation of operator with two coupled simple, linear equations
        #########################################################################################

        # create net config from YAML file
        net_config = CircuitTemplate.from_yaml("model_templates.test_resources.test_compute_graph.net0").apply()

        # instantiate compute graph from net config
        dt = 1e-1
        net = ComputeGraph(net_config=net_config, name='net0', vectorization='none', dt=dt, backend=b)

        # simulate operator behavior
        sim_time = 10.0
        results = net.run(sim_time, outputs={'a': 'pop0/op0/a'})

        # generate target values
        sim_steps = int(sim_time / dt)
        update0_1 = lambda x: x * 0.5
        update0_0 = lambda x: x + 2.0
        targets = np.zeros((sim_steps + 1, 2), dtype=np.float32)
        for i in range(sim_steps):
            targets[i + 1, 0] = update0_0(targets[i, 1])
            targets[i + 1, 1] = update0_1(targets[i, 0])

        # compare results with target values
        diff = results['a'].values[:, 0] - targets[:-1, 1]
        assert np.mean(np.abs(diff)) == pytest.approx(0., rel=1e-6, abs=1e-6)

        # test correct numerical evaluation of operator with a single differential equation and external input
        ######################################################################################################

        # set up operator in pyrates
        net_config = CircuitTemplate.from_yaml("model_templates.test_resources.test_compute_graph.net1").apply()
        net = ComputeGraph(net_config=net_config, name='net1', vectorization='none', dt=dt, backend=b)

        # define input
        inp = np.zeros((sim_steps, 1)) + 0.5

        # simulate operator behavior
        results = net.run(sim_time, inputs={'pop0/op1/u': inp}, outputs={'a': 'pop0/op1/a'})

        # calculate operator behavior from hand
        update1 = lambda x, y: x + dt * (y - x)
        targets = np.zeros((sim_steps + 1, 1), dtype=np.float32)
        for i in range(sim_steps):
            targets[i + 1] = update1(targets[i], inp[i])

        diff = results['a'].values[1:] - targets[:-2]
        assert np.mean(np.abs(diff)) == pytest.approx(0., rel=1e-6, abs=1e-6)

        # test correct numerical evaluation of operator with two coupled equations (1 ODE, 1 linear eq.)
        ################################################################################################

        net_config = CircuitTemplate.from_yaml("model_templates.test_resources.test_compute_graph.net2").apply()
        net = ComputeGraph(net_config=net_config, name='net2', vectorization='none', dt=dt, backend=b)
        results = net.run(sim_time, outputs={'a': 'pop0/op2/a'})

        # calculate operator behavior from hand
        update2 = lambda x: 1. / (1. + np.exp(-x))
        targets = np.zeros((sim_steps + 1, 2), dtype=np.float32)
        for i in range(sim_steps):
            targets[i + 1, 0] = update1(targets[i, 0], targets[i, 1])
            targets[i + 1, 1] = update2(targets[i, 0])

        diff = results['a'].values[:, 0] - targets[:-1, 0]
        assert np.mean(np.abs(diff)) == pytest.approx(0., rel=1e-6, abs=1e-6)

        # test correct numerical evaluation of operator with a two coupled DEs
        ######################################################################

        net_config = CircuitTemplate.from_yaml("model_templates.test_resources.test_compute_graph.net3").apply()
        net = ComputeGraph(net_config=net_config, name='net3', vectorization='none', dt=dt, backend=b)
        results = net.run(sim_time,
                          outputs={'b': 'pop0/op3/b'},
                          inputs={'pop0/op3/u': inp},
                          out_dir="/tmp/log")

        # calculate operator behavior from hand
        update3_0 = lambda a, b, u: a + dt * (-10. * a + b ** 2 + u)
        update3_1 = lambda b, a: b + dt * 0.1 * a
        targets = np.zeros((sim_steps + 1, 2), dtype=np.float32)
        for i in range(sim_steps):
            targets[i + 1, 0] = update3_0(targets[i, 0], targets[i, 1], inp[i])
            targets[i + 1, 1] = update3_1(targets[i, 1], targets[i, 0])

        diff = results['b'].values[1:, 0] - targets[:-2, 1]
        assert np.mean(np.abs(diff)) == pytest.approx(0., rel=1e-6, abs=1e-6)


def test_2_2_node():
    """Testing node functionality of compute graph class.

    See Also
    --------
    :method:`add_node`: Detailed documentation of method for adding nodes to instance of `ComputeGraph`.
    """

    backends = ['numpy', 'tensorflow']

    accuracy = 1e-4

    for b in backends:

        # test correct numerical evaluation of node with 2 operators, where op1 projects to op2
        #######################################################################################

        # set up node in pyrates
        dt = 1e-1
        sim_time = 10.
        sim_steps = int(sim_time / dt)
        net_config = CircuitTemplate.from_yaml("model_templates.test_resources.test_compute_graph.net4").apply()
        net = ComputeGraph(net_config=net_config, name='net0', vectorization=True, dt=dt, backend=b)

        # simulate node behavior
        results = net.run(sim_time, outputs={'a': 'pop0/op1/a'})

        # calculate node behavior from hand
        update0 = lambda x: x + dt * 2.
        update1 = lambda x, y: x + dt * (y - x)
        targets = np.zeros((sim_steps + 1, 2), dtype=np.float32)
        for i in range(sim_steps):
            targets[i + 1, 0] = update0(targets[i, 0])
            targets[i + 1, 1] = update1(targets[i, 1], targets[i + 1, 0])

        diff = results['a'].values[:, 0] - targets[:-1, 1]
        assert np.mean(np.abs(diff)) == pytest.approx(0., rel=accuracy, abs=accuracy)

        # test correct numerical evaluation of node with 2 independent operators
        ########################################################################

        net_config = CircuitTemplate.from_yaml("model_templates.test_resources.test_compute_graph.net5").apply()
        net = ComputeGraph(net_config=net_config, name='net1', vectorization='none', dt=dt, backend=b)

        # simulate node behavior
        results = net.run(sim_time, outputs={'a': 'pop0/op5/a'})

        # calculate node behavior from hand
        targets = np.zeros((sim_steps + 1, 2), dtype=np.float32)
        for i in range(sim_steps):
            targets[i + 1, 0] = update0(targets[i, 0])
            targets[i + 1, 1] = update1(targets[i, 1], 0.)

        diff = results['a'].values[:, 0] - targets[:-1, 1]
        assert np.mean(np.abs(diff)) == pytest.approx(0., rel=accuracy, abs=accuracy)

        # test correct numerical evaluation of node with 2 independent operators projecting to the same target operator
        ###############################################################################################################

        net_config = CircuitTemplate.from_yaml("model_templates.test_resources.test_compute_graph.net6").apply()
        net = ComputeGraph(net_config=net_config, name='net2', vectorization='none', dt=dt, backend=b)
        results = net.run(sim_time, outputs={'a': 'pop0/op1/a'})

        # calculate node behavior from hand
        targets = np.zeros((sim_steps + 1, 3), dtype=np.float32)
        update2 = lambda x: x + dt * (4. + np.tanh(0.5))
        for i in range(sim_steps):
            targets[i + 1, 0] = update0(targets[i, 0])
            targets[i + 1, 1] = update2(targets[i, 1])
            targets[i + 1, 2] = update1(targets[i, 2], targets[i + 1, 0] + targets[i + 1, 1])

        diff = results['a'].values[:, 0] - targets[:-1, 2]
        assert np.mean(np.abs(diff)) == pytest.approx(0., rel=accuracy, abs=accuracy)

        # test correct numerical evaluation of node with 1 source operator projecting to 2 independent targets
        ######################################################################################################

        net_config = CircuitTemplate.from_yaml("model_templates.test_resources.test_compute_graph.net7").apply()
        net = ComputeGraph(net_config=net_config, name='net3', vectorization='none', dt=dt, backend=b)
        results = net.run(sim_time, outputs={'a': 'pop0/op1/a', 'b': 'pop0/op3/b'})

        # calculate node behavior from hand
        targets = np.zeros((sim_steps + 1, 4), dtype=np.float32)
        update3 = lambda a, b, u: a + dt * (-10. * a + b ** 2 + u)
        update4 = lambda x, y: x + dt * 0.1 * y
        for i in range(sim_steps):
            targets[i + 1, 0] = update0(targets[i, 0])
            targets[i + 1, 1] = update1(targets[i, 1], targets[i + 1, 0])
            targets[i + 1, 2] = update3(targets[i, 2], targets[i, 3], targets[i + 1, 0])
            targets[i + 1, 3] = update4(targets[i, 3], targets[i, 2])

        diff = np.mean(np.abs(results['a'].values[1:, 0] - targets[:-2, 1])) + \
               np.mean(np.abs(results['b'].values[1:, 0] - targets[:-2, 3]))
        assert diff == pytest.approx(0., rel=accuracy, abs=accuracy)


def test_2_3_edge():
    """Testing edge functionality of compute graph class.

    See Also
    --------
    :method:`add_edge`: Detailed documentation of add_edge method of `ComputeGraph`class.

    """

    backends = ['numpy', 'tensorflow']
    accuracy = 1e-4

    for b in backends:

        # test correct numerical evaluation of graph with 1 source projecting unidirectional to 2 target nodes
        ######################################################################################################

        # set up edge in pyrates
        dt = 1e-1
        sim_time = 10.
        sim_steps = int(sim_time / dt)
        net_config = CircuitTemplate.from_yaml("model_templates.test_resources.test_compute_graph.net8").apply()
        net = ComputeGraph(net_config=net_config, name='net0', vectorization=True, dt=dt, backend=b)

        # calculate edge behavior from hand
        update0 = lambda x: x * 0.5
        update1 = lambda x: x + 2.0
        update2 = lambda x, y: x + dt * (y - x)
        targets = np.zeros((sim_steps + 1, 4), dtype=np.float32)
        for i in range(sim_steps):
            targets[i + 1, 0] = update0(targets[i, 1])
            targets[i + 1, 1] = update1(targets[i, 0])
            targets[i + 1, 2] = update2(targets[i, 2], targets[i, 0] * 2.0)
            targets[i + 1, 3] = update2(targets[i, 3], targets[i, 0] * 0.5)

        # simulate edge behavior
        results = net.run(sim_time, outputs={'a': 'pop1/op1/a', 'b': 'pop2/op1/a'})

        diff = np.mean(np.abs(results['a']['pop1/op1'].values[:] - targets[:-1, 2])) + \
               np.mean(np.abs(results['b']['pop2/op1'].values[:] - targets[:-1, 3]))
        assert diff == pytest.approx(0., rel=accuracy, abs=accuracy)

        # test correct numerical evaluation of graph with 2 bidirectionaly coupled nodes
        ################################################################################

        # define input
        inp = np.zeros((sim_steps, 1)) + 0.5

        net_config = CircuitTemplate.from_yaml("model_templates.test_resources.test_compute_graph.net9").apply()
        net = ComputeGraph(net_config=net_config, name='net1', vectorization=True, dt=dt, backend=b)
        results = net.run(sim_time, outputs={'a': 'pop0/op1/a',
                                             'b': 'pop1/op7/a'},
                          inputs={'pop1/op7/inp': inp})

        # calculate edge behavior from hand
        update3 = lambda x, y, z: x + dt * (y + z - x)
        targets = np.zeros((sim_steps + 1, 2), dtype=np.float32)
        for i in range(sim_steps):
            targets[i + 1, 0] = update2(targets[i, 0], targets[i, 1] * 0.5)
            targets[i + 1, 1] = update3(targets[i, 1], targets[i, 0] * 2.0, inp[i])

        diff = np.mean(np.abs(results['a']['pop0/op1'].values[1:] - targets[:-2, 0])) + \
               np.mean(np.abs(results['b']['pop1/op7'].values[1:] - targets[:-2, 1]))
        assert diff == pytest.approx(0., rel=accuracy, abs=accuracy)

        # test correct numerical evaluation of graph with 2 bidirectionally delay-coupled nodes
        #######################################################################################

        net_config = CircuitTemplate.from_yaml("model_templates.test_resources.test_compute_graph.net10").apply()
        net = ComputeGraph(net_config=net_config, name='net2', vectorization=True, dt=dt, backend=b)
        results = net.run(sim_time, outputs={'a': 'pop0/op8/a',
                                             'b': 'pop1/op8/a'})

        # calculate edge behavior from hand
        delay0 = int(0.5 / dt)
        delay1 = int(1. / dt)
        targets = np.zeros((sim_steps + 1, 2), dtype=np.float32)
        update4 = lambda y: 2.0 + y
        for i in range(sim_steps):
            inp0 = 0. if i < delay0 else targets[i - delay0, 1]
            inp1 = 0. if i < delay1 else targets[i - delay1, 0]
            targets[i + 1, 0] = update4(inp0 * 0.5)
            targets[i + 1, 1] = update4(inp1 * 2.0)

        diff = np.mean(np.abs(results['a']['pop0/op8'].values[1:] - targets[:-2, 0])) + \
               np.mean(np.abs(results['b']['pop1/op8'].values[1:] - targets[:-2, 1]))
        assert diff == pytest.approx(0., rel=accuracy, abs=accuracy)

        # test correct numerical evaluation of graph with 2 unidirectionally, multi-delay-coupled nodes
        ###############################################################################################

        # define input
        inp = np.zeros((sim_steps, 1)) + 0.5

        net_config = CircuitTemplate.from_yaml("model_templates.test_resources.test_compute_graph.net9").apply()
        net = ComputeGraph(net_config=net_config, name='net3', vectorization='none', dt=dt, backend=b)
        results = net.run(sim_time, outputs={'a': 'pop0/op1/a',
                                             'b': 'pop1/op7/a'},
                          inputs={'pop1/op7/inp': inp})

        # calculate edge behavior from hand
        update3 = lambda x, y, z: x + dt * (y + z - x)
        targets = np.zeros((sim_steps + 1, 2), dtype=np.float32)
        for i in range(sim_steps):
            targets[i + 1, 0] = update2(targets[i, 0], targets[i, 1] * 0.5)
            targets[i + 1, 1] = update3(targets[i, 1], targets[i, 0] * 2.0, inp[i])

        diff = np.mean(np.abs(results['a']['pop0/op1'].values[1:] - targets[:-2, 0])) + \
               np.mean(np.abs(results['b']['pop1/op7'].values[1:] - targets[:-2, 1]))
        assert diff == pytest.approx(0., rel=accuracy, abs=accuracy)


# @pytest.mark.skip
def test_2_4_vectorization():
    """Testing vectorization functionality of ComputeGraph class.

    See Also
    --------
    :method:`_vectorize`: Detailed documentation of vectorize method of `ComputeGraph` class.
    """

    backends = ['tensorflow', 'numpy']
    for b in backends:
        # test whether vectorized networks produce same output as non-vectorized backend
        ################################################################################

        # define simulation params
        dt = 1e-2
        sim_time = 10.
        sim_steps = int(sim_time / dt)
        inp = np.zeros((sim_steps, 2)) + 0.5

        # set up networks
        net_config0 = CircuitTemplate.from_yaml("model_templates.test_resources.test_compute_graph.net12").apply()
        net_config1 = CircuitTemplate.from_yaml("model_templates.test_resources.test_compute_graph.net12").apply()
        net0 = ComputeGraph(net_config=net_config0, name='net0', vectorization='none', dt=dt, build_in_place=False,
                            backend=b)
        net1 = ComputeGraph(net_config=net_config1, name='net1', vectorization='nodes', dt=dt, build_in_place=False,
                            backend=b)

        # simulate network behaviors
        results0 = net0.run(sim_time, outputs={'a': 'pop0/op7/a', 'b': 'pop1/op7/a'},
                            inputs={'all/op7/inp': inp})
        results1 = net1.run(sim_time, outputs={'a': 'pop0/op7/a', 'b': 'pop1/op7/a'},
                            inputs={'all/op7/inp': inp}, out_dir='/tmp/log')

        error1 = nmrse(results0.values, results1.values)

        assert np.sum(results1.values) > 0.
        assert np.mean(error1) == pytest.approx(0., rel=1e-6, abs=1e-6)
