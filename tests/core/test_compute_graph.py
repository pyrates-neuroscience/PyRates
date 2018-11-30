"""Test suite for basic operator module functionality.
"""

# external imports
import numpy as np
import tensorflow as tf
from networkx import MultiDiGraph
import pytest
from copy import deepcopy

# pyrates internal imports
from pyrates.backend import ComputeGraph
from pyrates.utility import nmrse
from pyrates.frontend.circuit import CircuitTemplate

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


#########
# Tests #
#########


def test_2_1_operator():
    """Testing operator functionality of compute graph class:

    See Also
    --------
    :method:`add_operator`: Detailed documentation of method for adding operations to instance of `ComputeGraph`.
    """

    # test correct numerical evaluation of operator with two simple, linear equations
    #################################################################################

    # create net config from YAML file
    net_config0 = CircuitTemplate.from_yaml("pyrates.examples.test_compute_graph.net0").apply()

    # instantiate compute graph from net config
    dt = 1e-1
    net0 = ComputeGraph(net_config=net_config0, name='net0', vectorize='none', dt=dt)

    # simulate operator behavior
    sim_time = 10.0
    results0, _ = net0.run(sim_time, outputs={'a': ('pop0.0', 'op0.0', 'a')})

    # generate target values
    sim_steps = int(sim_time/dt)
    update0_1 = lambda x: x*0.5
    update0_0 = lambda x: x + 2.0
    targets0 = np.zeros((sim_steps+1, 2), dtype=np.float32)
    for i in range(sim_steps):
        targets0[i+1, 0] = update0_0(targets0[i, 1])
        targets0[i+1, 1] = update0_1(targets0[i, 0])

    # compare results with target values
    diff0 = results0['a'].values - targets0[1:, 1].T
    assert np.mean(np.abs(diff0)) == pytest.approx(0., rel=1e-1, abs=1e-5)

    # test correct numerical evaluation of operator with a single differential equation and external input
    ######################################################################################################

    # set up operator in pyrates
    net_config1 = CircuitTemplate.from_yaml("pyrates.examples.test_compute_graph.net1").apply()
    net1 = ComputeGraph(net_config=net_config1, name='net1', vectorize='none', dt=dt)

    # define input
    inp = np.zeros((sim_steps, 1)) + 0.5

    # simulate operator behavior
    results1, _ = net1.run(sim_time, inputs={('pop1.0', 'op1.0', 'u'): inp}, outputs={'a': ('pop1.0', 'op1.0', 'a')})

    # calculate operator behavior from hand
    update1 = lambda x, y: x + dt*(y-x)
    targets1 = np.zeros((sim_steps + 1, 1), dtype=np.float32)
    for i in range(sim_steps):
        targets1[i+1] = update1(targets1[i], inp[i])

    diff1 = results1['a'].values - targets1[1:].T
    assert np.mean(np.abs(diff1)) == pytest.approx(0., rel=1e-4, abs=1e-6)

    # test correct numerical evaluation of operator with a two equations (1 ODE, 1 linear eq.)
    ##########################################################################################

    net_config2 = CircuitTemplate.from_yaml("pyrates.examples.test_compute_graph.net2").apply()
    net2 = ComputeGraph(net_config=net_config2, name='net2', vectorize='none', dt=dt)
    results2, _ = net2.run(sim_time, outputs={'a': ('pop2.0', 'op2.0', 'a')})

    # calculate operator behavior from hand
    update2 = lambda x: 1./(1. + np.exp(-x))
    targets2 = np.zeros((sim_steps + 1, 2), dtype=np.float32)
    for i in range(sim_steps):
        targets2[i+1, 1] = update2(targets2[i, 0])
        targets2[i+1, 0] = update1(targets2[i, 0], targets2[i, 1])

    diff2 = results2['a'].values - targets2[1:, 0].T
    assert np.mean(np.abs(diff2)) == pytest.approx(0., rel=1e-4, abs=1e-6)

    # test correct numerical evaluation of operator with a two coupled DEs and two simple equations
    ###############################################################################################

    net_config3 = CircuitTemplate.from_yaml("pyrates.examples.test_compute_graph.net3").apply()
    net3 = ComputeGraph(net_config=net_config3, name='net3', vectorize='none', dt=dt)
    results3, _ = net3.run(sim_time,
                           outputs={'b': ('pop3.0', 'op3.0', 'b'),
                                    'a': ('pop3.0', 'op3.0', 'a')},
                           inputs={('pop3.0', 'op3.0', 'u'): inp},
                           out_dir="/tmp/log")

    # calculate operator behavior from hand
    update3_0 = lambda a, b, u: a + dt*(-10.*a + b**2 + u)
    update3_1 = lambda b, a: b + dt*a
    targets3 = np.zeros((sim_steps + 1, 2), dtype=np.float32)
    for i in range(sim_steps):
        targets3[i+1, 0] = update3_0(targets3[i, 0], targets3[i, 1], inp[i])
        targets3[i+1, 1] = update3_1(targets3[i, 1], targets3[i, 0])

    diff3 = results3['a'].values - targets3[1:, 0].T
    assert np.mean(np.abs(diff3)) == pytest.approx(0., rel=1e-4, abs=1e-6)


def test_2_2_node():
    """Testing node functionality of compute graph class.

    See Also
    --------
    :method:`add_node`: Detailed documentation of method for adding nodes to instance of `ComputeGraph`.
    """

    # test dependencies between operations of node
    ##############################################

    ops1 = {'op_1': {'equations': ["d/dt * a = a^2", "b = a*2"], 'inputs': {}, 'output': 'b'},
            'op_2': {'equations': ["c = b / sum(b)"], 'inputs': {'b': {'sources': ['op_1'], 'reduce_dim': False}},
                     'output': 'c'}
            }
    ops2 = {'op_1': {'equations': ["d/dt * a = a^2", "b = a*2"], 'inputs': {}, 'output': 'b'},
            'op_2': {'equations': ["c = b / sum(b)"], 'inputs': {}, 'output': 'c'}
            }

    op_args = {'op_1/a': {'vtype': 'state_var',
                          'shape': (10,),
                          'dtype': 'float32',
                          'value': 0.5}}
    op_args1 = op_args.copy()
    op_args2 = op_args.copy()
    op_args2['op_2/b'] = {'vtype': 'state_var',
                          'shape': (10,),
                          'dtype': 'float32',
                          'value': 1.}
    op_args2['op_1/b'] = {'vtype': 'state_var',
                          'shape': (10,),
                          'dtype': 'float32',
                          'value': 1.}

    n1 = {'operators': ops1,
          'operator_args': op_args1,
          'operator_order': ['op_1', 'op_2'],
          'inputs': {}}
    n2 = {'operators': ops2,
          'operator_args': op_args2,
          'operator_order': ['op_2', 'op_1'],
          'inputs': {}}

    gr = MultiDiGraph()
    gr.add_node('n1', **n1)
    gr.add_node('n2', **n2)

    net = ComputeGraph(gr, vectorize='none', key='test_node', tf_graph=tf.Graph())
    results, _ = net.run(outputs={'r1': ('n1', 'op_2', 'c'), 'r2': ('n1', 'op_2', 'c')})

    assert results['r1_0'].values[-1] == results['r1_0'].values[0]
    assert np.sum(results.values[0, :10]) == pytest.approx(1., rel=1e-6)
    assert results['r1_0'].values[-1] == results['r2_0'].values[-1]


def test_2_3_edge():
    """Testing edge functionality of compute graph class.

    See Also
    --------
    :method:`add_edge`: Detailed documentation of add_edge method of `ComputeGraph`class.

    """

    # test correct projection and dependencies between operations of edge
    #####################################################################

    ops1 = {'op_1': {'equations': ["d/dt * a = -a", "b = a*2."], 'inputs': {}, 'output': 'b'},
            'op_2': {'equations': ["c = b / sum(b)"], 'inputs': {'b': {'sources': ['op_1'], 'reduce_dim': False}},
                     'output': 'c'}
            }
    ops2 = {'op_1': {'equations': ["d/dt * a = -a + inp", "b = a*2"], 'inputs': {}, 'output': 'b'},
            'op_2': {'equations': ["c = b / sum(b)"], 'inputs': {'b': {'sources': ['op_1'], 'reduce_dim': False}},
                     'output': 'c'}
            }

    a = np.random.rand(10)
    op_args1 = {'op_1/a': {'vtype': 'state_var',
                           'shape': (10,),
                           'dtype': 'float32',
                           'value': a},
                'op_1/b': {'vtype': 'state_var',
                           'shape': (10,),
                           'dtype': 'float32',
                           'value': 0.},
                'op_2/c': {'vtype': 'state_var',
                           'shape': (10,),
                           'dtype': 'float32',
                           'value': 1.}}
    op_args2 = {'op_1/a': {'vtype': 'state_var',
                           'shape': (10,),
                           'dtype': 'float32',
                           'value': a},
                'op_1/b': {'vtype': 'state_var',
                           'shape': (10,),
                           'dtype': 'float32',
                           'value': 0.},
                'op_2/c': {'vtype': 'state_var',
                           'shape': (10,),
                           'dtype': 'float32',
                           'value': 1.},
                'op_1/inp': {'vtype': 'state_var',
                             'shape': (10,),
                             'dtype': 'float32',
                             'value': 0.}
                }

    n1 = {'operators': ops1,
          'operator_args': op_args1,
          'operator_order': ['op_1', 'op_2'],
          'inputs': {}}
    n2 = {'operators': ops2,
          'operator_args': op_args2,
          'operator_order': ['op_1', 'op_2'],
          'inputs': {}}

    gr = MultiDiGraph()
    gr.add_node('n1', **n1)
    gr.add_node('n2', **n2)
    gr.add_edge('n1', 'n2', source_var='op_1/b', target_var='op_1/inp', delay=None, weight=1.)

    net = ComputeGraph(gr, vectorize='none', key='test_edge', dt=1e-3, tf_graph=tf.Graph())
    results, _ = net.run(simulation_time=3e-3, outputs={'r1': ('n1', 'op_1', 'b'), 'r2': ('n2', 'op_1', 'b')})

    assert results['r1_0'].values[-1] > 0.
    assert results['r1_0'].values[-1] < results['r2_0'].values[-1]


def test_2_4_vectorization():
    """Testing vectorization functionality of ComputeGraph class.

    See Also
    --------
    :method:`_vectorize`: Detailed documentation of vectorize method of `ComputeGraph` class.
    """

    # test whether vectorized networks produce same output as non-vectorized backend
    ################################################################################

    ops = {'op_1': {'equations': ["d/dt * a = -a + inp", "b = a*2."], 'inputs': {}, 'output': 'b'},
           'op_2': {'equations': ["c = b / 0.5"], 'inputs': {'b': {'sources': ['op_1'], 'reduce_dim': False}},
                    'output': 'c'}
           }

    n1 = {'operators': deepcopy(ops),
          'operator_args': {'op_1/a': {'vtype': 'state_var',
                                       'shape': (),
                                       'dtype': 'float32',
                                       'value': np.random.rand()},
                            'op_2/c': {'vtype': 'state_var',
                                       'shape': (),
                                       'dtype': 'float32',
                                       'value': 1.},
                            'op_1/inp': {'vtype': 'state_var',
                                         'shape': (),
                                         'dtype': 'float32',
                                         'value': 0.}
                            },
          'operator_order': ['op_1', 'op_2'],
          'inputs': {}}
    n2 = {'operators': deepcopy(ops),
          'operator_args': {'op_1/a': {'vtype': 'state_var',
                                       'shape': (),
                                       'dtype': 'float32',
                                       'value': np.random.rand()},
                            'op_2/c': {'vtype': 'state_var',
                                       'shape': (),
                                       'dtype': 'float32',
                                       'value': 1.},
                            'op_1/inp': {'vtype': 'state_var',
                                         'shape': (),
                                         'dtype': 'float32',
                                         'value': 0.}
                },
          'operator_order': ['op_1', 'op_2'],
          'inputs': {}}

    gr1 = MultiDiGraph()
    gr1.add_node('n1', **deepcopy(n1))
    gr1.add_node('n2', **deepcopy(n2))
    gr1.add_edge('n1', 'n2', source_var='op_2/c', target_var='op_1/inp', delay=None, weight=1.)

    gr2 = MultiDiGraph()
    gr2.add_node('n1', **deepcopy(n1))
    gr2.add_node('n2', **deepcopy(n2))
    gr2.add_edge('n1', 'n2', source_var='op_2/c', target_var='op_1/inp', delay=None, weight=1.)

    gr3 = MultiDiGraph()
    gr3.add_node('n1', **deepcopy(n1))
    gr3.add_node('n2', **deepcopy(n2))
    gr3.add_edge('n1', 'n2', source_var='op_2/c', target_var='op_1/inp', delay=None, weight=1.)

    net1 = ComputeGraph(gr1, vectorize='none', key='no_vec', dt=1e-3, tf_graph=tf.Graph())
    net2 = ComputeGraph(gr2, vectorize='nodes', key='node_vec', dt=1e-3, tf_graph=tf.Graph())
    net3 = ComputeGraph(gr3, vectorize='ops', key='op_vec', dt=1e-3, tf_graph=tf.Graph())
    results1, _ = net1.run(simulation_time=1., outputs={'r1': ('n1', 'op_2', 'c'), 'r2': ('n2', 'op_2', 'c')})
    results2, _ = net2.run(simulation_time=1., outputs={'r1': ('n1', 'op_2', 'c'), 'r2': ('n2', 'op_2', 'c')})
    results3, _ = net3.run(simulation_time=1., outputs={'r1': ('n1', 'op_2', 'c'), 'r2': ('n2', 'op_2', 'c')})

    results1.pop('time'), results2.pop('time'), results3.pop('time')

    error1 = nmrse(results1.values, results2.values)
    error2 = nmrse(results1.values, results3.values)

    assert np.sum(results1.values) > 0.
    assert np.mean(error1) == pytest.approx(0., rel=1e-5)
    assert np.mean(error2) == pytest.approx(0., rel=1e-5)
