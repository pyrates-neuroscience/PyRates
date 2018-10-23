"""Test suite for basic operator module functionality.
"""

# external imports
import numpy as np
import tensorflow as tf
from networkx import MultiDiGraph
import pytest
from copy import deepcopy

# pyrates internal imports
from pyrates.network import Network
from pyrates.utility import nmrse

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

    # test dependencies between equations of operator
    #################################################

    eq1 = "d/dt * a = a + 2."
    eq2 = "b = a / sum(a)"

    # version 1 [eq1 -> eq2]
    gr1 = MultiDiGraph()
    gr1.add_node('n',
                 operators={'op1': {'equations': [eq1, eq2], 'inputs': {}, 'output': 'a'}},
                 operator_args={'op1/a': {'vtype': 'state_var', 'dtype': 'float32', 'shape': (10,), 'value': 0.}},
                 operator_order=['op1'],
                 inputs={}
                 )
    net1 = Network(net_config=gr1, key='net1', vectorize='none')

    # version 2 [eq2 -> eq1]
    gr2 = MultiDiGraph()
    gr2.add_node('n',
                 operators={'op1': {'equations': [eq2, eq1], 'inputs': {}, 'output': 'a'}},
                 operator_args={'op1/a': {'vtype': 'state_var', 'dtype': 'float32', 'shape': (10,), 'value': 0.}},
                 operator_order=['op1'],
                 inputs={}
                 )
    net2 = Network(net_config=gr2, key='test_op', vectorize='none', tf_graph=tf.Graph())

    result1, _ = net1.run(outputs={'b1': ('n', 'op1', 'b'), 'a1': ('n', 'op1', 'a')})
    result2, _ = net2.run(outputs={'b2': ('n', 'op1', 'b')})

    assert result1['a1'].values[-1] == pytest.approx(0.002, rel=1e-6)
    assert np.sum(result1['b1'].values[:]) == pytest.approx(1., rel=1e-6)
    assert result1['b1'].values[-1] == result2['b2'].values[-1]


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

    net = Network(gr, vectorize='none', key='test_node', tf_graph=tf.Graph())
    results, _ = net.run(outputs={'r1': ('n1', 'op_2', 'c'), 'r2': ('n1', 'op_2', 'c')})

    assert results['r1'].values[-1] != pytest.approx(1., rel=1e-2)
    assert np.sum(results['r1'].values) == pytest.approx(1., rel=1e-6)
    assert results['r1'].values[-1] == results['r2'].values[-1]


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

    op_args1 = {'op_1/a': {'vtype': 'state_var',
                           'shape': (10,),
                           'dtype': 'float32',
                           'value': np.random.rand(10)},
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
                           'value': np.random.rand(10)},
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
    gr.add_edge('n1', 'n2', source_var='op_1/b', target_var='op_1/inp', delay=0, weight=1.)

    net = Network(gr, vectorize='none', key='test_edge', dt=1e-3, tf_graph=tf.Graph())
    results, _ = net.run(simulation_time=3e-3, outputs={'r1': ('n1', 'op_1', 'b'), 'r2': ('n2', 'op_1', 'b')})

    assert results['r1_0'].values[-1] > 0.
    assert results['r1_0'].values[-1] < 1.
    assert results['r1_0'].values[-1] < results['r2_0'].values[-1]


def test_2_4_vectorization():
    """Testing vectorization functionality of ComputeGraph class.

    See Also
    --------
    :method:`_vectorize`: Detailed documentation of vectorize method of `ComputeGraph` class.
    """

    # test whether vectorized networks produce same output as non-vectorized network
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
    gr1.add_edge('n1', 'n2', source_var='op_2/c', target_var='op_1/inp', delay=0, weight=1.)

    gr2 = MultiDiGraph()
    gr2.add_node('n1', **deepcopy(n1))
    gr2.add_node('n2', **deepcopy(n2))
    gr2.add_edge('n1', 'n2', source_var='op_2/c', target_var='op_1/inp', delay=0, weight=1.)

    gr3 = MultiDiGraph()
    gr3.add_node('n1', **deepcopy(n1))
    gr3.add_node('n2', **deepcopy(n2))
    gr3.add_edge('n1', 'n2', source_var='op_2/c', target_var='op_1/inp', delay=0, weight=1.)

    net1 = Network(gr1, vectorize='none', key='no_vec', dt=1e-3, tf_graph=tf.Graph())
    net2 = Network(gr2, vectorize='nodes', key='node_vec', dt=1e-3, tf_graph=tf.Graph())
    net3 = Network(gr3, vectorize='ops', key='op_vec', dt=1e-3, tf_graph=tf.Graph())
    results1, _ = net1.run(simulation_time=5e-3, outputs={'r1': ('n1', 'op_2', 'c'), 'r2': ('n2', 'op_2', 'c')})
    results2, _ = net2.run(simulation_time=5e-3, outputs={'r1': ('n1', 'op_2', 'c'), 'r2': ('n2', 'op_2', 'c')})
    results3, _ = net3.run(simulation_time=5e-3, outputs={'r1': ('n1', 'op_2', 'c'), 'r2': ('n2', 'op_2', 'c')})

    error1 = nmrse(results1.values, results2.values)
    error2 = nmrse(results1.values, results3.values)

    assert np.sum(results1.values) > 0.
    assert np.sum(error1) == pytest.approx(0., rel=1e-6)
    assert np.sum(error2) == pytest.approx(0., rel=1e-6)
