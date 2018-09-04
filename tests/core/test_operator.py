"""Test suite for basic operator module functionality.
"""

# external imports
import numpy as np
import tensorflow as tf
import pytest

# pyrates internal imports
from pyrates.operator import Operator
from pyrates.node import Node
from pyrates.edge import Edge
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
    """Testing operator functionality:

    See Also
    --------
    :class:`Operator`: Detailed documentation of operator attributes and methods.
    """

    # test minimal call example
    ###########################

    expressions = ["a = 5.0 - 2.0"]

    op = Operator(expressions=expressions, expression_args={'a': tf.Variable(0.)}, key='test',
                  variable_scope='test_scope')
    assert isinstance(op, Operator)

    # test dependencies between equations of operator
    #################################################

    eq1 = "d/dt * a = a^2"
    eq2 = "b = a / sum(a)"

    gr = tf.Graph()
    with gr.as_default():
        v1 = tf.Variable(np.ones((1, 10))*2., dtype=tf.float32)
        v2 = tf.Variable(np.ones((1, 10))*2., dtype=tf.float32)

    args1 = {'a': v1, 'dt': .5}
    args2 = {'a': v2, 'dt': .5}
    op1 = Operator([eq1, eq2], args1, 'test1', 'test_scope1', tf_graph=gr).create()
    op2 = Operator([eq2, eq1], args2, 'test2', 'test_scope2', tf_graph=gr).create()

    with tf.Session(graph=gr) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(op1)
        result1 = np.sum(args1['b'].eval())
        sess.run(op2)
        result2 = np.sum(args2['b'].eval())

    assert result1 == pytest.approx(1., rel=1e-6)
    assert result2 == pytest.approx(1., rel=1e-6)


def test_2_2_node():
    """Testing node functionality:

    See Also
    --------
    :class:`Node`: Detailed documentation of operator attributes and methods.
    """

    # test minimal call example
    ###########################

    ops = {'operator_1': ["a[:] = 2"]}
    op_args = {'a': {'variable_type': 'state_variable',
                     'shape': (1, 10),
                     'name': 'a',
                     'data_type': 'float32',
                     'initial_value': 1.},
               }

    node = Node(ops, op_args, 'test')
    assert isinstance(node, Node)

    # test dependencies between operations of node
    ##############################################

    op1 = ["d/dt * a = a^2", "b = a*2"]
    op2 = ["c = b / sum(b)"]

    op_args['dt'] = {'variable_type': 'constant',
                     'shape': (),
                     'name': 'dt',
                     'data_type': 'float32',
                     'initial_value': 0.5}
    op_args1 = op_args.copy()
    op_args2 = op_args.copy()
    op_args2['b'] = {'variable_type': 'state_variable',
                     'shape': (1, 10),
                     'name': 'b',
                     'data_type': 'float32',
                     'initial_value': 1.}
    gr = tf.Graph()

    n1 = Node({'operator_1': op1, 'operator_2': op2}, op_args1, 'test1', tf_graph=gr)
    n2 = Node({'operator_1': op2, 'operator_2': op1}, op_args2, 'test2', tf_graph=gr)

    with tf.Session(graph=gr) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(n1.update)
        result1 = np.sum(n1.c.eval())
        sess.run(n2.update)
        result2 = np.sum(n2.c.eval())

    assert result1 == pytest.approx(1., rel=1e-6)
    assert result1 == result2


def test_2_3_edge():
    """Testing edge functionality:

    See Also
    --------
    :class:`Edge`: Detailed documentation of edge attributes and methods.
    """

    # test minimal call example
    ###########################

    gr = tf.Graph()
    ops = {'operator_1': ["a[:] = 1."]}
    op_args = {'a': {'variable_type': 'state_variable',
                     'shape': (1, 10),
                     'name': 'a',
                     'data_type': 'float32',
                     'initial_value': 1.}
               }
    n1 = Node(ops, op_args, 'n1', tf_graph=gr)
    n2 = Node(ops, op_args, 'n2', tf_graph=gr)

    edge_op = ["inp = out*2"]
    edge_args = {'out': {'variable_type': 'source_var',
                         'name': 'a'},
                 'inp': {'variable_type': 'target_var',
                         'name': 'a'}
                 }
    edge = Edge(n1, n2, edge_op, edge_args, 'test', tf_graph=gr)
    assert isinstance(edge, Edge)

    # test correct projection and dependencies between operations of edge
    #####################################################################

    gr = tf.Graph()

    n1 = Node(ops, op_args, 'n1', tf_graph=gr)
    n2 = Node(ops, op_args, 'n2', tf_graph=gr)

    eq1 = "c = c_ / sum(c_,1)"
    eq2 = "inp = out*c"
    edge_args = {'out': {'variable_type': 'source_var',
                         'name': 'a'},
                 'inp': {'variable_type': 'target_var',
                         'name': 'a'},
                 'c_': {'variable_type': 'state_variable',
                        'shape': (1, 10),
                        'name': 'c_',
                        'data_type': 'float32',
                        'initial_value': 5.}
                 }
    e1 = Edge(n1, n2, [eq1, eq2], edge_args, 'e1', tf_graph=gr)

    edge_args['c'] = {'variable_type': 'state_variable',
                      'shape': (1, 10),
                      'name': 'c',
                      'data_type': 'float32',
                      'initial_value': 5.}
    e2 = Edge(n2, n1, [eq2, eq1], edge_args, 'e2', tf_graph=gr)

    with tf.Session(graph=gr) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(e1.project)
        result1 = [np.sum(n1.a.eval()), np.sum(n2.a.eval())]

    with tf.Session(graph=gr) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(e2.project)
        result2 = [np.sum(n1.a.eval()), np.sum(n2.a.eval())]

    assert result1 == [10., 1.]
    assert result2 == [50., 10.]


def test_2_4_network():
    """Testing network functionality:

    See Also
    --------
    :class:`Network`: Detailed documentation of network attributes and methods.
    """

    # test minimal call example
    ###########################

    node_dict = {'n1': {'operator_1': ["d/dt * a = a^2"],
                        'a': {'variable_type': 'state_variable',
                              'shape': (1, 10),
                              'name': 'a',
                              'data_type': 'float32',
                              'initial_value': 1.},
                        }
                 }

    net = Network(node_dict, {}, dt=0.1)
    assert isinstance(net, Network)

    # test for correct simulation behavior of network
    #################################################

    # network test
    gr = tf.Graph()
    node_dict = {'n1': {'operator_1': ["d/dt * a = b^2"],
                        'a': {'variable_type': 'state_variable',
                              'shape': (1, 10),
                              'name': 'a',
                              'data_type': 'float32',
                              'initial_value': 1.},
                        'b': {'variable_type': 'state_variable',
                              'shape': (1, 10),
                              'name': 'b',
                              'data_type': 'float32',
                              'initial_value': 1.},
                        },
                 'n2': {'operator_1': ["d/dt * a = b^2"],
                        'a': {'variable_type': 'state_variable',
                              'shape': (1, 10),
                              'name': 'a',
                              'data_type': 'float32',
                              'initial_value': 1.},
                        'b': {'variable_type': 'state_variable',
                              'shape': (1, 10),
                              'name': 'b',
                              'data_type': 'float32',
                              'initial_value': 1.},
                        }
                 }
    conn_dict = {'coupling_operators': [["inp = out*c"], ["inp[0] = out[0]/c[0]"]],
                 'coupling_operator_args': {'out': {'variable_type': 'source_var',
                                                    'name': 'a'},
                                            'inp': {'variable_type': 'target_var',
                                                    'name': 'b'},
                                            'c': {'variable_type': 'constant',
                                                  'shape': (1, 10),
                                                  'name': 'c',
                                                  'data_type': 'float32',
                                                  'initial_value': 0.5},
                                            },
                 'sources': ['n1', 'n2'],
                 'targets': ['n2', 'n1']
                 }

    net = Network(nodes=node_dict, edges=conn_dict, tf_graph=gr, key='net', dt=0.1)
    results, _ = net.run(1., outputs={'a1': net.nodes['n1']['handle'].a, 'a2': net.nodes['n2']['handle'].a})

    # target results
    gr2 = tf.Graph()
    ops = {'operator_1': ["d/dt * a = b^2"]}
    op_args = {'a': {'variable_type': 'state_variable',
                     'shape': (1, 10),
                     'name': 'a',
                     'data_type': 'float32',
                     'initial_value': 1.},
               'b': {'variable_type': 'state_variable',
                     'shape': (1, 10),
                     'name': 'b',
                     'data_type': 'float32',
                     'initial_value': 1.},
               'dt': {'variable_type': 'constant',
                      'shape': (),
                      'name': 'dt',
                      'data_type': 'float32',
                      'initial_value': 0.1}
               }
    n1 = Node(ops, op_args.copy(), 'n1', tf_graph=gr2)
    n2 = Node(ops, op_args.copy(), 'n2', tf_graph=gr2)

    eq1 = "inp = out*c"
    eq2 = "inp[0] = out[0]/c[0]"
    edge_args = {'out': {'variable_type': 'source_var',
                         'name': 'a'},
                 'inp': {'variable_type': 'target_var',
                         'name': 'b'},
                 'c': {'variable_type': 'constant',
                       'shape': (1, 10),
                       'name': 'c',
                       'data_type': 'float32',
                       'initial_value': 0.5},
                 }
    e1 = Edge(n1, n2, [eq1], edge_args, 'e1', tf_graph=gr2)
    e2 = Edge(n2, n1, [eq2], edge_args, 'e2', tf_graph=gr2)

    targets = []
    with tf.Session(graph=gr2) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10):
            sess.run(n1.update)
            sess.run(n2.update)
            sess.run(e1.project)
            sess.run(e2.project)
            targets.append([n1.a.eval(), n2.a.eval()])

    error = nmrse(results.values, np.reshape(np.array(targets), (10, 20)))
    assert np.sum(error) == pytest.approx(0., rel=1e-6)
