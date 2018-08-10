"""Test suite for basic operator module functionality.
"""

# external imports
import numpy as np
import tensorflow as tf
import pytest

# pyrates internal imports
from pyrates.operator import Operator

# meta infos
__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"

###########
# Utility #
###########


def setup_module():
    print("\n")
    print("==================================")
    print("| Test Suite 1 : Operator Module |")
    print("==================================")


#########
# Tests #
#########

def test_1_1_operator():
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
    eq2 = "a = a / sum(a,1)"

    gr = tf.Graph()
    with gr.as_default():
        v1 = tf.Variable(np.ones((1, 10))*5., dtype=tf.float32)
        v2 = tf.Variable(np.ones((1, 10))*5., dtype=tf.float32)

    op1 = Operator([eq1, eq2], {'a': v1, 'dt': .5}, 'test1', 'test_scope1', tf_graph=gr).create()
    op2 = Operator([eq2, eq1], {'a': v2, 'dt': .5}, 'test2', 'test_scope2', tf_graph=gr).create()

    with tf.Session(graph=gr) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(op1)
        result1 = np.sum(v1.eval())
        sess.run(op2)
        result2 = np.sum(v2.eval())

    assert result1 == pytest.approx(1., rel=1e-6)
    assert result2 > result1
