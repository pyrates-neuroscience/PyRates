"""

"""

# external imports
from typing import List
import tensorflow as tf

# pyrates internal imports
from pyrates.parser import RHSParser, LHSParser

# meta infos
__author__ = "Richard Gast"
__status__ = "Development"


class Operator(object):

    def __init__(self, expressions: List[str], expression_args: dict, tf_graph: tf.Graph, key: str,
                 variable_scope: str):

        self.expressions = []
        self.key = key
        self.tf_graph = tf_graph

        with self.tf_graph.as_default():

            with tf.variable_scope(variable_scope):

                for i, expr in enumerate(expressions):

                    lhs, rhs = expr.split('=')

                    # right-hand side parsing
                    #########################

                    rhs_parser = RHSParser(rhs, expression_args, tf_graph)
                    rhs_tf_op = rhs_parser.parse()

                    # left-hand-side parsing
                    ########################

                    lhs_parser = LHSParser(lhs, expression_args, rhs_tf_op, tf_graph)
                    lhs_tf_op, _ = lhs_parser.parse()

                    self.expressions.append(lhs_tf_op)

    def create(self):

        with self.tf_graph.as_default():
            grouped_expressions = tf.group(self.expressions, name=self.key)

        return grouped_expressions


#gr = tf.Graph()
# with gr.as_default():
#     m_out = tf.get_variable(name='m_out', shape=(), dtype=tf.float32, initializer=tf.constant_initializer(0.))
#     v = tf.get_variable(name='v', shape=(), dtype=tf.float32, initializer=tf.constant_initializer(0.))
#     r = tf.constant(560., dtype=tf.float32, name='r')
#     m_max = tf.constant(5., dtype=tf.float32, name='m_max')
#     v_th = tf.constant(0.006, dtype=tf.float32, name='v_th')
#     v_update = v.assign(0.01)
#
# op = Operator(["d/dt * m_out = m_max / (1 + exp(r * (v_th - v)))"],
#               {'m_out': m_out, 'm_max': m_max, 'v': v, 'r': r, 'v_th': v_th, 'dt': 0.1},
#               gr)

# with gr.as_default():
#     x = tf.get_variable(name='x', shape=(), dtype=tf.float32, initializer=tf.constant_initializer(0.))
#     v = tf.get_variable(name='v', shape=(), dtype=tf.float32, initializer=tf.constant_initializer(0.))
#     H = tf.constant(3.25e-3, dtype=tf.float32, name='H')
#     tau = tf.constant(10e-3, dtype=tf.float32, name='tau')
#     dt = tf.constant(1e-3, dtype=tf.float32, name='dt')
#     m = tf.constant(500., dtype=tf.float32, name='m')
#
# op = Operator(["d/dt * x = H/tau * m - 2/tau * x - 1/tau**2 * v", "d/dt * v = x"],
#               {'x': x, 'v': v, 'H': H, 'tau': tau, 'dt': dt, 'm': m},
#               gr, key='test', variable_scope='test1')
#
# steps = 1000
# vs = []
# with tf.Session(graph=gr) as sess:
#     writer = tf.summary.FileWriter('/tmp/log/', graph=gr)
#     sess.run(tf.global_variables_initializer())
#
#     for step in range(steps):
#         sess.run(op.evaluate)
#         vs.append(v.eval())
#
#     writer.close()
#
# from matplotlib.pyplot import *
# plot(vs)
# show()
