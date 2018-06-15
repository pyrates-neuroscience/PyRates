"""

"""

# external imports
from typing import Dict, List, Optional
import tensorflow as tf

# pyrates imports
from pyrates.operator import Operator
from pyrates.parser import parse_dict

# meta infos
__author__ = "Richard Gast"
__status__ = "Development"


class Node(object):

    def __init__(self,
                 operations: Dict[str, List[str]],
                 operation_args: dict,
                 key: str,
                 tf_graph: Optional[tf.Graph] = None
                 ) -> None:

        self.key = key
        self.operations = dict()
        self.tf_graph = tf_graph if tf_graph else tf.get_default_graph()

        with self.tf_graph.as_default():

            with tf.variable_scope(self.key):

                tf_vars, var_names = parse_dict(operation_args, self.key, self.tf_graph)

                operator_args = dict()

                for tf_var, var_name in zip(tf_vars, var_names):
                    setattr(self, var_name, tf_var)
                    operator_args[var_name] = tf_var

                tf_ops = []

                for op_name, op in operations.items():

                    self.operations[op_name] = op
                    operator = Operator(expressions=op,
                                        expression_args=operator_args,
                                        tf_graph=self.tf_graph,
                                        key=op_name,
                                        variable_scope=self.key)
                    tf_ops.append(operator.create())

                self.update = tf.tuple(tf_ops, name='update')


# ops = {'rpo': ["d/dt * x = H/tau * inp - 2/tau * x - 1/tau**2 * v",
#                "d/dt * v = x"
#                ],
#        'rpo_sum': ["V = sum(v)"],
#        'pro': ["m = m_max / (1 + exp(r * (v_th - v)))"]}
# args = {'dt': {'variable_type': 'constant',
#                'name': 'dt',
#                'data_type': 'float32',
#                'shape': (),
#                'initial_value': 1e-3},
#         'v': {'variable_type': 'state_variable',
#               'name': 'v',
#               'data_type': 'float32',
#               'shape': (),
#               'initial_value': 0.},
#         'v_e': {'variable_type': 'state_variable',
#               'name': 'v_e',
#               'data_type': 'float32',
#               'shape': (),
#               'initial_value': 0.},
#         'H': {'variable_type': 'constant',
#               'name': 'H',
#               'data_type': 'float32',
#               'shape': (),
#               'initial_value': 3.25e-3},
#         'x': {'variable_type': 'state_variable',
#               'name': 'x',
#               'data_type': 'float32',
#               'shape': (),
#               'initial_value': 0.},
#         'tau': {'variable_type': 'constant',
#                 'name': 'tau',
#                 'data_type': 'float32',
#                 'shape': (),
#                 'initial_value': 10.0e-3},
#         'm': {'variable_type': 'state_variable',
#               'name': 'm',
#               'data_type': 'float32',
#               'shape': (),
#               'initial_value': 0.
#               },
#         'm_max': {'variable_type': 'constant',
#                   'name': 'm_max',
#                   'data_type': 'float32',
#                   'shape': (),
#                   'initial_value': 5.},
#         'r': {'variable_type': 'constant',
#               'name': 'r',
#               'data_type': 'float32',
#               'shape': (),
#               'initial_value': 560.},
#         'v_th': {'variable_type': 'constant',
#                  'name': 'v_th',
#                  'data_type': 'float32',
#                  'shape': (),
#                  'initial_value': 6.0e-3},
#         'inp': {'variable_type': 'placeholder',
#                 'name': 'inp',
#                 'data_type': 'float32',
#                 'shape': ()},
#         }
#
# gr = tf.Graph()
# EINs = Node(ops, args, 'EINs', gr)
#
# import numpy as np
# potentials = []
# rates = []
# inp = np.random.uniform(120, 320, 1000)
# with tf.Session(graph=gr) as sess:
#     sess.run(tf.global_variables_initializer())
#     for step in range(1000):
#         sess.run(EINs.update, {EINs.inp: inp[step]})
#         potentials.append(EINs.v.eval())
#         rates.append(EINs.m.eval())
#
# from matplotlib.pyplot import *
# potentials = (np.array(potentials) - np.mean(potentials))
# potentials = potentials / np.std(potentials)
# rates = (np.array(rates) - np.mean(rates))
# rates = rates / np.std(rates)
# plot(potentials)
# plot(rates)
