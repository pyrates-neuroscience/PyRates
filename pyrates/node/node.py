"""

"""

# external imports
from typing import Dict, List, Optional
import tensorflow as tf

# pyrates imports
from pyrates.operator import Operator

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

                data_types = {'float16': tf.float16,
                              'float32': tf.float32,
                              'float64': tf.float64,
                              'int16': tf.int16,
                              'int32': tf.int32,
                              'int64': tf.int64,
                              'double': tf.double,
                              'complex64': tf.complex64,
                              'complex128': tf.complex128,
                              'string': tf.string,
                              'bool': tf.bool}

                operator_args = dict()

                for arg in operation_args.values():

                    if isinstance(arg, tf.Variable) or isinstance(arg, tf.Tensor) or isinstance(arg, tf.Operation)\
                            or isinstance(arg, tf.IndexedSlices):

                        node_var = arg

                    elif arg['variable_type'] == 'state_variable':

                        node_var = tf.get_variable(name=arg['name'],
                                                   shape=arg['shape'],
                                                   dtype=data_types[arg['data_type']],
                                                   initializer=tf.constant_initializer(arg['initial_value'])
                                                   )

                    elif arg['variable_type'] == 'constant':

                        node_var = tf.constant(value=arg['initial_value'],
                                               name=arg['name'],
                                               shape=arg['shape'],
                                               dtype=data_types[arg['data_type']]
                                               )

                    elif arg['variable_type'] == 'placeholder':

                        node_var = tf.placeholder(name=arg['name'],
                                                  shape=arg['shape'],
                                                  dtype=data_types[arg['data_type']]
                                                  )

                    else:

                        raise AttributeError('Variable type of each variable needs to be either `state_variable`,'
                                             ' `constant` or `placeholder`.')

                    setattr(self, arg['name'], node_var)
                    operator_args[arg['name']] = node_var

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


ops = {'rpo': ["d/dt * x = H/tau * inp - 2/tau * x - 1/tau**2 * v",
               "d/dt * v = x"
               ],
       'rpo_sum': ["V = sum(v)"],
       'pro': ["m = m_max / (1 + exp(r * (v_th - v)))"]}
args = {'dt': {'variable_type': 'constant',
               'name': 'dt',
               'data_type': 'float32',
               'shape': (),
               'initial_value': 1e-3},
        'v': {'variable_type': 'state_variable',
              'name': 'v',
              'data_type': 'float32',
              'shape': (),
              'initial_value': 0.},
        'v_e': {'variable_type': 'state_variable',
              'name': 'v_e',
              'data_type': 'float32',
              'shape': (),
              'initial_value': 0.},
        'H': {'variable_type': 'constant',
              'name': 'H',
              'data_type': 'float32',
              'shape': (),
              'initial_value': 3.25e-3},
        'x': {'variable_type': 'state_variable',
              'name': 'x',
              'data_type': 'float32',
              'shape': (),
              'initial_value': 0.},
        'tau': {'variable_type': 'constant',
                'name': 'tau',
                'data_type': 'float32',
                'shape': (),
                'initial_value': 10.0e-3},
        'm': {'variable_type': 'state_variable',
              'name': 'm',
              'data_type': 'float32',
              'shape': (),
              'initial_value': 0.
              },
        'm_max': {'variable_type': 'constant',
                  'name': 'm_max',
                  'data_type': 'float32',
                  'shape': (),
                  'initial_value': 5.},
        'r': {'variable_type': 'constant',
              'name': 'r',
              'data_type': 'float32',
              'shape': (),
              'initial_value': 560.},
        'v_th': {'variable_type': 'constant',
                 'name': 'v_th',
                 'data_type': 'float32',
                 'shape': (),
                 'initial_value': 6.0e-3},
        'inp': {'variable_type': 'placeholder',
                'name': 'inp',
                'data_type': 'float32',
                'shape': ()},
        }

gr = tf.Graph()
EINs = Node(ops, args, 'EINs', gr)

import numpy as np
potentials = []
rates = []
inp = np.random.uniform(120, 320, 1000)
with tf.Session(graph=gr) as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1000):
        sess.run(EINs.update, {EINs.inp: inp[step]})
        potentials.append(EINs.v.eval())
        rates.append(EINs.m.eval())

from matplotlib.pyplot import *
potentials = (np.array(potentials) - np.mean(potentials))
potentials = potentials / np.std(potentials)
rates = (np.array(rates) - np.mean(rates))
rates = rates / np.std(rates)
plot(potentials)
plot(rates)
