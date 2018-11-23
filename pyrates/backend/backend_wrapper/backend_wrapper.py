"""This module provides wrappers for different backends that are needed by the parser class.
"""

# external imports
import tensorflow as tf
import time as t

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


class TensorflowBackend(tf.Graph):
    """

    """

    def __init__(self):

        super().__init__()

        # define operations and datatypes of the backend
        ################################################

        # base math operations
        self.ops = {"+": tf.add,
                    "-": tf.subtract,
                    "*": tf.multiply,
                    "/": tf.divide,
                    "%": tf.mod,
                    "^": tf.pow,
                    "@": tf.matmul,
                    ".T": tf.transpose,
                    ".I": tf.matrix_inverse,
                    ">": tf.greater,
                    "<": tf.less,
                    "==": tf.equal,
                    "!=": tf.not_equal,
                    ">=": tf.greater_equal,
                    "<=": tf.less_equal,
                    "=": tf.assign,
                    "+=": tf.assign_add,
                    "neg": lambda x: -x,
                    "sin": tf.sin,
                    "cos": tf.cos,
                    "tan": tf.tan,
                    "abs": tf.abs,
                    "sqrt": tf.sqrt,
                    "sq": tf.square,
                    "exp": tf.exp,
                    "max": tf.maximum,
                    "min": tf.minimum,
                    "argmax": tf.arg_max,
                    "argmin": tf.arg_min,
                    "round": tf.round,
                    "roundto": lambda x, y: tf.round(x * 10**y) / 10**y,
                    "sum": tf.reduce_sum,
                    "concat": tf.concat,
                    "reshape": tf.reshape,
                    "shape": tf.shape,
                    "dtype": tf.keras.backend.dtype,
                    'squeeze': tf.squeeze,
                    "cast": tf.cast,
                    "randn": tf.random_normal,
                    "ones": tf.ones,
                    "zeros":tf.zeros,
                    "softmax": tf.nn.softmax,
                    "sigmoid": tf.sigmoid,
                    "tanh": tf.tanh,
                    "gather": tf.gather_nd,
                    "scatter": tf.scatter_nd,
                    "scatter_update": tf.scatter_update,
                    "mask": tf.boolean_mask,
                    "stack": tf.parallel_stack,
                    "group": tf.group
                    }

        self.dtypes = {"float16": tf.float16,
                       "float32": tf.float32,
                       "float64": tf.float64,
                       "int16": tf.int16,
                       "int32": tf.int32,
                       "int64": tf.int64,
                       "uint16": tf.uint16,
                       "uint32": tf.uint32,
                       "uint64": tf.uint64,
                       "complex64": tf.complex64,
                       "complex128": tf.complex128,
                       "bool": tf.bool
                       }

        self.existing_scopes = {}

    def run(self, steps, ops, inputs, outputs, sampling_steps=None, sampling_ops=None, out_dir=None):
        """

        Parameters
        ----------
        simulation_time
        ops
        sampling_steps
        sampling_ops
        out_dir

        Returns
        -------

        """

        # initialize session log
        if out_dir:
            writer = tf.summary.FileWriter(out_dir, graph=self)

        # start session
        with tf.Session(graph=self) as sess:

            # initialize variables on graph
            sess.run(tf.global_variables_initializer())

            # simulate backend behavior for each time-step
            t_start = t.time()
            if type(ops) is list and type(sampling_ops) is list:
                for step in range(steps):
                    for op in ops:
                        sess.run(op, inputs[step])
                    if step % sampling_steps == 0:
                        for sop in sampling_ops:
                            sess.run(sop)
            elif type(ops) is list:
                for step in range(steps):
                    for op in ops:
                        sess.run(op, inputs[step])
                    if step % sampling_steps == 0:
                        sess.run(sampling_ops)
            elif type(sampling_ops) is list:
                for step in range(steps):
                    sess.run(ops, inputs[step])
                    if step % sampling_steps == 0:
                        for sop in sampling_ops:
                            sess.run(sop)
            else:
                for step in range(steps):
                    sess.run(ops, inputs[step])
                    if step % sampling_steps == 0:
                        sess.run(sampling_ops)
            t_end = t.time()

            # close session log
            if out_dir:
                writer.close()

            # store output variables in output dictionary
            for i, (key, var) in enumerate(outputs.items()):
                outputs[key] = var.eval(sess)

        return outputs, t_end - t_start

    def add_variable(self, name, value, shape, dtype, scope=None):
        """

        Parameters
        ----------
        name
        value
        shape
        dtype
        scope

        Returns
        -------

        """

        scope, reuse = self.get_scope(scope) if scope else tf.get_variable_scope()
        with self.as_default():
            with scope as sc:
                if reuse:
                    with tf.name_scope(sc.original_name_scope):
                        return tf.get_variable(name, shape, dtype, initializer=tf.constant_initializer(value))
                else:
                    return tf.get_variable(name, shape, dtype, initializer=tf.constant_initializer(value))

    def add_constant(self, name, value, shape, dtype, scope=None):
        """

        Parameters
        ----------
        name
        value
        shape
        dtype
        scope

        Returns
        -------

        """

        scope, reuse = self.get_scope(scope)
        with self.as_default():
            with scope as sc:
                if reuse:
                    with tf.name_scope(sc.original_name_scope):
                        return tf.constant(value, dtype, shape, name)
                else:
                    return tf.constant(value, dtype, shape, name)

    def add_placeholder(self, name, shape, dtype, scope=None):
        """

        Parameters
        ----------
        name
        shape
        dtype
        scope

        Returns
        -------

        """

        scope, reuse = self.get_scope(scope)
        with self.as_default():
            with scope as sc:
                if reuse:
                    with tf.name_scope(sc.original_name_scope):
                        return tf.placeholder(dtype, shape, name)
                else:
                    return tf.placeholder(dtype, shape, name)

    def add_op(self, op: str, *args, **kwargs):
        """

        Parameters
        ----------
        op
        args
        kwargs

        Returns
        -------

        """

        scope = kwargs.pop('scope', None)
        scope, reuse = self.get_scope(scope)
        with self.as_default():
            with scope as sc:
                if reuse:
                    with tf.name_scope(sc.original_name_scope):
                        try:
                            return self.ops[op](*args, **kwargs)
                        except TypeError:
                            return self.ops[op](list(args), **kwargs)
                else:
                    try:
                        return self.ops[op](*args, **kwargs)
                    except TypeError:
                        return self.ops[op](list(args), **kwargs)

    def get_scope(self, scope):
        """

        Parameters
        ----------
        scope

        Returns
        -------

        """

        with self.as_default():
            if scope is None:
                return tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE)
            if scope not in self.existing_scopes.keys():
                self.existing_scopes[scope] = True
            return tf.variable_scope(scope, reuse=tf.AUTO_REUSE), self.existing_scopes[scope]
