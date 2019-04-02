# -*- coding: utf-8 -*-
#
#
# PyRates software framework for flexible implementation of neural
# network model_templates and simulations. See also:
# https://github.com/pyrates-neuroscience/PyRates
#
# Copyright (C) 2017-2018 the original authors (Richard Gast and
# Daniel Rose), the Max-Planck-Institute for Human Cognitive Brain
# Sciences ("MPI CBS") and contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#
# CITATION:
#
# Richard Gast and Daniel Rose et. al. in preparation

"""Contains wrapper classes for different backends that are needed by the parser module.

A new backend needs to implement the following methods

Methods
-------
__init__
run
add_var
add_op
add_layer

Currently supported backends:
- Tensorflow: TensorflowBackend.

"""

# external imports
import time as t
from typing import Optional, Dict, Callable, List, Union, Tuple, Any
import numpy as np
import tensorflow as tf

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


class TensorflowBackend(tf.Graph):
    """Wrapper to tensorflow.

    Parameters
    ----------
    ops
        Additional operations this backend instance can perform, defined as key-value pairs.
    dtypes
        Additional data-types this backend instance can use, defined as key-value pairs.
    use_device
        Default default_device on which to place variables and operations.

    """

    def __init__(self,
                 ops: Optional[Dict[str, Callable]] = None,
                 dtypes: Optional[Dict[str, object]] = None,
                 use_device: str = 'cpu'
                 ) -> None:
        """Instantiates tensorflow backend, i.e. a tensorflow graph.
        """

        super().__init__()
        if use_device == 'cpu':
            device = '/cpu:0'
        elif use_device == 'gpu':
            device = '/gpu:0'
        else:
            device = use_device
        self.default_device = device

        # define operations and datatypes of the backend
        ################################################

        # base math operations
        self.ops = {"+": tf.add,
                    "-": tf.subtract,
                    "*": tf.multiply,
                    "/": tf.divide,
                    "%": tf.mod,
                    "^": tf.pow,
                    "**": tf.pow,
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
                    "cond": tf.cond,
                    "neg": lambda x: -x,
                    "sin": tf.sin,
                    "cos": tf.cos,
                    "tan": tf.tan,
                    "atan": tf.atan,
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
                    "mean": tf.reduce_mean,
                    "concat": tf.concat,
                    "reshape": tf.reshape,
                    "shape": tf.shape,
                    "dtype": tf.keras.backend.dtype,
                    'squeeze': tf.squeeze,
                    "roll": tf.roll,
                    "cast": tf.cast,
                    "randn": tf.random_normal,
                    "ones": tf.ones,
                    "zeros": tf.zeros,
                    "range": tf.range,
                    "softmax": tf.nn.softmax,
                    "sigmoid": tf.sigmoid,
                    "tanh": tf.tanh,
                    "gather": tf.gather,
                    "gather_nd": tf.gather_nd,
                    "scatter": tf.scatter_nd,
                    "scatter_update": tf.scatter_update,
                    "scatter_add": tf.scatter_nd_add,
                    "mask": tf.boolean_mask,
                    "stack": tf.stack,
                    "pstack": tf.parallel_stack,
                    "group": tf.group,
                    "tuple": tf.tuple,
                    "no_op": no_op,
                    }
        if ops:
            self.ops.update(ops)

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
        if dtypes:
            self.dtypes.update(dtypes)

        self.existing_scopes = {}

    def run(self,
            steps: int,
            ops: Union[tf.Operation, List[tf.Operation]],
            inputs: List[dict],
            outputs: Dict[str, tf.Variable],
            sampling_steps: Optional[int] = None,
            sampling_ops: Optional[Union[tf.Operation, List[tf.Operation]]] = None,
            out_dir: Optional[str] = None,
            profile: Optional[str] = None
            ) -> Union[Dict[str, tf.Variable], Tuple[dict, float, float]]:
        """Executes all operations in tensorflow graph for a given number of steps.

        Parameters
        ----------
        steps
            Number of graph evaluations.
        ops
            Graph operations to evaluate.
        inputs
            Inputs fed into the graph.
        outputs
            Variables in the graph to store the history from.
        sampling_steps
            Number of graph execution steps to combine into a single output step.
        sampling_ops
            Graph operations for output storage.
        out_dir
            Directory to write the session log into.
        profile
            Can be used to extract information about graph execution time and memory load. Can be:
            - `t` for returning the total graph execution time.
            - `m` for returning the peak memory consumption during graph excecution.
            - `mt` or `tm` for both

        Returns
        -------
        Union[Dict[str, tf.Variable], Tuple[dict, float, float]]
            If `profile` was requested, a tuple is returned that contains
                1) the results dictionary
                2) the simulation time if `t` was requested, else None.
                3) the peak memory consumption if `m` was requested, else None.
            If not, only the results dictionary is returned which contains a numpy array with results for each
            output key that was provided via `outputs`.

        """

        # initializations
        #################

        # initialize session log
        if out_dir:
            writer = tf.summary.FileWriter(out_dir, graph=self)

        # initialize profiler
        if profile is None:
            profile = ''
        if 'm' in profile:
            meta = tf.RunMetadata()
            time_and_memory = tf.profiler.ProfileOptionBuilder.time_and_memory()
        if 't' in profile:
            t0 = t.time()

        # graph execution
        #################

        # start session
        with tf.Session(graph=self, config=tf.ConfigProto(allow_soft_placement=True)) as sess:

            # initialize variables on graph
            sess.run(tf.global_variables_initializer())

            # simulate backend behavior for each time-step
            if 'm' in profile:

                # in profiler-mode
                for step in range(steps):
                    if step % sampling_steps == 0:
                        sess.run(sampling_ops, inputs[step], run_metadata=meta,
                                 options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE))
                    else:
                        sess.run(ops, inputs[step], run_metadata=meta,
                                 options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE))
            else:

                # in non-profiler mode
                for step in range(steps):
                    if step % sampling_steps == 0:
                        sess.run(sampling_ops, inputs[step])
                    else:
                        sess.run(ops, inputs[step])

            # output storage and clean-up
            #############################

            # store output variables in output dictionary
            for key, var in outputs.items():
                outputs[key] = var.eval(sess)

            # store profiling results
            if 't' in profile:
                sim_time = t.time() - t0
            else:
                sim_time = 0.
            if 'm' in profile:
                peak_memory = tf.profiler.profile(graph=self, run_meta=meta, cmd='op', options=time_and_memory
                                                  ).total_peak_bytes / 1e6
            else:
                peak_memory = 0.

            # close session stuff
            if out_dir:
                writer.close()
            sess.close()

        if profile:
            return outputs, sim_time, peak_memory
        return outputs

    def add_var(self,
                vtype: str,
                name: str,
                value: Optional[Any] = None,
                shape: Optional[Union[tuple, list, tf.TensorShape]] = None,
                dtype: Optional[Union[str, tf.dtypes.DType]] = None,
                **kwargs
                ) -> Union[tf.Variable, tf.Tensor]:
        """Adds a variable to the backend.

        Parameters
        ----------
        vtype
            Variable type. Can be
                - `state_var` for variables that can change over time.
                - `constant` for non-changing variables.
                - `placeholder` for variables with a value unknown during initialization.
        name
            Name of the variable.
        value
            Value of the variable. Not needed for placeholders.
        shape
            Shape of the variable.
        dtype
            Datatype of the variable.
        kwargs
            Additional keyword arguments passed to the tensorflow functions.

        Returns
        -------
        Union[tf.Variable, tf.Tensor]
            Handle for the tensorflow variable.

        """

        # processs input arguments
        ##########################

        # extract variable scope
        scope, reuse = self._get_scope(kwargs.pop('scope', None))

        # extract graph dependencies
        dependencies = kwargs.pop('dependencies', None)

        # set the default_device
        device = kwargs.pop('device', self.default_device)

        # check whether necessary arguments were provided
        if all([arg is None for arg in [shape, value, dtype]]):
            raise ValueError('Either `value` or `shape` and `dtype` need to be provided')

        # set shape, data-type and value
        if shape is None:
            shape = value.shape
        if dtype is None:
            dtype = value.dtype
        if value is None:
            value = np.zeros(shape, dtype=dtype.as_numpy_dtype)

        # create variable
        #################

        with tf.device(device):
            with self.as_default():
                with scope as sc:
                    with tf.control_dependencies(dependencies):

                        if reuse:

                            # create variable under existing scope
                            with tf.name_scope(sc.original_name_scope):
                                return self._create_var(vtype, name, value, shape, dtype, **kwargs)

                        else:

                            # create variable under new scope
                            return self._create_var(vtype, name, value, shape, dtype, **kwargs)

    def add_op(self,
               op: str,
               *args,
               **kwargs
               ) -> tf.Operation:
        """Add operation to the backend.

        Parameters
        ----------
        op
            Key of the operation. Needs to be a key of `TensorflowGraph.ops`
        args
            Positional arguments to be passed to the operation.
        kwargs
            Keyword arguments to be passed to the operation.

        Returns
        -------
        tf.Operation
            Handle for the tensorflow operation.

        """

        # process input arguments
        #########################

        # extract scope
        scope, reuse = self._get_scope(kwargs.pop('scope', None))

        # extract graph dependencies
        dependencies = kwargs.pop('dependencies', None)

        # set the default_device
        device = kwargs.pop('device', self.default_device)

        # extract additional infos
        assign_to_var = kwargs.pop('assign_to_var', False)

        # create operation
        ##################

        with tf.device(device):
            with self.as_default():
                with scope as sc:
                    with tf.control_dependencies(dependencies):

                        if reuse:

                            # create operation under existing scope
                            with tf.name_scope(sc.original_name_scope):
                                return self._create_op(op, assign_to_var, *args, **kwargs)

                        else:

                            # create operation under new scope
                            return self._create_op(op, assign_to_var, *args, **kwargs)

    def add_layer(self,
                  ops: List[tf.Operation],
                  *args,
                  **kwargs
                  ) -> List[tf.Operation]:
        """Adds a layer of operations to the backend using `tensorflow.tuple`.

        Parameters
        ----------
        ops
            All tensorflow operations that should be part of this layer.
        args
            Additional positional arguments to be passed to `tensorflow.tuple`.
        kwargs
            Additional keyword arguments to be passed to `tensorflow.tuple`.

        Returns
        -------
        List[tf.Operation]
            List of tensorflow operations with dependencies added (layer operations will all be evaluated before
            any layer operation can be used in successive layers.)

        """

        # process input arguments
        #########################

        dependencies = kwargs.pop('dependencies', None)
        scope, reuse = self._get_scope(kwargs.pop('scope', None))
        device = kwargs.pop('device', self.default_device)

        # create layer
        ##############

        with tf.device(device):
            with self.as_default():
                with scope as sc:
                    with tf.control_dependencies(dependencies):
                        if reuse:
                            with tf.name_scope(sc.original_name_scope):
                                return tf.tuple(ops, *args, **kwargs)
                        else:
                            return tf.tuple(ops, *args, **kwargs)

    def clear(self):
        """Resets all tensorflow operations on the graph.
        """
        tf.reset_default_graph()

    def _create_op(self, op: str, assign_to_var: bool, *args, **kwargs) -> Union[tf.Operation, tf.Variable]:
        """

        Parameters
        ----------
        op
            Key of the operation.
        assign_to_var
            If true, new variable will be created and result of op will be assigned to it.
        args
            Positional arguments to be passed to the tensorflow function.
        kwargs
            Keyword argumetns to be passed to the tensorflow function.

        Returns
        -------
        Union[tf.Operation, tf.Variable]
            Result of the tensorflow operation.

        """

        try:
            tf_op = self.ops[op](*args, **kwargs)
        except TypeError:
            try:
                tf_op = self.ops[op](list(args), **kwargs)
            except TypeError:
                # TODO: replace this with more generic way of handling args and kwargs
                args = list(args)
                arg_tmp = args.pop(-1)
                tf_op = self.ops[op](args, arg_tmp, **kwargs)

        # either assign result to new variable and return that variable or return result
        if assign_to_var:
            var = self.add_var(vtype='state_var', name=tf_op.name.split('/')[-1] + ['_tmp'],
                               value=np.zeros(tf_op.shape, dtype=tf_op.dtype.as_numpy_dtype))
            return var.assign(tf_op)
        else:
            return tf_op

    @staticmethod
    def _create_var(vtype: str,
                    name: str,
                    value: Optional[Any] = None,
                    shape: Optional[Union[tuple, list, tf.TensorShape]] = None,
                    dtype: Optional[Union[str, tf.dtypes.DType]] = None,
                    **kwargs
                    ) -> Union[tf.Variable, tf.Tensor]:
        """Instantiates new variable in backend.

        Parameters
        ----------
        vtype
            Variable type. Can be
                - `state_var` for variables that can change over time.
                - `constant` for non-changing variables.
                - `placeholder` for variables with a value unknown during initialization.
        name
            Name of the variable.
        value
            Value of the variable. Not needed for placeholders.
        shape
            Shape of the variable.
        dtype
            Datatype of the variable.
        kwargs
            Additional keyword arguments passed to the tensorflow functions.

        Returns
        -------
        Union[tf.Variable, tf.Tensor]
            Handle to the tensorflow variable.

        """

        if vtype == 'state_var':
            return tf.get_variable(name, shape, dtype, initializer=tf.constant_initializer(value), **kwargs)
        elif vtype == 'constant':
            return tf.constant(value, dtype, shape, name, **kwargs)
        elif vtype == 'placeholder':
            return tf.placeholder(dtype, shape, name, **kwargs)
        else:
            raise ValueError(f'`Type` is {vtype} but needs to be set to `state_var`, `constant` or '
                             f'`placeholder`.')

    def _get_scope(self, scope: str):
        """

        Parameters
        ----------
        scope
            Name of the name scope.

        Returns
        -------
        str
            Existing or new name scope.

        """

        with self.as_default():
            if scope is None:
                return tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE), True
            if scope not in self.existing_scopes.keys():
                self.existing_scopes[scope] = True
            return tf.variable_scope(scope, reuse=tf.AUTO_REUSE), self.existing_scopes[scope]


class KerasBackend(tf.keras.models.Model):
    """Expression parser that transforms expression into keras operations on a tensorflow graph.

    Parameters
    ----------
    expr_str
        See docstring of `ExpressionParser`.
    args
        See docstring of `ExpressionParser`. Each variable in args needs to be a dictionary with key-value pairs for:
            - `var`: contains the tensorflow variable.
            - `dependency`: Boolean. If True, the expression needs to wait for this variable to be calculated/updated
               before being evaluated.
    lhs
        See docstring of `ExpressionParser`.
    tf_graph
        Instance of `tensorflow.Graph`. Mathematical expression will be parsed into this graph.

    Attributes
    ----------
    tf_graph
        Instance of `tensorflow.Graph` containing a graph-representation of `expr_str`.
    ops
        Dictionary containing all mathematical operations available for this parser and their syntax. These include:
            - addition: `+`
            - subtraction: `-`
            - multiplication: `*`
            - division: `/`
            - modulo: `%`
            - exponentiation: `^`
            - matrix multiplication: `@`
            - matrix transposition: `.T`
            - matrix inversion: `.I`
            - logical greater than: `>`
            - logical less than: `<`
            - logical equal: `==`
            - logical unequal: `!=`
            - logical greater or equal: `>=`
            - logical smaller or equal: `<=`
    funcs
        Dicionary containing all additional functions available for this parser and their syntax. These include:
            - sinus: `sin()`.
            - cosinus: `cos()`.
            - tangens: `tan()`.
            - absolute: `abs()`.
            - maximum: `max()`
            - minimum: `min()`
            - index of maximum: `argmax()`
            - index of minimum: `argmin()`
            - round to next integer: `round()`. Tensorflow name: `tensorflow.to_int32()`.
            - round to certain decimal point: `roundto()`. Custom function using `tensorflow.round()`. Defined in
              `pyrates.parser.parser.py`.
            - sum over dimension(s): `sum()`. Tensorflow name: `reduce_sum()`.
            - Concatenate multiples of tensor over certain dimension: `tile()`.
            - Reshape tensor: `reshape()`.
            - Cut away dimensions of size 1: `squeeze()`.
            - Cast variable to data-type: `cast()`.
            - draw random variable from standard normal distribution: `randn()`.
              Tensorflow name: `tensorflow.random_normal`.
            - Create array filled with ones: `ones()`.
            - Create array filled with zeros: `zeros()`.
            - Apply softmax function to variable: `softmax()`. Tensorflow name: `tensorflow.nn.softmax()`.
            - Apply boolean mask to array: `boolean_mask()`.
            - Create new array with non-zero entries at certain indices: `scatter()`.
              Tensorflow name: `tensorflow.scatter_nd`
            - Add values to certain entries of tensor: 'scatter_add()'. Tensorflow name: `tensorflow.scatter_nd_add`.
            - Update values of certain tensor entries: `scatter_update()`.
              Tensorflow name: `tensorflow.scatter_nd_update`.
            - Apply tensor as index to other tensor: `array_idx()`. Tensorflow name: `tensorflow.gather_nd`.
            - Get variable from tensorflow graph or create new variable: `new_var()`:
              Tensorflow name: `tensorflow.get_variable`.
        For a detailed documentation of how to use these functions, see the tensorflow Python API.
    dtypes
        Dictionary containing all data-types available for this parser. These include:
            - float16, float32, float64
            - int16, int32, int64
            - uint16, uint32, uint64
            - complex64, complex128,
            - bool
        All of those data-types can be used inside a mathematical expression instead of using `cast()`
        (e.g. `int32(3.631)`.
    For all other attributes, see docstring of `ExpressionParser`.

    Methods
    -------
    See docstrings of `ExpressionParser` methods.

    Examples
    --------

    References
    ----------

    """

    def __init__(self, expr_str: str, args: dict, backend: tf.keras.layers.Layer, lhs: bool = False) -> None:
        """Instantiates keras expression parser.
        """

        # call super init
        #################

        super().__init__(expr_str=expr_str, args=args, backend=backend, lhs=lhs)

        # define operations and functions
        #################################

        # base math operations
        ops = {"+": tf.keras.layers.Lambda(lambda x: x[0] + x[1]),
               "-": tf.keras.layers.Lambda(lambda x: x[0] - x[1]),
               "*": tf.keras.layers.Lambda(lambda x: x[0] * x[1]),
               "/": tf.keras.layers.Lambda(lambda x: x[0] / x[1]),
               "%": tf.keras.layers.Lambda(lambda x: x[0] % x[1]),
               "^": tf.keras.layers.Lambda(lambda x: tf.keras.backend.pow(x[0], x[1])),
               "@": tf.keras.layers.Lambda(lambda x: tf.keras.backend.dot(x[0], x[1])),
               ".T": tf.keras.layers.Lambda(lambda x: tf.keras.backend.transpose(x)),
               ".I": tf.keras.layers.Lambda(lambda x: tf.matrix_inverse(x)),
               ">": tf.keras.layers.Lambda(lambda x: tf.keras.backend.greater(x[0], x[1])),
               "<": tf.keras.layers.Lambda(lambda x: tf.keras.backend.less(x[0], x[1])),
               "==": tf.keras.layers.Lambda(lambda x: tf.keras.backend.equal(x[0], x[1])),
               "!=": tf.keras.layers.Lambda(lambda x: tf.keras.backend.not_equal(x[0], x[1])),
               ">=": tf.keras.layers.Lambda(lambda x: tf.keras.backend.greater_equal(x[0], x[1])),
               "<=": tf.keras.layers.Lambda(lambda x: tf.keras.backend.less_equal(x[0], x[1])),
               "=": tf.keras.backend.update
               }
        self.ops.update(ops)

        # additional functions
        funcs = {"sin": tf.keras.layers.Lambda(lambda x: tf.keras.backend.sin(x)),
                 "cos": tf.keras.layers.Lambda(lambda x: tf.keras.backend.cos(x)),
                 "tanh": tf.keras.layers.Lambda(lambda x: tf.keras.backend.tanh(x)),
                 "abs": tf.keras.layers.Lambda(lambda x: tf.keras.backend.abs(x)),
                 "sqrt": tf.keras.layers.Lambda(lambda x: tf.keras.backend.sqrt(x)),
                 "sq": tf.keras.layers.Lambda(lambda x: tf.keras.backend.square(x)),
                 "exp": tf.keras.layers.Lambda(lambda x: tf.keras.backend.exp(x)),
                 "max": tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x)),
                 "min": tf.keras.layers.Lambda(lambda x: tf.keras.backend.min(x)),
                 "argmax": tf.keras.layers.Lambda(lambda x: tf.keras.backend.argmax(x)),
                 "argmin": tf.keras.layers.Lambda(lambda x: tf.keras.backend.argmin(x)),
                 "round": tf.keras.layers.Lambda(lambda x: tf.keras.backend.round(x)),
                 "roundto": tf.keras.layers.Lambda(lambda x: tf.keras.backend.round(x[0] * 10**x[1]) / 10**x[1]),
                 "sum": tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x[0], *x[1])
                                               if type(x) is list else tf.keras.backend.sum(x)),
                 "concat": tf.keras.layers.Lambda(lambda x: tf.keras.backend.concatenate(x[0], *x[1])
                                                  if type(x[0]) is list else tf.keras.backend.concatenate(x)),
                 "reshape": tf.keras.layers.Lambda(lambda x: tf.keras.backend.reshape(x[0], x[1])
                                                   if type(x) is list else tf.keras.backend.reshape(x)),
                 "shape": tf.keras.backend.shape,
                 "dtype": tf.keras.backend.dtype,
                 'squeeze': tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(x[0], x[1])
                                                   if type(x) is list else tf.keras.backend.squeeze(x[0], -1)),
                 "cast": tf.keras.layers.Lambda(lambda x: tf.keras.backend.cast(x[0], x[1])),
                 "randn": tf.keras.layers.Lambda(lambda x: tf.keras.backend.random_normal(x[0], *x[1])
                                                 if "Tensor" in str(type(x[0]))
                                                 else tf.keras.backend.random_normal(x)),
                 "ones": tf.keras.layers.Lambda(lambda x: tf.keras.backend.ones(x[0], x[1])
                                                if "Tensor" in str(type(x[0]))
                                                else tf.keras.backend.ones(x)),
                 "zeros": tf.keras.layers.Lambda(lambda x: tf.keras.backend.zeros(x[0], x[1])
                                                 if "Tensor" in str(type(x[0]))
                                                 else tf.keras.backend.zeros(x)),
                 "softmax": tf.keras.layers.Lambda(lambda x: tf.keras.activations.softmax(x[0], *x[1])
                                                   if type(x[0]) is list else tf.keras.activations.softmax(x)),
                 "gather": tf.keras.layers.Lambda(lambda x: tf.gather_nd(x[0], x[1])),
                 "mask": tf.keras.layers.Masking,
                 "lambda": tf.keras.layers.Lambda
                 }
        self.funcs.update(funcs)

        dtypes = {"float16": tf.float16,
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
        self.dtypes.update(dtypes)

    def add_variable(self, name, shape, dtype=None, initializer=None,
                   regularizer=None, trainable=True, constraint=None):
        pass

def no_op(*args, **kwargs):
    pass
