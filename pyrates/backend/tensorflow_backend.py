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

"""Wraps tensorflow such that it's low-level functions can be used by PyRates to create and simulate a compute graph.
"""

# external imports
from typing import Optional, Dict, Callable, List, Any, Union
import tensorflow as tf

# pyrates internal imports
from .funcs import *
from .numpy_backend import NumpyBackend, NumpyVar, PyRatesIndexOp, PyRatesAssignOp, PyRatesOp, CodeGen

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


class TensorflowVar(NumpyVar, tf.Variable):
    """Class for creating variables via a tensorflow-based PyRates backend.
    """

    @staticmethod
    def _get_value(value, dtype, shape):
        if value is None:
            return tf.zeros(shape=shape, dtype=dtype)
        elif not hasattr(value, 'shape'):
            if type(value) is list:
                return tf.zeros(shape=shape, dtype=dtype) + tf.reshape(tf.constant(value, dtype=dtype), shape=shape)
            else:
                return tf.zeros(shape=shape, dtype=dtype) + value
        elif shape:
            return tf.cast(tf.reshape(value, shape=shape), dtype)
        else:
            value = tf.constant(value, dtype=tf.float32)
            return tf.cast(tf.zeros(shape=value.shape, dtype=value.dtype) + value, dtype=dtype)

    @classmethod
    def _get_var(cls, value, name, dtype):
        return tf.Variable(value, name=name, dtype=dtype)

    @staticmethod
    def squeeze(var, **kwargs):
        return tf.squeeze(var, **kwargs)

    @staticmethod
    def __subclasscheck__(subclass):
        if tf.Variable.__subclasscheck__(subclass):
            return True
        else:
            return NumpyVar.__subclasscheck__(subclass)


class TensorflowOp(PyRatesOp):

    var_class = TensorflowVar

    def _generate_func(self):
        """Generates a function from operator value and arguments"""
        func_dict = {}
        func = CodeGen()
        func.add_code_line(f"def {self.short_name}(")
        for arg in self._op_dict['arg_names']:
            func.add_code_line(f"{arg},")
        if len(self._op_dict['arg_names']) > 0:
            func.code[-1] = func.code[-1][:-1]
        func.add_code_line("):")
        func.add_linebreak()
        func.add_indent()
        func.add_code_line("import tensorflow as tf")
        func.add_linebreak()
        func.add_code_line(f"return {self._op_dict['value']}")
        exec(func.generate(), globals(), func_dict)
        return func_dict

    def numpy(self):
        """Evaluates the return values of the PyRates operation.
        """
        result = self._callable(*self.args).numpy()
        self._check_numerics(result, self.name)
        return result

    @staticmethod
    def _index(x, y):
        found, idx, n = False, 0, len(x)
        while not found and idx < n:
            elem = x[idx]
            if str(type(elem)) == str(type(y)):
                found = elem == y
            else:
                pass
            if found:
                break
            idx += 1
        return idx


class TensorflowAssignOp(PyRatesAssignOp):

    var_class = TensorflowVar

    def _generate_func(self):
        """Generates a function from operator value and arguments"""
        func_dict = {}
        func = CodeGen()
        func.add_code_line(f"def {self.short_name}(")
        for arg in self._op_dict['arg_names']:
            func.add_code_line(f"{arg},")
        if len(self._op_dict['arg_names']) > 0:
            func.code[-1] = func.code[-1][:-1]
        func.add_code_line("):")
        func.add_linebreak()
        func.add_indent()
        func.add_code_line("import tensorflow as tf")
        func.add_linebreak()
        func.add_code_line(f"return {self._op_dict['value']}")
        exec(func.generate(), globals(), func_dict)
        return func_dict

    def numpy(self):
        """Evaluates the return values of the PyRates operation.
        """
        result = self._callable(*self.args).numpy()
        self._check_numerics(result, self.name)
        return result

    @classmethod
    def _extract_var_idx(cls, op, args, results_args, results_arg_names):

        if "scatter" in op:

            # for tensorflow-like scatter indexing
            if hasattr(args[2], 'short_name'):
                key = args[2].short_name
                if hasattr(args[2], 'value') and not callable(args[2].value):
                    var_idx = f"{args[2].value},"
                else:
                    var_idx = f"{key},"
            else:
                key = "__no_name__"
                var_idx = f"{args[2]},"

            if type(args[2]) is str and len(args) > 3:
                results_args.append(args[3])
            else:
                results_args.append(args[2])
            results_arg_names.append(key)

            return var_idx, results_args, results_arg_names

        else:

            return super()._extract_var_idx(op, args, results_args, results_arg_names)


class TensorflowIndexOp(PyRatesIndexOp):

    var_class = TensorflowVar

    def _generate_func(self):
        """Generates a function from operator value and arguments"""
        func_dict = {}
        func = CodeGen()
        func.add_code_line(f"def {self.short_name}(")
        for arg in self._op_dict['arg_names']:
            func.add_code_line(f"{arg},")
        if len(self._op_dict['arg_names']) > 0:
            func.code[-1] = func.code[-1][:-1]
        func.add_code_line("):")
        func.add_linebreak()
        func.add_indent()
        func.add_code_line("import tensorflow as tf")
        func.add_linebreak()
        func.add_code_line(f"return {self._op_dict['value']}")
        exec(func.generate(), globals(), func_dict)
        return func_dict

    def numpy(self):
        """Evaluates the return values of the PyRates operation.
        """
        result = self._callable(*self.args).numpy()
        self._check_numerics(result, self.name)
        return result


class TensorflowBackend(NumpyBackend):
    """Wrapper to tensorflow. This class provides an interface to all tensorflow functionalities that may be accessed
    via pyrates. All tensorflow variables and operations will be stored in a layered compute graph that can be executed
    to evaluate the (dynamic) behavior of the graph.

    Parameters
    ----------
    ops
        Additional operations this backend instance can perform, defined as key-value pairs. The key can then be used in
        every equation parsed into this backend. The value is a dictionary again, with two keys:
            1) name - the name of the function/operation (used during code generation)
            2) call - the call signature of the function including import abbreviations (i.e. `tf.add` for tensorflow's
            add function)
    dtypes
        Additional data-types this backend instance can use, defined as key-value pairs.
    name
        Name of the backend instance. Used during code generation to create a recognizable file structure.
    float_default_type
        Default float precision. If no data-type is indicated for a particular variable, this will be used.
    imports
        Can be used to pass additional import statements that are needed for code generation of the custom functions
        provided via `ops`. Will be added to the top of each generated code file.

    """

    def __init__(self,
                 ops: Optional[Dict[str, Callable]] = None,
                 dtypes: Optional[Dict[str, object]] = None,
                 name: str = 'net_0',
                 float_default_type: str = 'float32',
                 imports: Optional[List[str]] = None,
                 ) -> None:
        """Instantiates tensorflow backend, i.e. a tensorflow graph.
        """

        if not imports:
            imports = ["import tensorflow as tf"]

        super().__init__(ops, dtypes, name, float_default_type, imports)

        # define operations and datatypes of the backend
        ################################################

        # base math operations
        self.ops.update({"+": {'name': "tensorflow_add", 'call': "tf.add"},
                         "-": {'name': "tensorflow_subtract", 'call': "tf.subtract"},
                         "*": {'name': "tensorflow_multiply", 'call': "tf.multiply"},
                         "/": {'name': "tensorflow_divide", 'call': "tf.math.divide_no_nan"},
                         "%": {'name': "tensorflow_modulo", 'call': "tf.mod"},
                         "^": {'name': "tensorflow_power", 'call': "tf.pow"},
                         "**": {'name': "tensorflow_power", 'call': "tf.pow"},
                         "@": {'name': "tensorflow_dot", 'call': "tf.matmul"},
                         ".T": {'name': "tensorflow_transpose", 'call': "tf.transpose"},
                         ".I": {'name': "tensorflow_invert", 'call': "tf.invert"},
                         ">": {'name': "tensorflow_greater", 'call': "tf.greater"},
                         "<": {'name': "tensorflow_less", 'call': "tf.less"},
                         "==": {'name': "tensorflow_equal", 'call': "tf.equal"},
                         "!=": {'name': "tensorflow_not_equal", 'call': "tf.not_equal"},
                         ">=": {'name': "tensorflow_greater_equal", 'call': "tf.greater_equal"},
                         "<=": {'name': "tensorflow_less_equal", 'call': "tf.less_equal"},
                         "=": {'name': "assign", 'call': "assign"},
                         "+=": {'name': "assign_add", 'call': "assign_add"},
                         "-=": {'name': "assign_subtract", 'call': "assign_sub"},
                         "update": {'name': "tensorflow_update", 'call': "scatter_nd_update"},
                         "update_add": {'name': "tensorflow_update_add", 'call': "scatter_nd_add"},
                         "update_sub": {'name': "tensorflow_update_sub", 'call': "scatter_nd_sub"},
                         "neg": {'name': "negative", 'call': "neg_one"},
                         "sin": {'name': "tensorflow_sin", 'call': "tf.sin"},
                         "cos": {'name': "tensorflow_cos", 'call': "tf.cos"},
                         "tan": {'name': "tensorflow_tan", 'call': "tf.tan"},
                         "atan": {'name': "tensorflow_atan", 'call': "tf.arctan"},
                         "abs": {'name': "tensorflow_abs", 'call': "tf.abs"},
                         "sqrt": {'name': "tensorflow_sqrt", 'call': "tf.sqrt"},
                         "sq": {'name': "tensorflow_square", 'call': "tf.square"},
                         "exp": {'name': "tensorflow_exp", 'call': "tf.exp"},
                         "max": {'name': "tensorflow_max", 'call': "tf.max"},
                         "min": {'name': "tensorflow_min", 'call': "tf.min"},
                         "argmax": {'name': "tensorflow_transpose", 'call': "tf.argmax"},
                         "argmin": {'name': "tensorflow_argmin", 'call': "tf.argmin"},
                         "round": {'name': "tensorflow_round", 'call': "tf.round"},
                         "sum": {'name': "tensorflow_sum", 'call': "tf.reduce_sum"},
                         "mean": {'name': "tensorflow_mean", 'call': "tf.reduce_mean"},
                         "concat": {'name': "tensorflow_concatenate", 'call': "tf.concat"},
                         "reshape": {'name': "tensorflow_reshape", 'call': "tf.reshape"},
                         "shape": {'name': "tensorflow_shape", 'call': "tf.shape"},
                         "dtype": {'name': "tensorflow_dtype", 'call': "tf.dtype"},
                         'squeeze': {'name': "tensorflow_squeeze", 'call': "tf.squeeze"},
                         'expand': {'name': 'tensorflow_expand', 'call': "tf.expand_dims"},
                         "roll": {'name': "tensorflow_roll", 'call': "tf.roll"},
                         "cast": {'name': "tensorflow_cast", 'call': "tf.cast"},
                         "randn": {'name': "tensorflow_randn", 'call': "tf.randn"},
                         "ones": {'name': "tensorflow_ones", 'call': "tf.ones"},
                         "zeros": {'name': "tensorflow_zeros", 'call': "tf.zeros"},
                         "range": {'name': "tensorflow_arange", 'call': "tf.arange"},
                         "softmax": {'name': "tensorflow_softmax", 'call': "tf.softmax"},
                         "sigmoid": {'name': "tensorflow_sigmoid", 'call': "tf.sigmoid"},
                         "tanh": {'name': "tensorflow_tanh", 'call': "tf.tanh"},
                         "index": {'name': "pyrates_index", 'call': "pyrates_index"},
                         "gather": {'name': "tensorflow_gather", 'call': "tf.gather"},
                         "gather_nd": {'name': "tensorflow_gather_nd", 'call': "tf.gather_nd"},
                         "mask": {'name': "tensorflow_mask", 'call': "tf.boolean_mask"},
                         "group": {'name': "tensorflow_group", 'call': "tf.group"},
                         "stack": {'name': "tensorflow_stack", 'call': "tf.stack"},
                         "no_op": {'name': "tensorflow_identity", 'call': "tf.identity"},
                         })

        # base data types
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

    def compile(self, build_dir: Optional[str] = None, decorator: Optional[Callable] = tf.function, **kwargs) -> tuple:
        """Compile the graph layers/operations. Creates python files containing the functions in each layer.

        Parameters
        ----------
        build_dir
            Directory in which to create the file structure for the simulation.
        decorator
        kwargs

        Returns
        -------
        tuple
            Contains tuples of layer run functions and their respective arguments.

        """
        return super().compile(build_dir=build_dir, decorator=decorator, **kwargs)

    def broadcast(self, op1: Any, op2: Any, **kwargs) -> tuple:

        # match data types
        if not self._compare_dtypes(op1, op2):
            op1, op2 = self._match_dtypes(op1, op2)

        # broadcast shapes
        return super().broadcast(op1, op2, **kwargs)

    def get_var(self, name):
        """Retrieve variable from graph.

        Parameters
        ----------
        name
            Identifier of the variable.

        Returns
        -------
        NumpyVar
            Variable from graph.

        """
        try:
            return self.vars[name]
        except KeyError as e:
            if ":" in name:
                idx = name.index(':')
                return self.vars[name[:idx]]
            for var in self.vars:
                if f"{name}:" in var:
                    return self.vars[var]
            else:
                raise e

    def stack_vars(self, vars, **kwargs):
        var_count = {}
        for var in vars:
            if hasattr(var, 'short_name'):
                if var.short_name in var_count:
                    var.short_name += f'_{var_count[var.short_name]}'
                else:
                    var_count[var.short_name] = 0
        return self.add_op('stack', vars)

    def add_input_layer(self, inputs: list, T: float, continuous=False) -> NumpyVar:
        if continuous:
            raise ValueError('Invalid input structure. The tensorflow backend can only be used with fixed step-size '
                             'solvers and thus only supports inputs with discrete time steps. Either change the '
                             'backend or set `continuous` to False.')
        for i in range(len(inputs)):
            inputs[i] = (np.asarray(inputs[i][0], dtype=inputs[i][1].dtype.as_numpy_dtype), inputs[i][1], inputs[i][2])
        return super().add_input_layer(inputs=inputs, T=T, continuous=False)

    def apply_idx(self, var: Any, idx: Any, update: Optional[Any] = None, update_type: str = None, *args) -> Any:
        """Applies index to a variable. IF update is passed, variable is updated at positions indicated by index.

        Parameters
        ----------
        var
            Variable to index/update
        idx
            Index to variable
        update
            Update to variable entries
        update_type
            Type of lhs update (e.g. `=` or `+=`)

        Returns
        -------
        Any
            Updated/indexed variable.

        """

        if type(idx) is str:

            idx_tmp = idx.split(',')
            n_indices = len(idx_tmp)
            if n_indices > 1 and args:

                # create boolean mask as index
                indices = []
                for i, idx in enumerate(idx_tmp):
                    if idx == ':':
                        indices.append(list(range(var.shape[i])))
                    elif len(args) > i and hasattr(args[i], 'numpy'):
                        indices.append(args[i].numpy())
                    else:
                        indices.append([int(idx)])

                idx_var = np.zeros(tuple(var.shape), dtype=np.int32)
                idx_var[tuple(indices)] = 1
                idx_var = self.add_var('constant', name=f"{var.name.split(':')[0]}_idx", value=idx_var != 0)
                return super().apply_idx(var, idx_var, update, update_type)

        # standard indexing
        return super().apply_idx(var, idx, update, update_type, *args)

    def _integrate(self, rhs_func, func_args, T, dt, dts, t, output_indices):

        sampling_steps = int(np.round(T / dts, decimals=0))

        # initialize results storage vectors
        results = []
        for idx in output_indices:
            var_dim = idx[1] - idx[0] if type(idx) is tuple else 1
            results.append(tf.Variable(np.zeros((sampling_steps, var_dim), dtype=self._float_def)))

        # solve via pyrates internal explicit euler algorithm
        sampling_idx = tf.Variable(0, dtype='int32')
        sampling_steps = tf.constant(int(np.round(dts / dt, decimals=0)))
        dt = tf.constant(dt)
        steps = tf.constant(int(np.round(T / dt, decimals=0)))
        results = self._run(rhs_func=rhs_func, func_args=func_args, t=t, dt=dt, steps=steps,
                            sampling_steps=sampling_steps, results=results, sampling_idx=sampling_idx,
                            output_indices=output_indices)

        results = np.asarray([r.numpy() for r in results])
        times = np.arange(0, T, dts)

        return times, results

    @tf.function
    def _run(self, rhs_func, func_args, t, dt, steps, sampling_steps, results, sampling_idx,
             output_indices):

        zero = tf.constant(0, dtype=tf.int32)
        state_vars = self.vars['y']

        for step in tf.range(steps):

            deltas = rhs_func(t, state_vars, func_args)
            t.assign_add(dt)
            state_vars.assign_add(dt*deltas)

            if tf.equal(tf.math.floormod(step, sampling_steps), zero):

                for r, idx in zip(results, output_indices):
                    r.scatter_nd_update([[sampling_idx, 0]], [state_vars[idx]])

                sampling_idx.assign_add(1)

        return results

    def _create_var(self, vtype, dtype, shape, value, name, squeeze=True):
        var, name = TensorflowVar(vtype=vtype, dtype=dtype, shape=shape, value=value, name=name, backend=self,
                                  squeeze=squeeze)
        if ':' in name:
            name = name.split(':')[0]
        return var, name

    def _create_op(self, op, name, *args):
        if op in ["=", "+=", "-=", "*=", "/="]:
            if len(args) > 2 and (hasattr(args[2], 'shape') or type(args[2]) is list):
                if op == "=":
                    op = "update"
                elif op == "+=":
                    op = "update_add"
                else:
                    op = "update_sub"
                if type(args[2]) is list:
                    args = list(args)
                    args[2] = tf.constant(args[2])
                    args = self._process_update_args(*tuple(args))
            return TensorflowAssignOp(self.ops[op]['call'], self.ops[op]['name'], name, *args)
        if op is "index":
            if hasattr(args[1], 'dtype') and 'bool' in str(args[1].dtype):
                return TensorflowOp(self.ops['mask']['call'], self.ops['mask']['name'], name, *args)
            if (hasattr(args[1], 'shape') and len(args[1].shape)) or type(args[1]) in (list, tuple):
                try:
                    return TensorflowOp(self.ops['gather']['call'], self.ops['gather']['name'], name, *args)
                except (ValueError, IndexError):
                    args = self._process_idx_args(*args)
                    return TensorflowOp(self.ops['gather_nd']['call'], self.ops['gather_nd']['name'], name, *args)
            return TensorflowIndexOp(self.ops[op]['call'], self.ops[op]['name'], name, *args)
        if op is "cast":
            args = list(args)
            for dtype in self.dtypes:
                if dtype in str(args[1]):
                    args[1] = f"tf.{dtype}"
                    break
            args = tuple(args)
        return TensorflowOp(self.ops[op]['call'], self.ops[op]['name'], name, *args)

    def _process_idx_args(self, var, idx):
        """Preprocesses the index to a variable.
        """

        shape = var.shape

        # match shape of index to shape
        ###############################

        if len(idx.shape) < 2:
            idx = self.add_op('reshape', idx, tuple(idx.shape) + (1,))

        shape_diff = len(shape) - idx.shape[1]
        if shape_diff < 0:
            raise ValueError(f'Invalid index shape. Operation has shape {shape}, but received an index of '
                             f'length {idx.shape[1]}')

        # manage shape of idx
        if shape_diff > 0:
            indices = []
            indices.append(idx)
            for i in range(shape_diff):
                indices.append(self.add_op('zeros', tuple(idx.shape), idx.dtype))
            idx = self.add_op('concat', indices, 1)

        return var, idx

    def _match_dtypes(self, op1: Any, op2: Any) -> tuple:
        """Match data types of two operators/variables.
        """

        if issubclass(type(op1), tf.Variable):

            if issubclass(type(op2), tf.Variable):

                # cast both variables to lowest precision
                for acc in ["16", "32", "64", "128"]:
                    if acc in op1.dtype.name:
                        return op1, self.add_op('cast', op2, op1.dtype.name)
                    elif acc in op2.dtype.name:
                        return self.add_op('cast', op1, op2.dtype.name), op2

            if type(op2) is int or type(op2) is float:

                # transform op2 into constant tensor with dtype of op1
                return op1, self.add_op('cast', tf.constant(op2), op1.dtype.name)

            return op1, self.add_op('cast', op2, op1.dtype.name)

        elif issubclass(type(op2), tf.Variable):

            # transform op1 into constant tensor with dtype of op2
            if type(op1) is int or type(op2) is float:
                return self.add_op('cast', tf.constant(op1), op2.dtype.name), op2

            return self.add_op('cast', op1, op2.dtype.name), op2

        elif hasattr(op1, 'numpy') or type(op1) is np.ndarray:

            # cast op2 to numpy dtype of op1
            return op1, self.add_op('cast', op2, op1.dtype)

        elif hasattr(op2, 'numpy') or type(op2) is np.ndarray:

            # cast op1 to numpy dtype of op2
            return self.add_op('cast', op1, op2.dtype), op2

        else:

            # cast op2 to dtype of op1 referred from its type string
            return super()._match_dtypes(op1, op2)

    def _is_state_var(self, key):
        if ':' not in key and f"{key}:0" in self.state_vars:
            return f"{key}:0", True
        return super()._is_state_var(key)

    @staticmethod
    def _compare_shapes(op1: Any, op2: Any) -> bool:

        if hasattr(op1, 'shape') and hasattr(op2, 'shape'):
            if tuple(op1.shape) == tuple(op2.shape):
                return True
            elif len(op1.shape) > 1 and len(op2.shape) > 1 and tuple(op1.shape)[1] == tuple(op2.shape)[0]:
                return True
            elif len(op1.shape) == 0 and len(op2.shape) == 0:
                return True
            else:
                return False
        if hasattr(op1, 'shape'):
            return len(tuple(op1.shape)) == 0
        if hasattr(op2, 'shape'):
            return len(tuple(op2.shape)) == 0
        else:
            try:
                return len(op1) == len(op2)
            except TypeError:
                return True

    @staticmethod
    def _compare_dtypes(op1: Any, op2: Any) -> bool:
        """Checks whether the data types of op1 and op2 are compatible with each other.

        Parameters
        ----------
        op1
            First operator.
        op2
            Second operator.

        Returns
        -------
        bool
            If true, the data types of op1 and op2 are compatible.

        """

        try:
            x = op1 + op2
            return True
        except (TypeError, ValueError, Exception):
            if hasattr(op1, 'dtype') and hasattr(op2, 'dtype'):
                dtype1 = op1.dtype.as_numpy_dtype if hasattr(op1.dtype, 'as_numpy_dtype') else op1.dtype
                dtype2 = op2.dtype.as_numpy_dtype if hasattr(op2.dtype, 'as_numpy_dtype') else op2.dtype
                return dtype1 == dtype2
            else:
                return False
