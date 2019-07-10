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
from typing import Optional, Dict, Callable, List, Any
import tensorflow as tf

# pyrates internal imports
from .funcs import *
from .numpy_backend import NumpyBackend, NumpyVar, PyRatesIndexOp, PyRatesAssignOp, PyRatesOp

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


class TensorflowVar(NumpyVar):
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
        else:
            return value

    @classmethod
    def _get_var(cls, value, name, dtype):
        return tf.Variable(value, name=name, dtype=dtype)


class TensorflowOp(PyRatesOp):

    def eval(self):
        globals()['tf'] = tf
        result = self._callable(*self.args)
        self._check_numerics(result, self.name)
        return result

    @staticmethod
    def _generate_func(func_str):
        func_dict = {}
        exec(func_str, globals(), func_dict)
        return func_dict


class TensorflowAssignOp(PyRatesAssignOp):

    def eval(self):
        result = self._callable(*self.args)
        self._check_numerics(result, self.name)
        return result

    @staticmethod
    def _generate_func(func_str):
        func_dict = {}
        exec(func_str, globals(), func_dict)
        return func_dict


class TensorflowIndexOp(PyRatesIndexOp):

    def eval(self):
        result = self._callable(*self.args)
        self._check_numerics(result, self.name)
        return result

    @staticmethod
    def _generate_func(func_str):
        func_dict = {}
        exec(func_str, globals(), func_dict)
        return func_dict


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
                         "/": {'name': "tensorflow_divide", 'call': "tf.divide"},
                         "%": {'name': "tensorflow_modulo", 'call': "tf.mod"},
                         "^": {'name': "tensorflow_power", 'call': "tf.pow"},
                         "**": {'name': "tensorflow_power", 'call': "tf.pow"},
                         "@": {'name': "tensorflow_dot", 'call': "tf.dot"},
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

    def broadcast(self, op1: Any, op2: Any, **kwargs) -> tuple:

        # match data types
        if not self._compare_dtypes(op1, op2):
            op1, op2 = self._match_dtypes(op1, op2)

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

    def _create_var(self, vtype, dtype, shape, value, name):
        return TensorflowVar(vtype=vtype, dtype=dtype, shape=shape, value=value, name=name, backend=self)

    def _create_op(self, op, *args, decorator=None):
        if op in ["=", "+=", "-=", "*=", "/="]:
            if len(args) > 2 and hasattr(args[2], 'shape'):
                if op == "=":
                    op = "update"
                elif op == "+=":
                    op = "update_add"
                else:
                    op = "update_sub"
                args = self._process_update_args_old(*args)
            return TensorflowAssignOp(self.ops[op]['call'], self.ops[op]['name'], *args)
        if op is "index":
            if hasattr(args[1], 'dtype') and 'bool' in str(args[1].dtype):
                return TensorflowOp(self.ops['mask']['call'], self.ops['mask']['name'], *args)
            if hasattr(args[1], 'shape') or type(args[1]) in (list, tuple):
                try:
                    return TensorflowOp(self.ops['gather']['call'], self.ops['gather']['name'], *args)
                except (ValueError, IndexError):
                    args = self._process_idx_args(*args)
                    return TensorflowOp(self.ops['gather_nd']['call'], self.ops['gather_nd']['name'], *args)
            return TensorflowIndexOp(self.ops[op]['call'], self.ops[op]['name'], *args)
        if op is "cast":
            args = list(args)
            for dtype in self.dtypes:
                if dtype in str(args[1]):
                    args[1] = self.dtypes[dtype]
                    break
            args = tuple(args)
        return TensorflowOp(self.ops[op]['call'], self.ops[op]['name'], *args)

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
                for acc in ["16", "32", "64", "128"]:
                    if acc in op1.dtype.name:
                        return op1, self.add_op('cast', op2, op1.dtype)
                    elif acc in op2.dtype.name:
                        return self.add_op('cast', op1, op2.dtype), op2
            return op1, self.add_op('cast', op2, op1.dtype)
        elif issubclass(type(op2), tf.Variable):
            return self.add_op('cast', op1, op2.dtype), op2
        elif hasattr(op1, 'numpy') or type(op1) is np.ndarray:
            return op1, self.add_op('cast', op2, op1.dtype)
        elif hasattr(op2, 'numpy') or type(op2) is np.ndarray:
            return self.add_op('cast', op1, op2.dtype), op2
        else:
            return op1, self.add_op('cast', op2, str(type(op1)).split('\'')[-2])

    def _run(self, layers, sampling_layer, steps, sampling_steps):
        if sampling_layer is None:
            run_without_sampling(layers, steps)
        else:
            sampling_func, sampling_args = sampling_layer
            run_with_sampling(layers, steps, sampling_func, sampling_args, sampling_steps)

    @staticmethod
    def _compare_shapes(op1: Any, op2: Any) -> bool:

        if hasattr(op1, 'shape') and hasattr(op2, 'shape'):
            return tuple(op1.shape) == tuple(op2.shape)
        if hasattr(op1, 'shape'):
            return len(tuple(op1.shape)) == 0
        if hasattr(op2, 'shape'):
            return len(tuple(op2.shape)) == 0
        else:
            return len(op1) == len(op2)

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
                return op1.dtype == op2.dtype
            else:
                return False


@tf.function
def run_without_sampling(layers, steps):
    for _ in tf.range(steps):
        for func, args in layers:
            func(*args)


@tf.function
def run_with_sampling(layers, steps, sampling_func, sampling_args, sampling_steps):
    for step in tf.range(steps):
        if tf.equal(step % sampling_steps, 0):
            sampling_func(*sampling_args)
        for func, args in layers:
            func(*args)
