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

# pyrates internal imports
from .base_funcs import *
from .base_backend import BaseBackend, BaseVar, CodeGen, sort_equations

# external imports
from typing import Optional, Dict, Callable, List, Any, Union
import tensorflow as tf

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


# Helper Functions
##################

#################################
# classes for backend variables #
#################################


class TensorflowVar(BaseVar, tf.Variable):
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
            return BaseVar.__subclasscheck__(subclass)


#######################################
# classes for backend functionalities #
#######################################


class TensorflowBackend(BaseBackend):
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

    create_backend_var = TensorflowVar

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
        self.ops.update({
                    "max": {'func': tf.reduce_max, 'str': "tf.reduce_max"},
                    "min": {'func': tf.reduce_min, 'str': "tf.reduce_min"},
                    "argmax": {'func': tf.argmax, 'str': "tf.argmax"},
                    "argmin": {'func': tf.argmin, 'str': "tf.argmin"},
                    "round": {'func': tf.round, 'str': "tf.round"},
                    "sum": {'func': tf.reduce_sum, 'str': "tf.reduce_sum"},
                    "mean": {'func': tf.reduce_mean, 'str': "tf.reduce_mean"},
                    "matmul": {'func': tf.matmul, 'str': "tf.matmul"},
                    "concat": {'func': tf.concat, 'str': "tf.concat"},
                    "reshape": {'func': tf.reshape, 'str': "tf.reshape"},
                    "shape": {'func': tf.shape, 'str': "tf.shape"},
                    'squeeze': {'func': tf.squeeze, 'str': "np.squeeze"},
                    'expand': {'func': tf.expand_dims, 'str': "tf.expand_dims"},
                    "roll": {'func': tf.roll, 'str': "tf.roll"},
                    "cast": {'func': tf.Variable, 'str': "tf.Variable"},
                    "ones": {'func': tf.ones, 'str': "tf.ones"},
                    "zeros": {'func': tf.zeros, 'str': "tf.zeros"},
                    "range": {'func': tf.range, 'str': "tf.range"},
                    "softmax": {'func': tf.math.softmax, 'str': "tf.math.softmax"},
                    "sigmoid": {'func': tf.math.sigmoid, 'str': "tf.math.sigmoid"},
                    "tanh": {'func': tf.tanh, 'str': "tf.tanh"},
                    "exp": {'func': tf.exp, 'str': "tf.exp"},
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

        self.type = 'tensorflow'

    def run(self, T: float, dt: float, dts: float = None, outputs: dict = None, solver: str = None,
            in_place: bool = True, func_name: str = None, file_name: str = None, compile_kwargs: dict = None, **kwargs
            ) -> dict:
        if not compile_kwargs:
            compile_kwargs = dict()
        # TODO: enable usage of tf.function decorator
        # TODO: make sure that naming schemes are not messed up by tensorflow
        #compile_kwargs['decorator'] = tf.function
        return super().run(T, dt, dts, outputs, solver, in_place, func_name, file_name, compile_kwargs, **kwargs)

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

    @staticmethod
    def get_var_shape(v: BaseVar):
        if not v.shape or not tuple(v.shape):
            v = np.reshape(v, (1,))
        return v.shape[0]

    def _generate_update_equations(self, code_gen: CodeGen, nodes: dict, rhs_var: str = None, indices: list = None,
                                   funcs: dict = None):

        # collect right-hand side expression and all input variables to these expressions
        func_args, expressions, var_names, defined_vars = [], [], [], []
        for node, update in nodes.items():

            # collect expression and variables of right-hand side of equation
            expr_args, expr = self.graph.node_to_expr(update, **funcs)
            func_args.extend(expr_args)
            expressions.append(self._expr_to_str(expr))

            # process left-hand side of equation
            var = self.get_var(node)
            if 'expr' in var:
                # process indexing of left-hand side variable
                idx_args, lhs = self.graph.node_to_expr(node, **funcs)
                func_args.extend(idx_args)
                lhs_var = self._expr_to_str(lhs)
            else:
                # process normal update of left-hand side variable
                lhs_var = var['symbol'].name
            var_names.append(lhs_var)

        # add the left-hand side assignments of the collected right-hand side expressions to the code generator
        if rhs_var:

            # differential equation (DE) update
            if indices:

                # DE updates stored in a state-vector
                for idx, expr in zip(indices, expressions):
                    code_gen.add_code_line(f"{rhs_var}.scatter_nd_update([{idx}], [{expr}])")

            else:

                if len(nodes) > 1:
                    raise ValueError('Received a request to update a variable via multiple right-hand side expressions.'
                                     )

                # DE update stored in a single variable
                code_gen.add_code_line(f"{rhs_var} = {expressions[0]}")

            # add rhs var to function arguments
            func_args = [rhs_var] + func_args

        else:

            # non-differential equation update

            var_names, expressions, undefined_vars, defined_vars = sort_equations(var_names, expressions)
            func_args.extend(undefined_vars)

            if indices:

                # non-DE update stored in a variable slice
                for idx, expr, target_var in zip(indices, expressions, var_names):
                    code_gen.add_code_line(f"{target_var}.scatter_nd_update([{idx}], [{expr}])")

            else:

                # non-DE update stored in a single variable
                for target_var, expr in zip(var_names, expressions):
                    code_gen.add_code_line(f"{target_var} = {expr}")

        code_gen.add_linebreak()

        return func_args, defined_vars, code_gen

    @staticmethod
    def _euler_integration(steps, store_step, state_vec, state_rec, dt, rhs_func, rhs, *args):
        idx = 0
        for step in range(steps):
            if step % store_step == 0:
                state_rec[idx, :] = state_vec.numpy()
                idx += 1
            state_vec.assign_add(dt * rhs_func(step, state_vec, rhs, *args))
        return state_rec
