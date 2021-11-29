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
from .base_backend import BaseBackend
from .computegraph import ComputeVar

# external imports
from typing import Optional, Dict, Callable, List, Any, Union
import tensorflow as tf

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


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

    def __init__(self,
                 ops: Optional[Dict[str, str]] = None,
                 dtypes: Optional[Dict[str, object]] = None,
                 imports: Optional[List[str]] = None,
                 **kwargs
                 ) -> None:
        """Instantiates tensorflow backend, i.e. a tensorflow graph.
        """

        # base math operations
        ops_tf = {
            "max": {'func': tf.reduce_max, 'str': "reduce_max"},
            "min": {'func': tf.reduce_min, 'str': "reduce_min"},
            "argmax": {'func': tf.argmax, 'str': "argmax"},
            "argmin": {'func': tf.argmin, 'str': "argmin"},
            "round": {'func': tf.round, 'str': "round"},
            "sum": {'func': tf.reduce_sum, 'str': "reduce_sum"},
            "mean": {'func': tf.reduce_mean, 'str': "reduce_mean"},
            "matmul": {'func': tf.matmul, 'str': "matmul"},
            "matvec": {'func': tf.linalg.matvec, 'str': "linalg.matvec"},
            'expand': {'func': tf.expand_dims, 'str': "expand_dims"},
            "roll": {'func': tf.roll, 'str': "roll"},
            "softmax": {'func': tf.math.softmax, 'str': "math.softmax"},
            "sigmoid": {'func': tf.math.sigmoid, 'str': "math.sigmoid"},
            "tanh": {'func': tf.tanh, 'str': "tanh"},
            "exp": {'func': tf.exp, 'str': "exp"},
        }
        if ops:
            ops_tf.update(ops)

        # base data types
        dtypes_tf = {
            "float16": tf.float16,
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
            dtypes_tf.update(dtypes)

        # base imports
        imports_tf = ["from tensorflow import *"]
        for op in ops_tf.values():
            *op_import, op_key = op['str'].split('.')
            if op_import:
                imports_tf.append(f"from tensorflow.{'.'.join(op_import)} import {op_key}")
                op['str'] = op_key

        if imports:
            for imp in imports:
                if imp not in imports_tf:
                    imports_tf.append(imp)

        super().__init__(ops_tf, dtypes_tf, imports_tf, **kwargs)

    def add_var_update(self, lhs: str, rhs: str, lhs_idx: Optional[str] = None, rhs_shape: Optional[tuple] = ()):

        if lhs_idx:
            lhs_idx = self._flatten_idx(lhs_idx, rhs_shape)
            idx = self.create_index_str(lhs_idx, separator=':', scatter=True)
            if not rhs_shape:
                rhs = f"[{rhs}]"
            else:
                try:
                    rhs = f"[{float(rhs)}]"
                except ValueError:
                    pass
            self.add_code_line(f"{lhs}.scatter_nd_update([{idx}], {rhs})")
        else:
            self.add_code_line(f"{lhs} = {rhs}")

    def create_index_str(self, idx: Union[str, int, tuple], separator: str = ',', scatter: bool = False) -> str:
        if scatter and separator in idx:
            idx_start, idx_stop = idx.split(separator)
            return ",".join([f"[{i}]" for i in range(int(idx_start), int(idx_stop))])
        return super().create_index_str(idx=idx, separator=separator)

    def _solve_euler(self, func: Callable, args: tuple, T: float, dt: float, dts: float, y: tf.Variable):

        # preparations for fixed step-size integration
        zero = tf.constant(0, dtype=tf.int32)
        idx = tf.Variable(0, dtype=tf.int32)
        steps = tf.constant(int(np.round(T / dt)))
        store_steps = int(np.round(T / dts))
        store_step = tf.constant(int(np.round(dts / dt)))
        state_rec = tf.Variable(np.zeros((store_steps, y.shape[0])))

        # perform fixed step-size integration
        self._euler_integrate(func, args, steps, store_step, zero, state_rec, idx, y, dt)
        return state_rec.numpy()

    @staticmethod
    @tf.function
    def _euler_integrate(func, args, steps, store_step, zero, state_rec, idx, y, dt):

        for step in tf.range(steps):

            if tf.equal(tf.math.floormod(step, store_step), zero):
                state_rec.scatter_nd_update([[idx]], [y])
                idx.assign_add(1)

            rhs = func(step, y, *args)
            y.assign_add(dt*rhs)

    @staticmethod
    def get_var(v: ComputeVar):
        if v.vtype == 'constant':
            return tf.constant(name=v.name, value=v.value)
        else:
            return tf.Variable(name=v.name, initial_value=v.value)

    @staticmethod
    def _euler_integration(steps, store_step, state_vec, state_rec, dt, rhs_func, rhs, *args):
        idx = 0
        for step in range(steps):
            if step % store_step == 0:
                state_rec[idx, :] = state_vec.numpy()
                idx += 1
            state_vec.assign_add(dt * rhs_func(step, state_vec, rhs, *args))
        return state_rec

    @staticmethod
    def process_idx(v: ComputeVar, idx: Union[str, ComputeVar]):
        if idx == ':':
            idx = f'0:{rhs_shape[0] if rhs_shape else 1}'