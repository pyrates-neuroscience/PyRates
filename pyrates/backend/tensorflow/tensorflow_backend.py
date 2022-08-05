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

# pyrates internal _imports
from ..base import BaseBackend
from ..computegraph import ComputeVar
from .tensorflow_funcs import tf_funcs

# external _imports
from typing import Optional, Dict, Callable, List, Tuple, Union
import tensorflow as tf
import numpy as np

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
                 imports: Optional[List[str]] = None,
                 **kwargs
                 ) -> None:
        """Instantiates tensorflow backend, i.e. a tensorflow graph.
        """

        # add user-provided operations to function dict
        tf_ops = tf_funcs.copy()
        if ops:
            tf_ops.update(ops)

        # call parent method
        super().__init__(ops=tf_ops, imports=imports, **kwargs)

    def get_var(self, v: ComputeVar):
        v_tmp = super().get_var(v)
        if v.vtype == 'constant':
            return tf.constant(name=v.name, value=v_tmp, dtype=v_tmp.dtype)
        else:
            return tf.Variable(name=v.name, initial_value=v_tmp, dtype=v_tmp.dtype)

    def add_var_update(self, lhs: ComputeVar, rhs: str, lhs_idx: Optional[str] = None, rhs_shape: Optional[tuple] = ()):

        lhs = lhs.name
        if lhs_idx:
            idx, _ = self.create_index_str(lhs_idx, scatter=True, size=rhs_shape[0] if rhs_shape else 1, apply=True)
            if not rhs_shape:
                rhs = f"[{rhs}]"
            else:
                try:
                    rhs = f"[{float(rhs)}]"
                except ValueError:
                    pass
            self.add_code_line(f"{lhs}.scatter_nd_update({idx}, {rhs})")
        else:
            self.add_code_line(f"{lhs} = {rhs}")

    def create_index_str(self, idx: Union[str, int, tuple], separator: str = ',', apply: bool = True,
                         scatter: bool = False, size: Optional[int] = 1, max_length: int = 2) -> Tuple[str, dict]:

        # case I: multiple indices
        if scatter:

            # recognize indices that already match tensorflow requirements
            if type(idx) is list:
                return f"{idx}", dict()

            # create tensorflow-specific index
            if type(idx) is not tuple:
                idx = tuple(idx.split(separator)) if separator in idx else (idx,)
            new_idx, return_dict = self._get_tf_idx(idx, size=size, max_length=max_length)
            return f"{new_idx}" if apply else new_idx, return_dict

        # case II: indexing via tf.gather_nd
        if type(idx) is tuple and any([type(i) is ComputeVar and i.is_constant for i in idx]):

            # create tensorflow-specific index
            new_idx, return_dict = self._get_tf_idx(idx, size=size, max_length=max_length)
            return f".gather_nd({new_idx})" if apply else new_idx, return_dict

        # case III: default indexing
        return super().create_index_str(idx=idx, separator=separator, apply=apply, scatter=False)

    def finalize_idx_str(self, var: ComputeVar, idx: str):
        gather = ".gather_nd("
        if gather in idx:
            start = idx.find(gather) + len(gather)
            stop = idx.find(")")
            idx_str = idx[start:stop]
            import_line = "from tensorflow import gather_nd"
            if import_line not in self._imports:
                self._imports.append(import_line)
            return f"gather_nd({var.name}, {idx_str})"
        return f"{var.name}{idx}"

    def _process_idx(self, idx: Union[Tuple[int, int], int, str, ComputeVar], size: int = 1, scatter: bool = False
                     ) -> Union[str, list]:

        # case I: tensorflow-specific mode where lists of indexing integers are collected for each variable axis
        if scatter:
            if type(idx) is ComputeVar:
                return (idx.value + self._start_idx).tolist()
            if type(idx) is tuple:
                idx_range = np.arange(idx[0] + self._start_idx, idx[1]).tolist()
                return idx_range
            if type(idx) is int:
                return [idx + self._start_idx]
            if idx == ':':
                return self._process_idx((self._start_idx, size), scatter=scatter)
            if ':' in idx:
                idx0, idx1 = idx.split(':')
                idx0 = self._process_idx(idx0, scatter=scatter)[0]
                idx1 = self._process_idx(idx1, scatter=scatter)[0]
                return self._process_idx((idx0, idx1), scatter=scatter)
            try:
                return self._process_idx(int(idx), scatter=scatter)
            except (ValueError, TypeError):
                return idx

        # case II: multiple indices into a single dimension
        if type(idx) is ComputeVar and idx.is_constant:
            return (idx.value + self._start_idx).tolist()

        # case III: default mode
        return super()._process_idx(idx)

    def _get_tf_idx(self, idx: tuple, size: int, max_length) -> Tuple[list, dict]:

        # collect axis indices
        idx = list(idx)
        for i in range(len(idx)):
            idx[i] = self._process_idx(idx[i], size=size, scatter=True)

        # bring indices into form required for `tf.Variable.scatter_nd_update`
        length = max([len(i) for i in idx])
        new_idx = []
        for i in range(length):
            idx_tmp = []
            for j in range(len(idx)):
                try:
                    idx_tmp.append(idx[j][i])
                except IndexError:
                    idx_tmp.append(idx[j][0])
            new_idx.append(idx_tmp)

        if len(new_idx) > max_length:

            # for many indices, create new indexing constant
            idx = self.idx_dummy_var
            return_dict = {idx: new_idx}

        else:

            return_dict = dict()

        return new_idx, return_dict

    def _solve_euler(self, func: Callable, args: tuple, T: float, dt: float, dts: float, y: np.ndarray, t0: np.ndarray):

        # preparations for fixed step-size integration
        zero = tf.constant(0, dtype=tf.int32)
        steps = tf.constant(int(np.round(T / dt)))
        store_steps = int(np.round(T / dts))
        store_step = tf.constant(int(np.round(dts / dt)))
        state_rec = tf.Variable(np.zeros((store_steps, y.shape[0])), dtype=y.dtype)
        dt = tf.constant(dt)

        return self._solve_euler_tf(func, args, steps, store_step, zero, state_rec, y, t0, dt).numpy()

    @staticmethod
    def _solve_scipy(func: Callable, args: tuple, T: float, dt: float, y: tf.Variable, t0: tf.Variable,
                     times: np.ndarray, **kwargs):

        # solve ivp via scipy methods (solvers of various orders with adaptive step-size)
        from scipy.integrate import solve_ivp
        kwargs['t_eval'] = times
        func = tf.function(func)
        dtype = y.dtype

        # wrapper function to scipy solver
        def f(t: float, y: np.ndarray):
            rhs = func(tf.constant(t, dtype=dtype), tf.constant(y, dtype=dtype), *args)
            return rhs.numpy()

        # call scipy solver
        results = solve_ivp(fun=f, t_span=(t0.numpy(), T), y0=y.numpy(), first_step=dt, **kwargs)
        return results['y'].T

    @staticmethod
    @tf.function
    def _solve_euler_tf(func: Callable, args: tuple, steps: tf.constant, store_step: int, zero: tf.constant,
                        state_rec: tf.Variable, y: tf.Variable, idx: tf.Variable, dt: tf.constant):

        for step in tf.range(steps):

            if tf.equal(tf.math.floormod(step, store_step), zero):
                state_rec.scatter_nd_update([[idx]], [y])
                idx.assign_add(1)

            rhs = func(step, y, *args)
            y.assign_add(dt*rhs)
