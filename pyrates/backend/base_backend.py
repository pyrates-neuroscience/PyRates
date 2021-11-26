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

A new backend needs to implement the following methods:
- __init__
- run
- add_var
- add_op
- add_layer

Currently supported backends:
- Numpy: BaseBackend.
- Tensorflow: TensorflowBackend.
- Fortran: FortranBackend (experimental).

"""

# pyrates internal imports
from .base_funcs import *

# external imports
from typing import Optional, Dict, List, Union, Any, Callable
import numpy as np
from numba import njit


# Helper Functions and Classes
##############################


class CodeGen:
    """Generates python code. Can add code lines, line-breaks, indents and remove indents.
    """

    def __init__(self):
        self.code = []
        self.lvl = 0

    def generate(self):
        """Generates a single code string from its history of code additions.
        """
        return '\n'.join(self.code)

    def add_code_line(self, code_str):
        """Add code line string to code.
        """
        code_str = code_str.split('\n')
        for code in code_str:
            self.code.append("\t" * self.lvl + code)

    def add_linebreak(self):
        """Add a line-break to the code.
        """
        self.code.append("")

    def add_indent(self):
        """Add an indent to the code.
        """
        self.lvl += 1

    def remove_indent(self):
        """Remove an indent to the code.
        """
        if self.lvl == 0:
            raise(SyntaxError("Error in generation of network function file: A net negative indentation was requested.")
                  )
        self.lvl -= 1

    def clear(self):
        """Deletes all code lines from the memory of the generator.
        """
        self.code.clear()


#######################################
# classes for backend functionalities #
#######################################


class BaseBackend(CodeGen):
    """Default backend class. Transforms all network equations into their numpy equivalents. Based on a Python code
    generator.
    """

    def __init__(self,
                 ops: Optional[Dict[str, str]] = None,
                 dtypes: Optional[Dict[str, object]] = None,
                 imports: Optional[List[str]] = None,
                 **kwargs
                 ) -> None:
        """Instantiates the standard, numpy-based backend, i.e. a compute graph with numpy operations.

        Parameters
        ----------
        ops
        dtypes
        float_default_type
        imports
        kwargs
        """

        # call to super method (initializes code generator)
        super().__init__()

        # define operations and datatypes of the backend
        ################################################

        # base math operations
        self.ops = {
                    "max": {'func': np.maximum, 'str': "np.maximum"},
                    "min": {'func': np.minimum, 'str': "np.minimum"},
                    "argmax": {'func': np.argmax, 'str': "np.argmax"},
                    "argmin": {'func': np.argmin, 'str': "np.argmin"},
                    "round": {'func': np.round, 'str': "np.round"},
                    "sum": {'func': np.sum, 'str': "np.sum"},
                    "mean": {'func': np.mean, 'str': "np.mean"},
                    "matmul": {'func': np.dot, 'str': "np.dot"},
                    "concat": {'func': np.concatenate, 'str': "np.concatenate"},
                    "reshape": {'func': np.reshape, 'str': "np.reshape"},
                    "append": {'func': np.append, 'str': "np.append"},
                    "shape": {'func': np.shape, 'str': "np.shape"},
                    "dtype": {'func': np.dtype, 'str': "np.dtype"},
                    'squeeze': {'func': np.squeeze, 'str': "np.squeeze"},
                    'expand': {'func': np.expand_dims, 'str': "np.expand_dims"},
                    "roll": {'func': np.roll, 'str': "np.roll"},
                    "cast": {'func': np.asarray, 'str': "np.asarray"},
                    "randn": {'func': np.random.randn, 'str': "np.randn"},
                    "ones": {'func': np.ones, 'str': "np.ones"},
                    "zeros": {'func': np.zeros, 'str': "np.zeros"},
                    "range": {'func': np.arange, 'str': "np.arange"},
                    "softmax": {'func': pr_softmax, 'str': "pr_softmax"},
                    "sigmoid": {'func': pr_sigmoid, 'str': "pr_sigmoid"},
                    "tanh": {'func': np.tanh, 'str': "np.tanh"},
                    "sinh": {'func': np.sinh, 'str': "np.sinh"},
                    "cosh": {'func': np.cosh, 'str': "np.cosh"},
                    "arctan": {'func': np.arctan, 'str': "np.arctan"},
                    "arcsin": {'func': np.arcsin, 'str': "np.arcsin"},
                    "arccos": {'func': np.arccos, 'str': "np.arccos"},
                    "sin": {'func': np.sin, 'str': "np.sin"},
                    "cos": {'func': np.cos, 'str': "np.cos"},
                    "tan": {'func': np.tan, 'str': "np.tan"},
                    "exp": {'func': np.exp, 'str': "exp"},
                    "no_op": {'func': pr_identity, 'str': "pr_identity"},
                    "interp": {'func': pr_interp, 'str': "pr_interp"},
                    "interpolate_1d": {'func': pr_interp_1d, 'str': "pr_interp_1d"},
                    "interpolate_nd": {'func': pr_interp_nd, 'str': "pr_interp_nd"},
                    "index": {'func': pr_base_index, 'str': "pr_base_index"},
                    "index_axis": {'func': pr_axis_index, 'str': "pr_axis_index"},
                    "index_2d": {'func': pr_2d_index, 'str': "pr_2d_index"},
                    "index_range": {'func': pr_range_index, 'str': "pr_range_index"},
                    }
        if ops:
            self.ops.update(ops)

        # base data-types
        self.dtypes = {"float16": np.float16,
                       "float32": np.float32,
                       "float64": np.float64,
                       "int16": np.int16,
                       "int32": np.int32,
                       "int64": np.int64,
                       "uint16": np.uint16,
                       "uint32": np.uint32,
                       "uint64": np.uint64,
                       "complex64": np.complex64,
                       "complex128": np.complex128,
                       "bool": np.bool
                       }
        if dtypes:
            self.dtypes.update(dtypes)

        # further attributes
        self._file_ending = ".py"
        self.imports = ["from numpy import *", "from pyrates.backend.base_funcs import *"]
        if imports:
            for imp in imports:
                if imp not in self.imports:
                    self.imports.append(imp)
        self._idx_left = "["
        self._idx_right = "]"
        self._start_idx = 0

    def expr_to_str(self, expr: Any, expr_str: str = None) -> str:

        if not expr_str:
            expr_str = str(expr)
            for arg in expr.args:
                expr_str = expr_str.replace(str(arg), self.expr_to_str(arg))

        while 'pr_base_index(' in expr_str:

            # replace `index` calls with brackets-based indexing
            start = expr_str.find('pr_base_index(')
            end = expr_str[start:].find(')') + 1
            idx = expr.args[1]
            expr_str = expr_str.replace(expr_str[start:start+end], f"{expr.args[0]}{self.create_index_str(idx)}")

        while 'pr_2d_index(' in expr_str:

            # replace `index` calls with brackets-based indexing
            start = expr_str.find('pr_2d_index(')
            end = expr_str[start:].find(')') + 1
            idx = (expr.args[1], expr.args[2])
            expr_str = expr_str.replace(expr_str[start:start + end], f"{expr.args[0]}{self.create_index_str(idx)}")

        while 'pr_range_index(' in expr_str:

            # replace `index` calls with brackets-based indexing
            start = expr_str.find('pr_range_index(')
            end = expr_str[start:].find(')') + 1
            idx = (expr.args[1], f"{expr.args[2]}")
            expr_str = expr_str.replace(expr_str[start:start + end],
                                        f"{expr.args[0]}{self.create_index_str(idx, separator=':')}")

        while 'pr_axis_index(' in expr_str:

            # replace `index` calls with brackets-based indexing
            start = expr_str.find('pr_axis_index(')
            end = expr_str[start:].find(')') + 1
            if len(expr.args) == 1:
                expr_str = expr_str.replace(expr_str[start:start + end], f"{expr.args[0]}{self.create_index_str(':')}")
            else:
                idx = tuple([':' for _ in range(expr.args[2])] + [expr.args[1]])
                expr_str = expr_str.replace(expr_str[start:start + end], f"{expr.args[0]}{self.create_index_str(idx)}")

        while 'pr_identity(' in expr_str:

            # replace `no_op` calls with first argument to the function call
            start = expr_str.find('pr_identity(')
            end = expr_str[start:].find(')') + 1
            expr_str = expr_str.replace(expr_str[start:start+end], f"{expr.args[0]}")

        return expr_str

    def create_index_str(self, idx: Union[str, int, tuple], separator: str = ',') -> str:

        # case: multiple indices
        if type(idx) is tuple:
            for i in range(len(idx)):
                try:
                    idx[i] += self._start_idx
                except TypeError:
                    pass
            return f"{self._idx_left}{separator.join(idx)}{self._idx_right}"

        # case: single index
        try:
            idx += self._start_idx
        except TypeError:
            pass
        return f"{self._idx_left}{idx}{self._idx_right}"

    def get_fname(self, f: str) -> tuple:

        f_split = f.split('.')
        if len(f_split) > 2:
            raise ValueError(f'File name {f} has wrong format. Only one `.` can be used to separate file name from '
                             f'file ending.')
        if len(f_split) == 2:
            *path, file = f_split[0].split('/')
            return '/'.join(path), file, f_split[1]
        else:
            *path, file = f.split('/')
            return '/'.join(path), file, self._file_ending

    def generate_func_head(self, func_name: str, state_var: str = None, func_args: list = None):

        imports = self.imports

        if not func_args:
            func_args = []
        state_vars = ['t', state_var]
        _, indices = np.unique(func_args, return_index=True)
        func_args = state_vars + [func_args[idx] for idx in np.sort(indices)]

        if imports:

            # add imports at beginning of file
            for imp in imports:
                self.add_code_line(imp)
            self.add_linebreak()

        # add function header
        self.add_linebreak()
        self.add_code_line(f"def {func_name}({','.join(func_args)}):")
        self.add_indent()

        return func_args

    def generate_func_tail(self, rhs_var: str = None):

        self.add_code_line(f"return {rhs_var}")
        self.remove_indent()

    def run(self, func: Callable, func_args: tuple, T: float, dt: float, dts: float, y0: np.ndarray, outputs: dict,
            solver: str, **kwargs) -> dict:

        # simulation specs
        times = np.arange(0, T, dts) if dts else np.arange(0, T, dt)

        # perform simulation
        ####################

        # perform integration via scipy solver (mostly Runge-Kutta methods)
        if solver == 'euler':

            # solve ivp via forward euler method (fixed integration step-size)
            results = self._solve_euler(func, func_args, T, dt, dts, y0)

        else:

            # solve ivp via scipy methods (solvers of various orders with adaptive step-size)
            from scipy.integrate import solve_ivp
            t = 0.0
            kwargs['t_eval'] = times

            # call scipy solver
            results = solve_ivp(fun=func, t_span=(t, T), y0=y0, first_step=dt, args=func_args, **kwargs)
            results = results['y'].T

        # reduce state recordings to requested state variables
        for key, idx in outputs.items():
            if type(idx) is tuple and idx[1] - idx[0] == 1:
                idx = (idx[0],)
            elif type(idx) is int:
                idx = (idx,)
            outputs[key] = results[:, idx] if len(idx) == 1 else results[:, idx[0]:idx[1]]
        outputs['time'] = times

        return outputs

    @staticmethod
    def _solve_euler(func: Callable, args: tuple, T: float, dt: float, dts: float, y: np.ndarray):

        # preparations for fixed step-size integration
        idx = 0
        steps = int(np.round(T / dt))
        store_steps = int(np.round(T / dts))
        store_step = int(np.round(dts / dt))
        state_rec = np.zeros((store_steps, y.shape[0]))

        # solve ivp for forward Euler method
        for step in range(steps):
            if step % store_step == 0:
                state_rec[idx, :] = y
                idx += 1
            rhs = func(step, y, *args)
            y += dt * rhs

        return state_rec
