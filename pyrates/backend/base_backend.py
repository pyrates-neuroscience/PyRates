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
from .parser import var_in_expression, extract_var
from .computegraph import ComputeGraph

# external imports
from typing import Optional, Dict, List, Union, Any, Callable
import os
import sys
from shutil import rmtree
import numpy as np


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


def sort_equations(lhs_vars: list, rhs_expressions: list) -> tuple:

    # TODO: get rid of this shit

    vars_new, expressions_new, defined_vars, all_vars = [], [], [], []
    lhs_vars_old, expressions_old = lhs_vars.copy(), rhs_expressions.copy()

    # first, collect all variables that do not appear in any other equations
    while lhs_vars_old:
        for var, expr in zip(lhs_vars, rhs_expressions):
            appears_in_rhs = False
            v_tmp, indexed = extract_var(var)
            for expr_tmp in rhs_expressions:
                if var_in_expression(v_tmp, expr_tmp):
                    appears_in_rhs = True
                    break
            if not appears_in_rhs:
                idx = lhs_vars_old.index(var)
                var = lhs_vars_old.pop(idx)
                vars_new.append(var)
                expressions_new.append(expr)
                expressions_old.pop(idx)
                if not indexed and v_tmp not in defined_vars:
                    defined_vars.append(v_tmp)

            if v_tmp not in all_vars:
                all_vars.append(v_tmp)

        if lhs_vars and lhs_vars == lhs_vars_old:
            break
        else:
            lhs_vars = lhs_vars_old
            rhs_expressions = expressions_old

    # next, collect all other variables
    vars_new.extend(lhs_vars_old[::-1])
    expressions_new.extend(expressions_old[::-1])

    return vars_new[::-1], expressions_new[::-1], [v for v in all_vars if v not in defined_vars], defined_vars


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
                 float_default_type: str = 'float32',
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
                    "matmul": {'func': np.matmul, 'str': "np.matmul"},
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

    @staticmethod
    def generate_func_head(func_name: str, code_gen: CodeGen, state_var: str = None, func_args: list = None,
                            imports: list = None):

        if not func_args:
            func_args = []
        state_vars = ['t', state_var]
        _, indices = np.unique(func_args, return_index=True)
        func_args = state_vars + [func_args[idx] for idx in np.sort(indices)]

        if imports:

            # add imports at beginning of file
            for imp in imports:
                code_gen.add_code_line(imp)
            code_gen.add_linebreak()

        # add function header
        code_gen.add_linebreak()
        code_gen.add_code_line(f"def {func_name}({','.join(func_args)}):")
        code_gen.add_indent()

        return func_args, code_gen

    @staticmethod
    def generate_func_tail(code_gen: CodeGen, rhs_var: str = None):

        code_gen.add_code_line(f"return {rhs_var}")
        code_gen.remove_indent()

        return code_gen

    #     # create build dir
    #     self._orig_dir = os.getcwd()
    #     self._build_dir = f"{build_dir}/{self.name}" if build_dir else self._orig_dir
    #     if build_dir:
    #         os.makedirs(build_dir, exist_ok=True)
    #         try:
    #             os.mkdir(self._build_dir)
    #         except FileExistsError:
    #             rmtree(self._build_dir)
    #             os.mkdir(self._build_dir)
    #         sys.path.append(self._build_dir)
    #     else:
    #         sys.path.append(self._orig_dir)
    #
    # def run(self, T: float, dt: float, dts: float = None, outputs: dict = None, solver: str = 'euler',
    #         in_place: bool = True, func_name: str = None, file_name: str = None, compile_kwargs: dict = None, **kwargs
    #         ) -> dict:
    #
    #     # preparations
    #     ##############
    #
    #     if not compile_kwargs:
    #         compile_kwargs = dict()
    #     if not func_name and solver:
    #         func_name = self._func_name
    #     if not file_name and solver:
    #         file_name = self._file_name
    #     if not dts:
    #         dts = dt
    #     self._func_name = func_name
    #     self._file_name = file_name
    #
    #     # network specs
    #     run_info = self.compile(in_place=in_place, func_name=func_name, file_name=file_name, **compile_kwargs)
    #     state_vec = self.get_var(run_info['state_vec'])['value']
    #     rhs = self.ops['cast']['func'](self.ops['zeros']['func'](shape=tuple(state_vec.shape)))
    #
    #     # simulation specs
    #     times = np.arange(0, T, dts) if dts else np.arange(0, T, dt)
    #
    #     # perform simulation
    #     ####################
    #
    #     rhs_func = run_info['func']
    #     func_args = run_info['func_args'][3:]
    #     args = tuple([self.get_var(arg)['value'] for arg in func_args])
    #
    #     # perform integration via scipy solver (mostly Runge-Kutta methods)
    #     state_rec = self._solve_ivp(solver, T, state_vec, rhs, dt, times, dts, rhs_func, *args, **kwargs)
    #
    #     # reduce state recordings to requested state variables
    #     final_results = {}
    #     for key, var in outputs.items():
    #         idx = run_info['old_state_vars'].index(var)
    #         idx = run_info['vec_indices'][idx]
    #         if type(idx) is tuple and idx[1] - idx[0] == 1:
    #             idx = (idx[0],)
    #         elif type(idx) is int:
    #             idx = (idx,)
    #         final_results[key] = state_rec[:, idx] if len(idx) == 1 else state_rec[:, idx[0]:idx[1]]
    #     final_results['time'] = times
    #
    #     return final_results
    #
    # def clear(self) -> None:
    #     """Deletes build directory and removes all compute graph nodes
    #     """
    #
    #     # delete compute graph nodes
    #     nodes = [n for n in self.graph.nodes]
    #     for n in nodes:
    #         self.graph.remove_subgraph(n)
    #     self._var_map.clear()
    #
    #     # remove files and directories that have been created during simulation process
    #     if self._build_dir != self._orig_dir:
    #         rmtree(f"{self._orig_dir}/{self._build_dir}")
    #     else:
    #         try:
    #             os.remove(f"{self._orig_dir}/{self._file_name}{self._file_ending}")
    #         except FileNotFoundError:
    #             pass
    #
    #     if self._file_name in sys.modules:
    #         del sys.modules[self._file_name]
    #
    #     # clear code generator
    #     self._code_gen.clear()
    #
    # def _process_var_update(self, var: str, update: str) -> tuple:
    #
    #     # extract nodes
    #     var_info = self.get_var(var)
    #     update_info = self.get_var(update)
    #
    #     # extract common shape
    #     lhs = var_info['value']
    #     if not lhs.shape:
    #         lhs = np.reshape(lhs, (1,))
    #     rhs = self.graph.eval_node(update)
    #     if not rhs.shape:
    #         rhs = np.reshape(rhs, (1,))
    #     s1, s2 = self.get_var_shape(lhs), self.get_var_shape(rhs)
    #     if s1 == s2:
    #         return lhs, rhs, s1
    #     raise ValueError(
    #         f"Shapes of state variable {var} and its right-hand side update {update_info['expr']} do not"
    #         " match.")
    #
    # def _solve_ivp(self, solver: str, T: float, state_vec: np.ndarray, rhs: np.ndarray, dt: float,
    #                eval_times: np.ndarray, dts: float, rhs_func: Callable, *args, **kwargs) -> np.ndarray:
    #
    #     if solver == 'euler':
    #
    #         # solve ivp via forward euler method (fixed integration step-size)
    #         return self._solve_euler(T, state_vec, rhs, dt, eval_times, dts, rhs_func, *args, **kwargs)
    #
    #     else:
    #
    #         # solve ivp via scipy methods (solvers of various orders with adaptive step-size)
    #         from scipy.integrate import solve_ivp
    #         t = 0.0
    #         kwargs['t_eval'] = eval_times
    #
    #         # call scipy solver
    #         results = solve_ivp(fun=rhs_func, t_span=(t, T), y0=state_vec, first_step=dt, args=args, **kwargs)
    #
    #         return results['y'].T
    #
    # @staticmethod
    # def get_var_shape(v):
    #     if not v.shape:
    #         v = np.reshape(v, (1,))
    #     return v.shape[0]
    #
    # @staticmethod
    # def _solve_euler(T: float, state_vec: np.ndarray, rhs: np.ndarray, dt: float, eval_times: np.ndarray, dts: float,
    #                  rhs_func: Callable, *args, **kwargs):
    #
    #     # preparations for fixed step-size integration
    #     idx = 0
    #     steps = int(np.round(T / dt))
    #     store_steps = int(np.round(T / dts))
    #     store_step = int(np.round(dts / dt))
    #     state_rec = np.zeros((store_steps, state_vec.shape[0]))
    #
    #     # solve ivp for forward Euler method
    #     for step in range(steps):
    #         if step % store_step == 0:
    #             state_rec[idx, :] = state_vec
    #             idx += 1
    #         rhs_func(step, state_vec, rhs, *args)
    #         state_vec += dt * rhs
    #
    #     return state_rec
