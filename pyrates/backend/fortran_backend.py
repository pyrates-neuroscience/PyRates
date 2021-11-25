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
import gc

from .base_backend import BaseBackend, CodeGen
from .fortran_funcs import get_fortran_func, fortran_identifiers
from .base_funcs import *

# external imports
from typing import Optional, Dict, Union, List, Any, Callable
import os
from numpy import f2py
import numpy as np

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


# helper functions and classes
##############################


class FortranGen(CodeGen):

    n1 = 62
    n2 = 72
    linebreak_start = "     & "
    linebreak_end = "&"

    def add_code_line(self, code_str):
        """Add code line string to code.
        """
        for code in code_str.split('\n'):
            if self.linebreak_end not in code:
                code = code.replace('\t', '')
                code = '\t' * self.lvl + code
            if self.break_line(code):
                idx = self._find_first_op(code, start=len(self.linebreak_start),
                                          stop=self.n2 - len(self.linebreak_end))
                self.add_code_line(f'{code[0:idx]}{self.linebreak_end}')
                code = f"{self.linebreak_start}{code[idx:]}"
                self.add_code_line(code)
            else:
                self.code.append(code)

    def break_line(self, code: str):
        n = len(code)
        if n > self.n2:
            return True
        if n > self.n1:
            if self.linebreak_start in code:
                if self.linebreak_end in code[len(self.linebreak_start):]:
                    if n - len(self.linebreak_start) - len(self.linebreak_end) < self.n2:
                        return False
                    return True
                if n - len(self.linebreak_start) < self.n2:
                    return False
                return True
            if self.linebreak_end in code:
                if n - len(self.linebreak_end) < self.n2:
                    return False
                return True
            return False
        return False

    @staticmethod
    def _find_first_op(code, start, stop):
        if stop < len(code):
            code_tmp = code[start:stop]
            ops = ["+", "-", "*", "/", "**", "^", "%", "<", ">", "==", "!=", "<=", ">="]
            indices = [code_tmp.index(op) for op in ops if op in code_tmp]
            if indices and max(indices) > 0:
                return max(indices) + start
            idx = start
            for break_sign in [',', ')', ' ']:
                if break_sign in code_tmp:
                    idx_tmp = len(code_tmp) - code_tmp[::-1].index(break_sign)
                    if len(code_tmp)-idx_tmp < len(code_tmp)-idx:
                        idx = idx_tmp
            return idx + start
        return stop + start


# backend classes
#################


class FortranBackend(BaseBackend):

    def __init__(self,
                 ops: Optional[Dict[str, str]] = None,
                 dtypes: Optional[Dict[str, object]] = None,
                 name: str = 'net_0',
                 float_default_type: str = 'float64',
                 imports: Optional[List[str]] = None,
                 build_dir: Optional[str] = None,
                 **kwargs
                 ) -> None:
        """Instantiates fortran backend, which will set up a compute graph and translate it into Fortran90 code.
        """

        # define fortran operations
        ops_f = {
            "max": {'func': np.maximum, 'str': "max"},
            "min": {'func': np.minimum, 'str': "min"},
            "argmax": {'func': np.argmax, 'str': "maxloc"},
            "argmin": {'func': np.argmin, 'str': "minloc"},
            "round": {'func': np.round, 'str': "nint"},
            "sum": {'func': np.sum, 'str': "sum"},
            "mean": {'func': np.mean, 'str': "import:fmean"},
            "matmul": {'func': np.matmul, 'str': "matmul"},
            "reshape": {'func': np.reshape, 'str': "reshape"},
            "shape": {'func': np.shape, 'str': "shape"},
            "roll": {'func': np.roll, 'str': "cshift"},
            "softmax": {'func': pr_softmax, 'str': "import:fsoftmax"},
            "sigmoid": {'func': pr_sigmoid, 'str': "import:fsigmoid"},
            "tanh": {'func': np.tanh, 'str': "tanh"},
            "sinh": {'func': np.sinh, 'str': "sinh"},
            "cosh": {'func': np.cosh, 'str': "cosh"},
            "arctan": {'func': np.arctan, 'str': "atan"},
            "arcsin": {'func': np.arcsin, 'str': "asin"},
            "arccos": {'func': np.arccos, 'str': "acos"},
            "sin": {'func': np.sin, 'str': "sin"},
            "cos": {'func': np.cos, 'str': "cos"},
            "tan": {'func': np.tan, 'str': "tan"},
            "exp": {'func': np.exp, 'str': "exp"},
            "interp": {'func': pr_interp, 'str': "import:finterp"},
        }
        if not ops:
            ops = {}
        ops.update(ops_f)

        if not dtypes:
            dtypes = {}
        dtypes.update({"float16": np.float64,
                       "float32": np.float64,
                       "float64": np.float64,
                       "int16": np.int64,
                       "int32": np.int64,
                       "int64": np.int64,
                       "uint16": np.uint64,
                       "uint32": np.uint64,
                       "uint64": np.uint64,
                       "complex64": np.complex128,
                       "complex128": np.complex128,
                       "bool": np.bool})

        # TODO: in pyauto compatibility mode, the run function has to be defined differently. Most importantly,
        #  all function parameters except the continuation parameter have to enter the function via a single list.
        #  This works only for scalar parameters. Thus, other parameters should be saved to the file and loaded instead of imports
        #  above the function definition.
        super().__init__(ops=ops, dtypes=dtypes, name=name, float_default_type=float_default_type,
                         imports=imports, build_dir=build_dir, code_gen=FortranGen(), **kwargs)

        self._imports = []
        self.npar = 0
        self.ndim = 0
        self._idx = "()"
        self._file_ending = ".f90"
        self._start_idx = 1
        self._op_calls = {key: {'shapes': [], 'names': []} for key in fortran_identifiers}
        self._defined_vars = []
        self._func_args = []
        self._lhs_vars = []
        self._global_variables = ["double precision :: PI = 4.0*atan(1.0)"]
        self._state_vars = {}

    def clear(self) -> None:
        """Removes all layers, variables and operations from graph. Deletes build directory.
        """

        # call parent method
        super().clear()

        # delete fortran-specific temporary files
        wdir = f"{self._orig_dir}/{self._build_dir}" if self._build_dir != self._orig_dir else self._orig_dir
        for f in [f for f in os.listdir(wdir)
                  if "cpython" in f and (self._func_name in f or self._file_name in f) and f[-3:] == ".so"]:
            os.remove(f"{wdir}/{f}")

    def _graph_to_str(self, code_gen: CodeGen = None, rhs_indices: list = None, state_var: str = None,
                      rhs_var: str = None, backend_funcs: dict = None):

        if not code_gen:
            code_gen = self._code_gen

        # add variable declarations for all state variables that will be extracted from state vector
        indices_new = []
        if rhs_indices:

            # go through each to-be-extracted state variable
            for i in range(len(rhs_indices)):

                idx = rhs_indices[i]

                # re-define the extraction index to account for fortran indexing
                indices_new.append(idx + self._start_idx if type(idx) is int else (idx[0] + self._start_idx, idx[1]))

            # go through all state-variables
            for var in self.graph.var_updates['DEs']:

                # add variable declarations for fortran subroutine
                dtype, _, shape = self._get_var_declaration_info(var)
                if var not in self._defined_vars:
                    code_gen.add_code_line(f"{dtype} :: {var}{shape}")
                    self._defined_vars.append(var)

        # add variable declarations for all temporary, non-state variables used in the fortran subroutine
        for var in self.graph.var_updates['non-DEs']:
            dtype, _, shape = self._get_var_declaration_info(var)
            if var not in self._defined_vars:
                code_gen.add_code_line(f"{dtype} :: {var}{shape}")
                self._defined_vars.append(var)

        code_gen.add_linebreak()

        # call parent method
        return super()._graph_to_str(code_gen=code_gen, rhs_indices=indices_new, state_var=state_var, rhs_var=rhs_var,
                                     backend_funcs=backend_funcs)

    def _generate_update_equations(self, code_gen: CodeGen, nodes: dict, rhs_var: str = None, indices: list = None,
                                   funcs: dict = None):

        # collect right-hand side expression and all input variables to these expressions
        func_args, expressions, var_names, defined_vars = [], [], [], []
        for node, update in nodes.items():

            # create pyrates-specific fortran function calls where necessary
            vshape = self._get_var_shape(update)
            for op in self._op_calls:
                if op in str(self.graph.node_to_expr(update)[1]):

                    if vshape not in self._op_calls[op]['shapes']:

                        # create new fortran function with output dimensions matching the variable shape
                        self._op_calls[op]['shapes'].append(vshape)
                        if self._op_calls[op]['names']:
                            counter = int(self._op_calls[op]['names'][-1].split('_')[-1])
                        else:
                            counter = 1
                        fstr, fcall = get_fortran_func(op, out_shape=self._var_shape_to_str(vshape), idx=counter)
                        self._imports.append(fstr)
                        self._op_calls[op]['names'].append(fcall)

                    else:

                        # retrieve fortran function specified for the shape of this variable
                        idx = self._op_calls[op]['shapes'].index(vshape)
                        fcall = self._op_calls[op]['names'][idx]

                    # remember the new function definition, so that it can replace the old function call later on
                    funcs[op] = fcall
                    break

            # collect expression and variables of right-hand side of equation
            expr_args, expr = self.graph.node_to_expr(update, **funcs)
            func_args.extend(expr_args)
            expressions.append(self._expr_to_str(expr, var_dim=vshape[0] if vshape else 1))

            # process left-hand side of equation
            var = self.get_var(node)
            vshape = self._get_var_shape(node)

            if 'expr' in var:

                # collect expression and variables of left-hand side indexing operation
                idx_args, lhs = self.graph.node_to_expr(node, **funcs)
                lhs_var = self._expr_to_str(lhs, var_dim=vshape[0] if vshape else 1)
                if not idx_args:
                    idx_args.append(lhs_var.split('(')[0])
                func_args.extend(idx_args)
                self._lhs_vars.append(idx_args[0])

            else:

                # process normal update of left-hand side variable
                lhs_var = var['symbol'].name

            # collect variable name of left-hand side of equation
            var_names.append(lhs_var)

        # add the left-hand side assignments of the collected right-hand side expressions to the code generator
        if rhs_var:

            # differential equation (DE) update
            if not indices:
                raise ValueError('State variables need to be stored in a single state vector, for which the indices '
                                 'have to be passed to this method.')

            # DE updates stored in a state-vector
            for idx, expr, var in zip(indices, expressions, var_names):
                index = self._idx_to_str(idx, var)
                code_gen.add_code_line(f"{rhs_var}{index} = {expr}")

            # add rhs var to function arguments
            func_args = [rhs_var] + func_args

        else:

            # non-differential equation update
            var_names, expressions, undefined_vars, defined_vars = sort_equations(var_names, expressions)
            func_args.extend(undefined_vars)

            if indices:
                raise ValueError('Indices to non-state variables should be defined in the respective equations, not'
                                 'be passed to this method.')

            for target_var, expr in zip(var_names, expressions):
                code_gen.add_code_line(f"{target_var} = {expr}")

        return func_args, defined_vars, code_gen

    def _generate_func_head(self, func_name: str, code_gen: CodeGen, state_var: str = None, func_args: list = None,
                            imports: list = None):

        # pre-process function arguments
        if not func_args:
            func_args = []
        state_vars = ['t', state_var]
        _, indices = np.unique(func_args, return_index=True)
        func_args = state_vars + [func_args[idx] for idx in np.sort(indices)]

        # add function header
        code_gen.add_code_line(f"subroutine {func_name}({','.join(func_args)})")
        code_gen.add_linebreak()
        code_gen.add_code_line("implicit none")
        code_gen.add_linebreak()

        # add function argument declarations
        for arg in func_args:
            dtype, intent, shape = self._get_var_declaration_info(arg)
            if arg not in self._defined_vars:
                code_gen.add_code_line(f"{dtype}, intent({intent}) :: {arg}{shape}")
                self._defined_vars.append(arg)
        self._func_args = func_args[3:]

        return func_args, code_gen

    def _generate_func_tail(self, code_gen: CodeGen, rhs_var: str = None):

        # end the subroutine
        code_gen.add_code_line(f"end subroutine {self._func_name}")
        code_gen.add_linebreak()
        code_gen.add_linebreak()

        # add additional function definitions to module
        for imp in self._imports:
            code_gen.add_code_line(imp)
            code_gen.add_linebreak()
            code_gen.add_linebreak()

        # end the module
        code_gen.add_code_line(f"end module {self._file_name}")

        return code_gen

    def _generate_func(self, func_str: str, func_name: str = None, file_name: str = None, build_dir: str = None,
                       decorator: Any = None, **kwargs):

        # preparations
        if not file_name:
            file_name = self._file_name
        if not build_dir:
            build_dir = f'{self._orig_dir}/{self._build_dir}' if self._build_dir else self._orig_dir
        if not func_name:
            func_name = self._func_name

        # add global variables and module definition above the funcion
        code_gen = FortranGen()
        code_gen.add_code_line(f"module {self._file_name}")
        code_gen.add_linebreak()
        for line in self._global_variables:
            code_gen.add_code_line(line)
        code_gen.add_linebreak()
        code_gen.add_code_line("contains")
        code_gen.add_linebreak()
        code_gen.add_linebreak()
        code_gen.add_code_line(func_str)

        # save rhs function to file
        f2py.compile(code_gen.generate(), modulename=file_name, extension=self._file_ending,
                     source_fn=f'{build_dir}/{file_name}{self._file_ending}', verbose=False)

        # import function from temporary file
        exec(f"from {file_name} import {file_name}", globals())
        exec(f"rhs_eval = {file_name}.{func_name}", globals())
        rhs_eval = globals().pop('rhs_eval')

        # apply function decorator
        if decorator:
            rhs_eval = decorator(rhs_eval, **kwargs)

        return rhs_eval

    def _get_var_declaration_info(self, var: str) -> tuple:
        try:
            val = self.get_var(var)['value']
            dtype = 'double precision' if 'float' in str(val.dtype) else 'integer'
            intent = 'inout' if var == 'state_vec_update' or var in self._lhs_vars else 'in'
            shape = self._var_shape_to_str(var)
            if not shape and 'state_vec' in var:
                shape = '(1)'
        except KeyError:
            dtype = 'double precision'
            intent = 'in'
            shape = ''
        return dtype, intent, shape

    def _expr_to_str(self, expr: Any, expr_str: str = None, var_dim: int = 1) -> str:
        if not expr_str:
            expr_str = str(expr)
            for arg in expr.args:
                expr_str = expr_str.replace(str(arg), self._expr_to_str(arg))
        while 'pr_2d_index(' in expr_str:
            # replace `index` calls with brackets-based indexing
            start = expr_str.find('pr_2d_index(')
            end = expr_str[start:].find(')') + 1
            try:
                new_idx1 = int(expr.args[1]) + self._start_idx
                new_idx2 = int(expr.args[2]) + self._start_idx
                expr_str = expr_str.replace(expr_str[start:start + end],
                                            f"{expr.args[0]}{self._idx[0]}{new_idx1}, {new_idx2}{self._idx[1]}")
            except TypeError:
                raise NotImplementedError('The FortranBackend does not allow for indexing in multiple dimensions via '
                                          'arrays. Please choose another backend for this type of operation.')
        while 'pr_axis_index(' in expr_str:
            # replace `index` calls with brackets-based indexing
            start = expr_str.find('pr_axis_index(')
            end = expr_str[start:].find(')') + 1
            if len(expr.args) == 1 and var_dim == 1:
                expr_str = expr_str.replace(expr_str[start:start + end], f"{expr.args[0]}{self._idx[0]}:{self._idx[1]}")
            elif len(expr.args) == 1:
                idx = ','.join([':' for _ in range(var_dim)])
                expr_str = expr_str.replace(expr_str[start:start + end],
                                            f"{expr.args[0]}{self._idx[0]}{idx}{self._idx[1]}")
            else:
                idx = f','.join([':' for _ in range(expr.args[2])])
                idx = f'{idx},{expr.args[1] + self._start_idx}'
                expr_str = expr_str.replace(expr_str[start:start + end],
                                            f"{expr.args[0]}{self._idx[0]}{idx}{self._idx[1]}")
        while '((' in expr_str:
            # replace `index` calls with brackets-based indexing
            idx_l = expr_str.find('((') + 1
            idx_r = expr_str[idx_l:].find(')') + 1
            expr_str = expr_str.replace(expr_str[idx_l:idx_l+idx_r], f"[{expr_str[idx_l+1:idx_l+idx_r-1]}]")
            expr_str = expr_str.replace("], 0)", "], 1)")
        return super()._expr_to_str(expr=expr, expr_str=expr_str)

    def _idx_to_str(self, idx: str, var: str):
        return f"{self._idx[0]}{idx}{self._idx[1]}"

    def _get_unique_var_name(self, var: str) -> str:
        while not self._is_unique(var):
            try:
                v_split = var.split('_')
                counter = int(v_split.pop(-1))
                var = f"{'_'.join(v_split)}_{counter + 1}"
            except TypeError:
                var = f"{var}_1"
        return var

    def _is_unique(self, var_name: str) -> bool:
        try:
            self.get_var(var_name, get_key=True)
            return False
        except KeyError:
            return True

    def _get_var_shape(self, var: str) -> tuple:
        v = self.get_var(var)
        try:
            return v['value'].shape
        except KeyError:
            v = self.graph.eval_node(var)
            try:
                return v.shape
            except AttributeError:
                return ()

    def _var_shape_to_str(self, v: Union[str, tuple]):
        if type(v) is str:
            v = self._get_var_shape(v)
        s = f"{v}" if sum(v) > 1 else ""
        if len(s) > 3 and s[-2] == ',':
            s = s[:-2] + s[-1]
        return s

    @staticmethod
    def _compare_shapes(op1: Any, op2: Any, index=False) -> bool:
        """Checks whether the shapes of op1 and op2 are compatible with each other.

        Parameters
        ----------
        op1
            First operator.
        op2
            Second operator.

        Returns
        -------
        bool
            If true, the shapes of op1 and op2 are compatible.

        """

        if hasattr(op1, 'shape') and hasattr(op2, 'shape'):
            if op1.shape == op2.shape:
                return True
            elif len(op1.shape) > 1 and len(op2.shape) > 1:
                return True
            elif len(op1.shape) == 0 and len(op2.shape) == 0:
                return True
            else:
                return False
        elif hasattr(op1, 'shape'):
            if sum(op1.shape) > 0:
                return False
            else:
                return True
        else:
            return True


class PyAutoBackend(FortranBackend):

    _blocked_indices = (10, 32)
    _time_scale = 1e1

    def __init__(self,
                 ops: Optional[Dict[str, str]] = None,
                 dtypes: Optional[Dict[str, object]] = None,
                 name: str = 'net_0',
                 float_default_type: str = 'float64',
                 imports: Optional[List[str]] = None,
                 build_dir: Optional[str] = None,
                 **kwargs
                 ) -> None:

        # validate auto directory
        try:
            self._auto_dir = kwargs.pop('auto_dir')
        except KeyError as e:
            print("WARNING: No auto installation directory has been passed to the backend. If this directory has "
                  "not been set as an environment variable, this will cause an error when attempting to perform "
                  "simulations. The Auto-07p installation directory can be passed to the `run()` method as follows: "
                  "apply_kwargs={'backend_kwargs': {'auto_dir': '<path>'}}.")

        # set auto default constants
        self.auto_constants = {'NDIM': 1, 'NPAR': 1,
                              'IPS': -2, 'ILP': 0, 'ICP': [14], 'NTST': 1, 'NCOL': 3, 'IAD': 0, 'ISP': 0, 'ISW': 1,
                              'IPLT': 0, 'NBC': 0, 'NINT': 0, 'NMX': 10000, 'NPR': 1, 'MXBF': 10, 'IID': 2, 'ITMX': 2,
                              'ITNW': 5, 'NWTN': 2, 'JAC': 0, 'EPSL': 1e-6, 'EPSU': 1e-6, 'EPSS': 1e-4, 'IRS': 0,
                              'DS': 1e-4, 'DSMIN': 1e-8, 'DSMAX': 1e-2, 'IADS': 1, 'THL': {},
                              'THU': {}, 'UZR': {}, 'STOP': {}}

        super().__init__(ops=ops, dtypes=dtypes, name=name, float_default_type=float_default_type, imports=imports,
                         build_dir=build_dir, **kwargs)

    def clear(self) -> None:
        """Removes all layers, variables and operations from graph. Deletes build directory.
        """

        # call parent method
        super().clear()

        # delete fortran-specific temporary files
        wdir = f"{self._orig_dir}/{self._build_dir}" if self._build_dir != self._orig_dir else self._orig_dir
        fendings = ["so", "exe", "mod", "o"]
        for f in [f for f in os.listdir(wdir)]:
            fname = self._file_name
            fsplit = f.split('.')
            if fsplit[0] == fname and fsplit[-1] in fendings:
                os.remove(f"{wdir}/{f}")
            elif fsplit[0] == 'fort':
                os.remove(f"{wdir}/{f}")

    def _generate_func(self, func_str: str, func_name: str = None, file_name: str = None, build_dir: str = None,
                       decorator: Any = None, **kwargs):

        # preparations
        if not file_name:
            file_name = self._file_name
        if not build_dir:
            build_dir = f'{self._orig_dir}/{self._build_dir}' if self._build_dir else self._orig_dir

        # add auto-related subroutines
        code_gen = FortranGen()
        code_gen.add_code_line(func_str)
        code_gen.add_linebreak()
        code_gen.add_linebreak()
        code_gen, constants_gen = self._generate_auto_routines(code_gen, file_name, func_name, **kwargs)
        func_str = code_gen.generate()
        code_gen.clear()

        # add global variables and module definition above the funcion definitions
        code_gen = FortranGen()
        code_gen.add_code_line(f"module {self._file_name}")
        code_gen.add_linebreak()
        for line in self._global_variables:
            code_gen.add_code_line(line)
        code_gen.add_linebreak()
        code_gen.add_code_line("contains")
        code_gen.add_linebreak()
        code_gen.add_linebreak()
        code_gen.add_code_line(func_str)

        # write rhs function to file
        f2py.compile(code_gen.generate(), modulename=file_name, extension=self._file_ending,
                     source_fn=f'{build_dir}/{file_name}{self._file_ending}', verbose=False)

        # import function from temporary file
        exec(f"from {file_name} import func", globals())
        rhs_eval = globals().pop('func')

        # apply function decorator
        if decorator:
            rhs_eval = decorator(rhs_eval, **kwargs)

        # write auto constants to file
        try:
            with open(f'{build_dir}/c.ivp', 'wt') as cfile:
                cfile.write(constants_gen.generate())
        except FileNotFoundError:
            with open(f'{build_dir}/c.ivp', 'xt') as cfile:
                cfile.write(constants_gen.generate())

        return rhs_eval

    def _generate_auto_routines(self, code_gen: FortranGen, module_name: str, func_name: str, **kwargs) -> tuple:

        # wrapper to the right-hand side evaluation function
        ####################################################

        # add function header that Auto-07p expects
        code_gen.add_code_line("subroutine func(ndim,state_vec,icp,args,ijac,state_vec_update,dfdu,dfdp)")
        code_gen.add_linebreak()

        # load the module in which the pyrates function has been defined
        code_gen.add_code_line(f"use {module_name}")

        # declare auto-related variables
        code_gen.add_code_line("implicit none")
        code_gen.add_code_line("integer, intent(in) :: ndim, icp(*), ijac")
        code_gen.add_code_line("double precision, intent(in) :: state_vec(ndim), args(*)")
        code_gen.add_code_line("double precision, intent(out) :: state_vec_update(ndim)")
        code_gen.add_code_line("double precision, intent(inout) :: dfdu(ndim,ndim), dfdp(ndim,*)")

        # declare variables that need to be extracted from args array
        for arg in self._func_args.copy():
            dtype, _, shape = self._get_var_declaration_info(arg)
            try:
                s = [int(s) for s in shape[1:-1].split(',')]
                if len(s) == 1:
                    val = self.graph.eval_node(var)
                    val_init = f"{dtype}, parameter :: {var}{shape} = (/{','.join(val.tolist())}/)"
                    self._global_variables.append(val_init)
                else:
                    raise NotImplementedError('The PyAutoBackend cannot generate run functions that include parameters '
                                              'of dimensionality 2 or higher.')

                idx = self._func_args.index(arg)
                self._func_args.pop(idx)
            except ValueError:
                code_gen.add_code_line(f"{dtype} :: {arg}{shape}")
        code_gen.add_linebreak()

        # extract variables from args array
        increment = 1
        for i, arg in enumerate(self._func_args):
            idx = i + increment
            if idx >= self._blocked_indices[0] and idx <= self._blocked_indices[1]:
                increment += self._blocked_indices[1] - self._blocked_indices[0]
                idx += increment
            code_gen.add_code_line(f"{arg} = args({idx})")

        # call the pyrates subroutine
        additional_args = f", {', '.join(self._func_args)}" if self._func_args else ""
        code_gen.add_code_line(f"call {func_name}(args(14), state_vec, state_vec_update{additional_args})")
        code_gen.add_linebreak()
        code_gen.add_code_line("end subroutine func")

        # routine that sets up an initial value problem
        ###############################################

        # generate subroutine header
        code_gen.add_linebreak()
        code_gen.add_code_line("subroutine stpnt(ndim, state_vec, args, t)")
        code_gen.add_linebreak()
        code_gen.add_code_line("implicit None")
        code_gen.add_code_line("integer, intent(in) :: ndim")
        code_gen.add_code_line("double precision, intent(inout) :: state_vec(ndim), args(*)")
        code_gen.add_code_line("double precision, intent(in) :: t")
        code_gen.add_linebreak()

        # define parameter values
        code_gen.add_linebreak()
        increment = 1
        idx = 0
        for i, arg in enumerate(self._func_args):
            idx = i + increment
            if idx >= self._blocked_indices[0] and idx <= self._blocked_indices[1]:
                increment += self._blocked_indices[1] - self._blocked_indices[0]
                idx += increment
            val = self.graph.eval_node(arg)
            code_gen.add_code_line(f"args({idx}) = {val}")
        npar = idx

        # define initial state
        code_gen.add_linebreak()
        for key, info in self._state_vars.items():
            idx = info['index']
            v_init = info['init']
            code_gen.add_code_line(f"state_vec({idx}) = {v_init}")
        code_gen.add_linebreak()

        # end subroutine
        code_gen.add_linebreak()
        code_gen.add_code_line("end subroutine stpnt")
        code_gen.add_linebreak()

        # dummy routines (could be made available for more complex Auto-07p usages)
        ###########################################################################

        code_gen.add_linebreak()
        for routine in ['bcnd', 'icnd', 'fopt', 'pvls']:
            code_gen.add_linebreak()
            code_gen.add_code_line(f"subroutine {routine}")
            code_gen.add_code_line(f"end subroutine {routine}")
            code_gen.add_linebreak()

        # create auto constants file
        ############################

        self.auto_constants['NDIM'] = len(self.graph.eval_node('state_vec'))
        self.auto_constants['NPAR'] = npar
        for key, val in kwargs:
            if key in self.auto_constants:
                self.auto_constants[key] = kwargs.pop(key)

        # write auto constants to string
        const_gen = FortranGen()
        for key, val in self.auto_constants.items():
            const_gen.add_code_line(f"{key} = {val}")

        return code_gen, const_gen

    def _solve_ivp(self, solver: str, T: float, state_vec: np.ndarray, rhs: np.ndarray, dt: float,
                   eval_times: np.ndarray, dts: float, rhs_func: Callable, *args, **kwargs) -> np.ndarray:
        """

        Parameters
        ----------
        rhs_func
        func_args
        state_vars
        T
        dt
        dts
        t
        solver
        output_indices
        kwargs

        Returns
        -------

        """

        from pyrates.utility.pyauto import PyAuto
        from scipy.interpolate import interp1d

        # preparations
        pyauto = PyAuto(working_dir=self._build_dir, auto_dir=self._auto_dir)
        ds = dt * self._time_scale
        dsmin = ds*1e-2
        auto_defs = {'DSMIN': dsmin, 'DSMAX': ds*1e1, 'NMX': int(T/dsmin)}
        for key, val in auto_defs.items():
            if key not in kwargs:
                kwargs[key] = val
        ndim = len(state_vec)

        # solve ivp
        kwargs_tmp = {key: val for key, val in kwargs.items() if key in self.auto_constants}
        pyauto.run(e=self._file_name, c='ivp', DS=ds, name='t', UZR={14: T}, STOP={'UZ1'},
                   **kwargs_tmp)

        # extract results
        extract = [f'U({i+1})' for i in range(ndim)]
        extract.append('PAR(14)')
        results_tmp = pyauto.extract(keys=extract, cont='t')
        times = results_tmp.pop('PAR(14)')
        results = []
        for i in range(ndim):
            y_inter = interp1d(times, np.squeeze(results_tmp.pop(f'U({i+1})')))
            results.append(y_inter(eval_times))

        return np.asarray(results).T

    def _idx_to_str(self, idx, var: str):
        index = f"{self._idx[0]}{idx}{self._idx[1]}"
        value = self.graph.eval_node(var)
        self._state_vars[var] = {'index': index, 'init': value}
        return index
