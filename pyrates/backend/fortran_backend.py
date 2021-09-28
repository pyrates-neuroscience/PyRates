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

from .base_backend import BaseBackend, CodeGen, sort_equations
from .fortran_funcs import get_fortran_func, fortran_identifiers
from .base_funcs import *

# external imports
from typing import Optional, Dict, Callable, List, Any
import os
import sys
from numpy import f2py
import numpy as np

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


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
        """Instantiates numpy backend, i.e. a compute graph with numpy operations.
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
        self._auto_files_generated = False
        self._idx = "()"
        self._file_ending = ".f90"
        self._start_idx = 1
        self._op_calls = {key: {'shapes': [], 'names': []} for key in fortran_identifiers}
        self._defined_vars = []
        self._lhs_vars = []

    def _graph_to_str(self, code_gen: CodeGen = None, rhs_indices: list = None, state_var: str = None,
                      rhs_var: str = None, backend_funcs: dict = None):

        # add variable declarations for all state variables that will be extracted from state vector
        if not code_gen:
            code_gen = self._code_gen
        indices_new = []
        if rhs_indices:
            for i in range(len(rhs_indices)):
                idx = rhs_indices[i]
                indices_new.append(idx + self._start_idx if type(idx) is int else (idx[0] + self._start_idx, idx[1]))
            for var in self.graph.var_updates['DEs']:
                dtype, _, shape = self._get_var_declaration_info(var)
                if var not in self._defined_vars:
                    code_gen.add_code_line(f"{dtype} :: {var}{shape}")
                    self._defined_vars.append(var)

        # add variable declarations for all state variables that will be extracted from state vector
        for var in self.graph.var_updates['non-DEs']:
            dtype, _, shape = self._get_var_declaration_info(var)
            if var not in self._defined_vars:
                code_gen.add_code_line(f"{dtype} :: {var}{shape}")
                self._defined_vars.append(var)

        # call parent method
        return super()._graph_to_str(code_gen=code_gen, rhs_indices=indices_new, state_var=state_var, rhs_var=rhs_var,
                                     backend_funcs=backend_funcs)

    def _generate_update_equations(self, code_gen: CodeGen, nodes: dict, rhs_var: str = None, indices: list = None,
                                   funcs: dict = None):

        # collect right-hand side expression and all input variables to these expressions
        func_args, expressions, var_names, defined_vars = [], [], [], []
        for node, update in nodes.items():

            # collect expression and variables of right-hand side of equation
            val = self.graph.eval_node(update)
            for op in self._op_calls:
                if op in update:
                    if hasattr(val, 'shape'):
                        shape = f"{val.shape}" if len(val.shape) > 1 else "(1)"
                    else:
                        shape = ""
                    if shape not in self._op_calls[op]['shapes']:
                        self._op_calls[op]['shapes'].append(shape)
                        if self._op_calls[op]['names']:
                            counter = int(self._op_calls[op]['names'][-1].split('_')[-1])
                        else:
                            counter = 1
                        fstr, fcall = get_fortran_func(op, out_shape=shape, idx=counter)
                        self._imports.append(fstr)
                        self._op_calls[op]['names'].append(fcall)
                    else:
                        idx = self._op_calls[op]['shapes'].index(shape)
                        fcall = self._op_calls[op]['names'][idx]
                    var = self.get_var(update)
                    funcs[str(var['symbol'])] = fcall
                    break

            expr_args, expr = self.graph.node_to_expr(update, **funcs)
            func_args.extend(expr_args)
            expressions.append(self._expr_to_str(expr,
                                                 var_dim=val.shape[0] if hasattr(val, 'shape') and val.shape else 1))

            # process left-hand side of equation
            var = self.get_var(node)
            val = var['value'] if 'value' in var else self.graph.eval_node(node)
            if 'expr' in var:
                # process indexing of left-hand side variable
                idx_args, lhs = self.graph.node_to_expr(node, **funcs)
                func_args.extend(idx_args)
                lhs_var = self._expr_to_str(lhs, var_dim=val.shape[0] if hasattr(val, 'shape') and val.shape else 1)
                self._lhs_vars.append(idx_args[0])
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
                    code_gen.add_code_line(f"{rhs_var}{self._idx[0]}{idx}{self._idx[1]} = {expr}")

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

            for target_var, expr in zip(var_names, expressions):
                code_gen.add_code_line(f"{target_var} = {expr}")

        code_gen.add_linebreak()

        return func_args, defined_vars, code_gen

    def _generate_func_head(self, func_name: str, code_gen: CodeGen, state_var: str = None, func_args: list = None,
                            imports: list = None):

        if not func_args:
            func_args = []
        state_vars = ['t', state_var]
        _, indices = np.unique(func_args, return_index=True)
        func_args = state_vars + [func_args[idx] for idx in np.sort(indices)]

        # add function header
        code_gen.add_linebreak()
        code_gen.add_code_line(f"module {self._file_name}")
        code_gen.add_linebreak()
        code_gen.add_code_line("contains")
        code_gen.add_linebreak()
        code_gen.add_code_line(f"subroutine {func_name}({','.join(func_args)})")
        code_gen.add_linebreak()
        code_gen.add_code_line("implicit none")
        code_gen.add_linebreak()
        for arg in func_args:
            dtype, intent, shape = self._get_var_declaration_info(arg)
            if arg not in self._defined_vars:
                code_gen.add_code_line(f"{dtype}, intent({intent}) :: {arg}{shape}")
                self._defined_vars.append(arg)

        return func_args, code_gen

    def _generate_func_tail(self, code_gen: CodeGen, rhs_var: str = None):

        code_gen.add_code_line(f"end subroutine {self._func_name}")
        code_gen.add_linebreak()

        # add additional function definitions to module
        for imp in self._imports:
            code_gen.add_code_line(imp)
            code_gen.add_linebreak()
        code_gen.add_linebreak()
        code_gen.add_code_line(f"end module {self._file_name}")

        return code_gen

    def _generate_func(self, func_str: str, func_name: str = None, file_name: str = None, build_dir: str = None,
                       decorator: Any = None, **kwargs):

        if not file_name:
            file_name = self._file_name
        if not build_dir:
            build_dir = f'{self._orig_dir}/{self._build_dir}' if self._build_dir else self._orig_dir
        if not func_name:
            func_name = self._func_name

        # save rhs function to file
        f2py.compile(func_str, modulename=file_name, extension=self._file_ending,
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
            if val.shape:
                shape = f'{val.shape}'
                if shape[-2] == ',':
                    shape = shape[:-2] + shape[-1]
            else:
                shape = '' if 'state_vec' not in var else '(1)'
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

    # def compile(self, build_dir: Optional[str] = None, decorator: Optional[Callable] = None, **kwargs) -> tuple:
    #     """Compile the graph layers/operations. Creates python files containing the functions in each layer.
    #
    #     Parameters
    #     ----------
    #     build_dir
    #         Directory in which to create the file structure for the simulation.
    #     decorator
    #         Decorator function that should be applied to the right-hand side evaluation function.
    #     kwargs
    #         decorator keyword arguments
    #
    #     Returns
    #     -------
    #     tuple
    #         Contains tuples of layer run functions and their respective arguments.
    #
    #     """
    #
    #     # preparations
    #     ##############
    #
    #     # remove empty layers and operators
    #     new_layer_idx = 0
    #     for layer_idx, layer in enumerate(self.layers.copy()):
    #         for op in layer.copy():
    #             if op is None:
    #                 layer.pop(layer.index(op))
    #         if len(layer) == 0:
    #             self.layers.pop(new_layer_idx)
    #         else:
    #             new_layer_idx += 1
    #
    #     # remove previously imported rhs_funcs from system
    #     if 'rhs_func' in sys.modules:
    #         del sys.modules['rhs_func']
    #
    #     # collect state variable and parameter vectors
    #     state_vars, params, var_map = self._process_vars()
    #
    #     # update state variable getter operations to include the full state variable vector as first argument
    #     y = self.get_var('y')
    #     for key, (vtype, idx) in var_map.items():
    #         var = self.get_var(key)
    #         if vtype == 'state_var' and var.short_name != 'y':
    #             var.args[0] = y
    #             var.build_op(var.args)
    #
    #     # create rhs evaluation function
    #     ################################
    #
    #     # set up file header
    #     func_gen = FortranGen()
    #     for import_line in self._imports:
    #         func_gen.add_code_line(import_line)
    #         func_gen.add_linebreak()
    #     func_gen.add_linebreak()
    #
    #     # define function head
    #     func_gen.add_indent()
    #     if self.pyauto_compat:
    #         func_gen.add_code_line("subroutine func(ndim,y,icp,args,ijac,y_delta,dfdu,dfdp)")
    #         func_gen.add_linebreak()
    #         func_gen.add_code_line("implicit none")
    #         func_gen.add_linebreak()
    #         func_gen.add_code_line("integer, intent(in) :: ndim, icp(*), ijac")
    #         func_gen.add_linebreak()
    #         func_gen.add_code_line("double precision, intent(in) :: y(ndim), args(*)")
    #         func_gen.add_linebreak()
    #         func_gen.add_code_line("double precision, intent(out) :: y_delta(ndim)")
    #         func_gen.add_linebreak()
    #         func_gen.add_code_line("double precision, intent(inout) :: dfdu(ndim,ndim), dfdp(ndim,*)")
    #         func_gen.add_linebreak()
    #     else:
    #         func_gen.add_code_line("subroutine func(ndim,t,y,args,y_delta)")
    #         func_gen.add_linebreak()
    #         func_gen.add_code_line("implicit none")
    #         func_gen.add_linebreak()
    #         func_gen.add_code_line("integer, intent(in) :: ndim")
    #         func_gen.add_linebreak()
    #         func_gen.add_code_line("double precision, intent(in) :: t, y(ndim)")
    #         func_gen.add_linebreak()
    #         func_gen.add_code_line("double precision, intent(in) :: args(*)")
    #         func_gen.add_linebreak()
    #         func_gen.add_code_line("double precision, intent(out) :: y_delta(ndim)")
    #         func_gen.add_linebreak()
    #
    #     # declare variable types
    #     func_gen.add_code_line("double precision ")
    #     for key in var_map:
    #         var = self.get_var(key)
    #         if "float" in str(var.dtype) and var.short_name != 'y_delta':
    #             func_gen.add_code_line(f"{var.short_name},")
    #     if "," in func_gen.code[-1]:
    #         func_gen.code[-1] = func_gen.code[-1][:-1]
    #     else:
    #         func_gen.code.pop(-1)
    #     func_gen.add_linebreak()
    #     func_gen.add_code_line("integer ")
    #     for key in var_map:
    #         var = self.get_var(key)
    #         if "int" in str(var.dtype):
    #             func_gen.add_code_line(f"{var.short_name},")
    #     if "," in func_gen.code[-1]:
    #         func_gen.code[-1] = func_gen.code[-1][:-1]
    #     else:
    #         func_gen.code.pop(-1)
    #     func_gen.add_linebreak()
    #
    #     # declare constants
    #     args = [None for _ in range(len(params))]
    #     func_gen.add_code_line("! declare constants")
    #     func_gen.add_linebreak()
    #     updates, indices = [], []
    #     i = 0
    #     for key, (vtype, idx) in var_map.items():
    #         if vtype == 'constant':
    #             var = params[idx-self.idx_start][1]
    #             if var.short_name != 'y_delta':
    #                 func_gen.add_code_line(f"{var.short_name} = args({idx})")
    #                 func_gen.add_linebreak()
    #                 args[idx-1] = var
    #                 updates.append(f"{var.short_name}")
    #                 indices.append(i)
    #                 i += 1
    #     func_gen.add_linebreak()
    #
    #     # extract state variables from input vector y
    #     func_gen.add_code_line("! extract state variables from input vector")
    #     func_gen.add_linebreak()
    #     for key, (vtype, idx) in var_map.items():
    #         var = self.get_var(key)
    #         if vtype == 'state_var':
    #             func_gen.add_code_line(f"{var.short_name} = {var.value}")
    #             func_gen.add_linebreak()
    #     func_gen.add_linebreak()
    #
    #     # add equations
    #     func_gen.add_code_line("! calculate right-hand side update of equation system")
    #     func_gen.add_linebreak()
    #     arg_updates = []
    #     for i, layer in enumerate(self.layers):
    #         for j, op in enumerate(layer):
    #             lhs = op.value.split("=")[0]
    #             lhs = lhs.replace(" ", "")
    #             find_arg = [arg == lhs for arg in updates]
    #             if any(find_arg):
    #                 idx = find_arg.index(True)
    #                 arg_updates.append((updates[idx], indices[idx]))
    #             func_gen.add_code_line(op.value)
    #             func_gen.add_linebreak()
    #     func_gen.add_linebreak()
    #
    #     # update parameters where necessary
    #     # func_gen.add_code_line("! update system parameters")
    #     # func_gen.add_linebreak()
    #     # for upd, idx in arg_updates:
    #     #     update_str = f"args{self.idx_l}{idx}{self.idx_r} = {upd}"
    #     #     if f"    {update_str}" not in func_gen.code:
    #     #         func_gen.add_code_line(update_str)
    #     #         func_gen.add_linebreak()
    #
    #     # end function
    #     func_gen.add_code_line(f"end subroutine func")
    #     func_gen.add_linebreak()
    #     func_gen.remove_indent()
    #
    #     # save rhs function to file
    #     fname = f'{self._build_dir}/rhs_func'
    #     f2py.compile(func_gen.generate(), modulename='rhs_func', extension='.f', source_fn=f'{fname}.f', verbose=False)
    #
    #     # create additional subroutines in pyauto compatibility mode
    #     gen_def = kwargs.pop('generate_auto_def', True)
    #     if self.pyauto_compat and gen_def:
    #         self.generate_auto_def(self._build_dir)
    #
    #     # import function from file
    #     fname_import = 'func' if self.pyauto_compat else 'rhs_eval'
    #     exec(f"from rhs_func import {fname_import}", globals())
    #     rhs_eval = globals().pop('func')
    #
    #     # apply function decorator
    #     if decorator:
    #         rhs_eval = decorator(rhs_eval, **kwargs)
    #
    #     return rhs_eval, args, state_vars, var_map
    #
    # def generate_auto_def(self, directory):
    #     """
    #
    #     Parameters
    #     ----------
    #     directory
    #
    #     Returns
    #     -------
    #
    #     """
    #
    #     if not self.pyauto_compat:
    #         raise ValueError('This method can only be called in pyauto compatible mode. Please set `pyauto_compat` to '
    #                          'True upon calling the `CircuitIR.compile` method.')
    #
    #     # read file
    #     ###########
    #
    #     try:
    #
    #         # read file from excisting system compilation
    #         if directory is None:
    #             directory = self._build_dir
    #         fn = f"{directory}/rhs_func.f" if "rhs_func.f" not in directory else directory
    #         with open(fn, 'rt') as f:
    #             func_str = f.read()
    #
    #     except FileNotFoundError:
    #
    #         # compile system and then read the build files
    #         self.compile(build_dir=directory, generate_auto_def=False)
    #         fn = f"{self._build_dir}/rhs_func.f"
    #         with open(fn, 'r') as f:
    #             func_str = f.read()
    #
    #     end_str = "end subroutine func\n"
    #     idx = func_str.index(end_str)
    #     func_str = func_str[:idx+len(end_str)]
    #
    #     # generate additional subroutines
    #     #################################
    #
    #     func_gen = FortranGen()
    #
    #     # generate subroutine header
    #     func_gen.add_linebreak()
    #     func_gen.add_indent()
    #     func_gen.add_code_line("subroutine stpnt(ndim, y, args, t)")
    #     func_gen.add_linebreak()
    #     func_gen.add_code_line("implicit None")
    #     func_gen.add_linebreak()
    #     func_gen.add_code_line("integer, intent(in) :: ndim")
    #     func_gen.add_linebreak()
    #     func_gen.add_code_line("double precision, intent(inout) :: y(ndim), args(*)")
    #     func_gen.add_linebreak()
    #     func_gen.add_code_line("double precision, intent(in) :: T")
    #     func_gen.add_linebreak()
    #
    #     # declare variable types
    #     func_gen.add_code_line("double precision ")
    #     for key in self.vars:
    #         var = self.get_var(key)
    #         name = var.short_name
    #         if "float" in str(var.dtype) and name != 'y_delta' and name != 'y' and name != 't':
    #             func_gen.add_code_line(f"{var.short_name},")
    #     if "," in func_gen.code[-1]:
    #         func_gen.code[-1] = func_gen.code[-1][:-1]
    #     else:
    #         func_gen.code.pop(-1)
    #     func_gen.add_linebreak()
    #     func_gen.add_code_line("integer ")
    #     for key in self.vars:
    #         var = self.get_var(key)
    #         if "int" in str(var.dtype):
    #             func_gen.add_code_line(f"{var.short_name},")
    #     if "," in func_gen.code[-1]:
    #         func_gen.code[-1] = func_gen.code[-1][:-1]
    #     else:
    #         func_gen.code.pop(-1)
    #     func_gen.add_linebreak()
    #
    #     _, params, var_map = self._process_vars()
    #
    #     # define parameter values
    #     func_gen.add_linebreak()
    #     for key, (vtype, idx) in var_map.items():
    #         if vtype == 'constant':
    #             var = params[idx-self.idx_start][1]
    #             if var.short_name != 'y_delta' and var.short_name != 'y':
    #                 func_gen.add_code_line(f"{var.short_name} = {var}")
    #                 func_gen.add_linebreak()
    #     func_gen.add_linebreak()
    #
    #     # define initial state
    #     func_gen.add_linebreak()
    #     npar = 0
    #     for key, (vtype, idx) in var_map.items():
    #         if vtype == 'constant':
    #             var = params[idx-self.idx_start][1]
    #             if var.short_name != 'y_delta' and var.short_name != 'y':
    #                 func_gen.add_code_line(f"args({idx}) = {var.short_name}")
    #                 func_gen.add_linebreak()
    #                 if idx > npar:
    #                     npar = idx
    #     func_gen.add_linebreak()
    #
    #     func_gen.add_linebreak()
    #     for key, (vtype, idx) in var_map.items():
    #         var = self.get_var(key)
    #         if vtype == 'state_var':
    #             func_gen.add_code_line(f"{var.value} = {var.numpy()}")
    #             func_gen.add_linebreak()
    #     func_gen.add_linebreak()
    #
    #     # end subroutine
    #     func_gen.add_linebreak()
    #     func_gen.add_code_line("end subroutine stpnt")
    #     func_gen.add_linebreak()
    #
    #     # add dummy subroutines
    #     for routine in ['bcnd', 'icnd', 'fopt', 'pvls']:
    #         func_gen.add_linebreak()
    #         func_gen.add_code_line(f"subroutine {routine}")
    #         func_gen.add_linebreak()
    #         func_gen.add_code_line(f"end subroutine {routine}")
    #         func_gen.add_linebreak()
    #     func_gen.add_linebreak()
    #     func_gen.remove_indent()
    #
    #     func_combined = f"{func_str} \n {func_gen.generate()}"
    #     f2py.compile(func_combined, source_fn=fn, modulename='rhs_func', extension='.f', verbose=False)
    #
    #     self.npar = npar
    #     self.ndim = self.get_var('y').shape[0]
    #
    #     # generate constants file
    #     #########################
    #
    #     # declare auto constants and their values
    #     auto_constants = {'NDIM': self.ndim, 'NPAR': self.npar, 'IPS': -2, 'ILP': 0, 'ICP': [14], 'NTST': 1, 'NCOL': 4,
    #                       'IAD': 3, 'ISP': 0, 'ISW': 1, 'IPLT': 0, 'NBC': 0, 'NINT': 0, 'NMX': 10000, 'NPR': 100,
    #                       'MXBF': 10, 'IID': 2, 'ITMX': 8, 'ITNW': 5, 'NWTN': 3, 'JAC': 0, 'EPSL': 1e-7, 'EPSU': 1e-7,
    #                       'EPSS': 1e-5, 'IRS': 0, 'DS': 1e-4, 'DSMIN': 1e-8, 'DSMAX': 1e-2, 'IADS': 1, 'THL': {},
    #                       'THU': {}, 'UZR': {}, 'STOP': {}}
    #
    #     # write auto constants to string
    #     cgen = FortranGen()
    #     for key, val in auto_constants.items():
    #         cgen.add_code_line(f"{key} = {val}")
    #         cgen.add_linebreak()
    #
    #     # write auto constants to file
    #     try:
    #         with open(f'{directory}/c.ivp', 'wt') as cfile:
    #             cfile.write(cgen.generate())
    #     except FileNotFoundError:
    #         with open(f'{directory}/c.ivp', 'xt') as cfile:
    #             cfile.write(cgen.generate())
    #
    #     self._auto_files_generated = True
    #
    #     return fn
    #
    # def to_pyauto(self, directory=None, generate_auto_def=True, **kwargs):
    #     """
    #
    #     Parameters
    #     ----------
    #     directory
    #     generate_auto_def
    #
    #     Returns
    #     -------
    #
    #     """
    #     from pyrates.utility.pyauto import PyAuto
    #     directory = directory if directory else self._build_dir
    #     if generate_auto_def:
    #         self.generate_auto_def(directory)
    #     return PyAuto(directory, **kwargs)

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

    # def _solve(self, rhs_func, func_args, T, dt, dts, t, solver, output_indices, **kwargs):
    #     """
    #
    #     Parameters
    #     ----------
    #     rhs_func
    #     func_args
    #     state_vars
    #     T
    #     dt
    #     dts
    #     t
    #     solver
    #     output_indices
    #     kwargs
    #
    #     Returns
    #     -------
    #
    #     """
    #
    #     if self.pyauto_compat:
    #
    #         from pyrates.utility.pyauto import PyAuto
    #         pyauto = PyAuto(working_dir=self._build_dir)
    #         dsmin = dt*1e-2
    #         auto_defs = {'DSMIN': dsmin, 'DSMAX': dt*1e2, 'NMX': int(T/dsmin), 'NPR': int(dts/dt)}
    #         for key, val in auto_defs.items():
    #             if key not in kwargs:
    #                 kwargs[key] = val
    #         pyauto.run(e='rhs_func', c='ivp', DS=dt, name='t', UZR={14: T}, STOP={'UZ1'}, **kwargs)
    #
    #         extract = [f'U({i+1})' for i in range(self.ndim)]
    #         out_vars = [f'U({i[0]+self.idx_start if type(i) is list else i+self.idx_start})' for i in output_indices]
    #         extract.append('PAR(14)')
    #
    #         results = pyauto.extract(keys=extract, cont='t')
    #         times = results.pop('PAR(14)')
    #
    #         state_vars = self.get_var('y')
    #         for key, val in results.items():
    #             start, stop = key.index('('), key.index(')')
    #             idx = int(key[start+1:stop])
    #             state_vars[idx-self.idx_start] = val[-1]
    #
    #         return times, [results[v] for v in out_vars]
    #
    #     return super()._solve(rhs_func, func_args, T, dt, dts, t, solver, output_indices, **kwargs)

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


class FortranGen(CodeGen):

    n = 70

    def add_code_line(self, code_str):
        """Add code line string to code.
        """
        code_str = code_str.split('\n')
        for code in code_str:
            if '&' not in code:
                code = code.replace('\t', '')
                code = '\t' * self.lvl + code + '\n'
            elif '\n' not in code:
                code = code + '\n'
            if len(code) > self.n and code[-2] != '&':
                idx = self._find_first_op(code, start=0, stop=self.n)
                self.add_code_line(f'{code[0:idx]}&')
                code = f"     & {code[idx:]}"
                self.add_code_line(code)
            else:
                self.code.append(code)

    @staticmethod
    def _find_first_op(code, start, stop):
        if stop < len(code):
            code_tmp = code[start:stop]
            ops = ["+", "-", "*", "/", "**", "^", "%", "<", ">", "==", "!=", "<=", ">="]
            indices = [code_tmp.index(op) for op in ops if op in code_tmp]
            if indices and max(indices) > 0:
                return max(indices)
            idx = start
            for break_sign in [',', ')', ' ']:
                if break_sign in code_tmp:
                    idx_tmp = len(code_tmp) - code_tmp[::-1].index(break_sign)
                    if len(code_tmp)-idx_tmp < len(code_tmp)-idx:
                        idx = idx_tmp
            return idx
        return stop


# def generate_func(self, return_key='f', omit_assign=False, return_dim=None, return_intent='out'):
#     """Generates a function from operator value and arguments"""
#
#     global module_counter
#
#     # function head
#     func_dict = {}
#     func = FortranGen()
#     func.add_linebreak()
#     func.add_indent()
#     fname = self.short_name.lower()
#     func.add_code_line(f"subroutine {fname}({return_key}")
#     for arg in self._op_dict['arg_names']:
#         if arg != return_key:
#             func.add_code_line(f",{arg}")
#     func.add_code_line(")")
#     func.add_linebreak()
#
#     # argument type definition
#     for arg, name in zip(self._op_dict['args'], self._op_dict['arg_names']):
#         if name != return_key:
#             dtype = "integer" if "int" in str(arg.vtype) else "double precision"
#             dim = f"dimension({','.join([str(s) for s in arg.shape])}), " if arg.shape else ""
#             func.add_code_line(f"{dtype}, {dim}intent(in) :: {name}")
#             func.add_linebreak()
#     out_dim = f"({','.join([str(s) for s in return_dim])})" if return_dim else ""
#     func.add_code_line(f"double precision, intent({return_intent}) :: {return_key}{out_dim}")
#     func.add_linebreak()
#
#     func.add_code_line(f"{self._op_dict['value']}" if omit_assign else f"{return_key} = {self._op_dict['value']}")
#     func.add_linebreak()
#     func.add_code_line("end")
#     func.add_linebreak()
#     func.remove_indent()
#     module_counter += 1
#     fn = f"pyrates_func_{module_counter}"
#     f2py.compile(func.generate(), modulename=fn, extension=".f", verbose=False,
#                  source_fn=f"{self.build_dir}/{fn}.f" if self.build_dir else f"{fn}.f")
#     exec(f"from pyrates_func_{module_counter} import {fname}", globals())
#     func_dict[self.short_name] = globals().pop(fname)
#     return func_dict
