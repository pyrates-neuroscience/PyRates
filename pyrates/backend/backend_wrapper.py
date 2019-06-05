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
from copy import deepcopy
from numba import jit, prange
import os
import sys
from importlib import import_module, reload
from shutil import rmtree
import warnings
import timeit

# pyrates internal imports
from .funcs import *

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


#############################################
# basic classes for operators and variables #
#############################################

class NumpyVar(np.ndarray):

    n_vars = 0

    def __new__(cls, vtype: str, backend: Any, name: Optional[str] = None, dtype: Optional[str] = None,
                shape: Optional[tuple] = None, value: Optional[Any] = None):

        # check whether necessary arguments were provided
        if all([arg is None for arg in [shape, value, dtype]]):
            raise ValueError('Either `value` or `shape` and `dtype` need to be provided')

        # set name
        if not name:
            name = f"var_{cls.n_vars}"
            cls.n_vars += 1

        # get shape
        if not shape:
            shape = value.shape if hasattr(value, 'shape') else ()

        # get data type
        if not dtype:
            dtype = value.dtype if hasattr(value, 'dtype') else type(value)
        dtype = dtype.name if hasattr(dtype, 'name') else str(dtype)
        if dtype in backend.dtypes:
            dtype = backend.dtypes[dtype]
        else:
            for dtype_tmp in backend.dtypes:
                if dtype_tmp in dtype:
                    dtype = backend.dtypes[dtype_tmp]
                    break
            else:
                dtype = backend._float_def
                warnings.warn(f'WARNING! Unknown data type of variable {name}: {dtype}. '
                              f'Datatype will be set to default type: {dtype}.')

        # create variable
        value = cls._get_value(value, dtype, shape)
        if vtype == 'constant':
            return value, name
        obj = cls._get_var(value, name, dtype)
        obj.short_name = name.split('/')[-1]
        if not hasattr(obj, 'name'):
            obj.name = name
        else:
            name = obj.name

        return obj, name

    def eval(self):
        return self[:]

    @staticmethod
    def _get_value(value, dtype, shape):
        if value is None:
            return np.zeros(shape=shape, dtype=dtype)
        elif not hasattr(value, 'shape'):
            if type(value) is list:
                return np.zeros(shape=shape, dtype=dtype) + np.asarray(value, dtype=dtype).reshape(shape)
            else:
                return np.zeros(shape=shape, dtype=dtype) + value
        else:
            return value

    @classmethod
    def _get_var(cls, value, name, dtype):
        return np.array(value).view(cls)

    def __deepcopy__(self, memodict={}):
        obj = super().__deepcopy__(memodict)
        if not hasattr(obj, 'name'):
            obj.name = self.name
        if not hasattr(obj, 'short_name'):
            obj.short_name = self.short_name
        return obj


class TensorflowVar(NumpyVar):

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


class PyRatesOp:

    n_consts = 0
    arg_names = {}
    op_names = {}

    def __init__(self, op: str, name: str, decorator: str, *args) -> None:
        """

        Parameters
        ----------
        op
        backend
        args
        """

        self.op = op
        self.short_name = name.split('/')[-1]
        self.name = name
        if self.short_name in self.op_names:
            self.op_names[self.short_name] += 1
            self.short_name = f"{self.short_name}_{self.op_names[self.short_name]}"
        else:
            self.op_names[self.short_name] = 0

        # generate function string
        op_dict = self.generate_op(self.short_name, op, args, decorator)

        # extract information from parsing results
        self.func = op_dict['func']
        self.arg_names = op_dict['arg_names'].copy()
        self.call = op_dict['call']
        self.return_val = op_dict['return_val']
        self.input_ops = op_dict['input_ops']
        self.constant = op_dict['constant']

        # generate function
        func_dict = {}
        exec(op_dict['func'], globals(), func_dict)
        self._callable = func_dict.pop(self.short_name)

        # test function
        self.args = deepcopy(op_dict['args'])
        result = self.eval()
        self.shape = result.shape if hasattr(result, 'shape') else ()
        self.dtype = result.dtype if hasattr(result, 'dtype') else type(result)
        self.args = op_dict['args'].copy()

    def eval(self):
        result = self._callable(*self.args)
        self._check_numerics(result, self.name)
        return result

    @classmethod
    def generate_op(cls, name, op, args, decorator):

        # initialization
        code_gen = CodeGen()
        results = {'func': None, 'args': [], 'arg_names': [], 'constant': False, 'call': None, 'return_val': None,
                   'input_ops': []}

        # setup function head
        #####################

        # begin
        if decorator:
            code_gen.add_code_line(decorator)
            code_gen.add_linebreak()
        code_gen.add_code_line(f"def {name}(")

        # arguments
        code_gen, results, results_args, results_arg_names, n_vars = cls._process_args(code_gen, args, results)

        # end
        code_gen.code[-1] = code_gen.code[-1][:-1]
        code_gen.add_code_line(")")
        results['call'] = code_gen.generate()
        code_gen.add_code_line(":")
        code_gen.add_linebreak()
        code_gen.add_indent()

        # setup return line
        ###################

        code_gen.add_code_line(f"return ")
        return_gen = CodeGen()
        return_gen.add_code_line(f"{op}(")

        # add arguments
        for key, arg in zip(results_arg_names, results_args):
            if type(arg) is str:
                if arg is "[":
                    return_gen.code[-1] = code_gen.code[-1][:-1]
                if arg is "(":
                    return_gen.add_code_line(f"{arg}")
                else:
                    return_gen.add_code_line(f"{arg},")
                idx = results['args'].index(arg)
                results['args'].pop(idx)
                results['arg_names'].pop(idx)
            elif type(arg) is dict:
                return_gen.add_code_line(f"{arg['call']},")
            else:
                return_gen.add_code_line(f"{key},")

        # add function end
        return_gen.code[-1] = return_gen.code[-1][:-1]
        return_gen.add_code_line(")")
        results['return_val'] = return_gen.generate()
        code_gen.add_code_line(results['return_val'])

        # check whether operation arguments contain merely constants
        if n_vars == 0:
            results['constant'] = True

        # generate op
        results['func'] = code_gen.generate()

        return results

    @classmethod
    def _process_args(cls, code_gen, args, results):
        results_args = []
        results_arg_names = []
        n_vars = 0
        for idx, arg in enumerate(args):
            if type(arg) in (PyRatesOp, PyRatesAssignOp, PyRatesIndexOp):
                results['input_ops'].append(arg.short_name)
                pop_indices = []
                for arg_tmp in arg.arg_names:
                    if arg_tmp in results['arg_names']:
                        pop_indices.append(arg.arg_names.index(arg_tmp))
                new_args = arg.args.copy()
                new_arg_names = arg.arg_names.copy()
                for i, pop_idx in enumerate(pop_indices):
                    new_args.pop(pop_idx - i)
                    new_arg_names.pop(pop_idx - i)
                for arg_tmp in new_arg_names:
                    code_gen.add_code_line(f"{arg_tmp},")
                results_args.append({'args': new_args, 'call': arg.return_val,
                                     'arg_names': new_arg_names})
                results_arg_names.append(arg.short_name)
                results['args'] += new_args
                results['arg_names'] += new_arg_names
                n_vars += 1
            elif hasattr(arg, 'short_name'):
                n_vars += 1
                arg_name = cls._generate_unique_argnames([arg.short_name])[0]
                results_args.append(arg)
                results_arg_names.append(arg_name)
                results['args'].append(arg)
                results['arg_names'].append(arg_name)
                code_gen.add_code_line(f"{arg_name},")
            elif type(arg) is tuple or type(arg) is list:
                tuple_begin = '('
                results_args.append(tuple_begin)
                results_arg_names.append(tuple_begin)
                results['args'].append(tuple_begin)
                results['arg_names'].append(tuple_begin)
                code_gen, results, results_args_tmp, results_arg_names_tmp, n_vars_tmp = \
                    cls._process_args(code_gen, arg, results)
                results_args += results_args_tmp
                results_arg_names += results_arg_names_tmp
                n_vars += n_vars_tmp
                tuple_end = ')'
                results_args.append(tuple_end)
                results_arg_names.append(tuple_end)
                results['args'].append(tuple_end)
                results['arg_names'].append(tuple_end)
            else:
                if type(arg) is str:
                    arg_name = cls._generate_unique_argnames([arg])[0]
                else:
                    cls.n_consts += 1
                    arg_name = f"c_{cls.n_consts}"
                    code_gen.add_code_line(f"{arg_name},")
                results_args.append(arg)
                results_arg_names.append(arg_name)
                results['args'].append(arg)
                results['arg_names'].append(arg_name)
        return code_gen, results, results_args, results_arg_names, n_vars

    @staticmethod
    def _check_numerics(vals, name):
        check = []
        try:
            for val in vals:
                try:
                    for v in val:
                        check.append(np.isnan(v) or np.isneginf(v))
                except TypeError:
                    check.append(np.isnan(val) or np.isneginf(val))
        except TypeError:
            check.append(np.isnan(vals) or np.isneginf(vals))
        if any(check):
            raise ValueError(f'Result of operation ({name}) contains NaNs or infinite values.')

    @classmethod
    def _generate_unique_argnames(cls, args: list):
        for i, arg in enumerate(args.copy()):
            if arg in cls.arg_names:
                args[i] = f"{arg}_{cls.arg_names[arg]}"
                cls.arg_names[arg] += 1
            else:
                cls.arg_names[arg] = 0
        return args


class PyRatesAssignOp(PyRatesOp):

    @classmethod
    def generate_op(cls, name, op, args, decorator):

        # initialization
        code_gen = CodeGen()
        results = {'func': None, 'args': [], 'arg_names': [], 'constant': False, 'call': None, 'input_ops': [],
                   'return_val': None}
        results_args = []
        results_arg_names = []

        # extract variables
        var, upd = args[0:2]
        if len(args) > 2:
            if "scatter" in op:
                if hasattr(args[2], 'short_name'):
                    key = cls._generate_unique_argnames([args[2].short_name])[0]
                    if hasattr(args[2], 'return_val'):
                        var_idx = f"{args[2].return_val},"
                    else:
                        var_idx = f"{key},"
                else:
                    key = cls._generate_unique_argnames(["idx"])[0]
                    var_idx = f"{key},"
            elif hasattr(args[2], 'short_name'):
                key = cls._generate_unique_argnames([args[2].short_name])[0]
                if hasattr(args[2], 'return_val'):
                    var_idx = f"[{args[2].return_val}]"
                else:
                    var_idx = f"[{key}]"
            else:
                key = cls._generate_unique_argnames(["idx"])[0]
                var_idx = f"[{key}]"
            results_args.append(args[2])
            results_arg_names.append(key)
        elif hasattr(var, 'shape') and len(var.shape) > 0 and type(var) is NumpyVar:
            var_idx = "[:]"
        else:
            var_idx = ""
        upd_idx = "[:]" if hasattr(upd, 'shape') and len(upd.shape) > 0 and type(var) is NumpyVar else ""

        var_key = cls._generate_unique_argnames([var.short_name])[0]
        results_args.append(var)
        results_arg_names.append(var_key)
        upd_key = cls._generate_unique_argnames([upd.short_name])[0] if hasattr(upd, 'short_name') else \
            cls._generate_unique_argnames(["upd"])[0]
        upd_pos = len(results_args)
        results_args.append(upd)
        results_arg_names.append(upd_key)

        # setup function head
        #####################

        # begin
        if decorator:
            code_gen.add_code_line(decorator)
            code_gen.add_linebreak()
        code_gen.add_code_line(f"def {name}(")

        # add arguments
        for idx, (key, arg) in enumerate(zip(results_arg_names, results_args)):
            if type(arg) in (PyRatesOp, PyRatesAssignOp, PyRatesIndexOp):
                pop_indices = []
                for arg_tmp in arg.arg_names:
                    if arg_tmp in results['arg_names']:
                        pop_indices.append(arg.arg_names.index(arg_tmp))
                    else:
                        code_gen.add_code_line(f"{arg_tmp},")
                new_args = arg.args.copy()
                new_arg_names = arg.arg_names.copy()
                for i, pop_idx in enumerate(pop_indices):
                    new_args.pop(pop_idx - i)
                    new_arg_names.pop(pop_idx - i)
                results_args[idx] = {'args': new_args, 'call': arg.return_val, 'arg_names': new_arg_names}
                results['input_ops'].append(arg.short_name)
                results['args'] += new_args
                results['arg_names'] += new_arg_names
            else:
                results['args'].append(arg)
                results['arg_names'].append(key)
                code_gen.add_code_line(f"{key},")

        # end
        code_gen.code[-1] = code_gen.code[-1][:-1]
        code_gen.add_code_line("):")
        results['call'] = code_gen.generate()
        code_gen.add_linebreak()
        code_gen.add_indent()

        # assign update to variable and return variable
        ###############################################

        upd_str = results_args[upd_pos]['call'] if type(results_args[upd_pos]) is dict else upd_key
        if op in ["=", "+=", "-=", "*=", "/="]:
            code_gen.add_code_line(f"{var_key}{var_idx} {op} {upd_str}{upd_idx}")
        elif "update" in op:
            code_gen.add_code_line(f"{var_key}.{op}({var_idx}{upd_str}{upd_idx})")
        else:
            code_gen.add_code_line(f"{var_key}{var_idx}.{op}({upd_str}{upd_idx})")
        code_gen.add_linebreak()
        code_gen.add_code_line(f"return {var_key}")
        results['return_val'] = var_key

        # generate op
        results['func'] = code_gen.generate()

        return results


class PyRatesIndexOp(PyRatesOp):

    @classmethod
    def generate_op(cls, name, op, args, decorator):

        # initialization
        code_gen = CodeGen()
        results = {'func': None, 'args': [], 'arg_names': [], 'constant': False, 'shape': (), 'dtype': 'float32',
                   'call': None, 'input_ops': [], 'return_val': None}

        # extract variable
        var_tmp = args[0]
        if type(var_tmp) in (PyRatesOp, PyRatesAssignOp, PyRatesIndexOp):
            var = var_tmp.return_val
            pop_indices = []
            for arg_tmp in var_tmp.arg_names:
                if arg_tmp in results['arg_names']:
                    pop_indices.append(var_tmp.arg_names.index(arg_tmp))
            new_args = var_tmp.args.copy()
            new_arg_names = var_tmp.arg_names.copy()
            for i, pop_idx in enumerate(pop_indices):
                new_args.pop(pop_idx - i)
                new_arg_names.pop(pop_idx - i)
            results['input_ops'].append(var_tmp.short_name)
            results['args'] += new_args
            results['arg_names'] += new_arg_names
        elif type(var_tmp) is NumpyVar or issubclass(type(var_tmp), tf.Variable):
            var = cls._generate_unique_argnames([var_tmp.short_name])[0]
            results['args'].append(var_tmp)
            results['arg_names'].append(var)
        else:
            var = f"c_{cls.n_consts}"
            results['args'].append(var_tmp)
            results['arg_names'].append(var)
            cls.n_consts += 1

        # extract idx
        idx = args[1]
        if type(idx) in (PyRatesOp, PyRatesAssignOp, PyRatesIndexOp):
            var_idx = f"[{idx.return_val}]"
            pop_indices = []
            for arg_tmp in idx.arg_names:
                if arg_tmp in results['arg_names']:
                    pop_indices.append(idx.arg_names.index(arg_tmp))
            new_args = idx.args.copy()
            new_arg_names = cls._generate_unique_argnames(idx.arg_names.copy())
            for i, pop_idx in enumerate(pop_indices):
                new_args.pop(pop_idx - i)
                new_arg_names.pop(pop_idx - i)
            results['input_ops'].append(idx.short_name)
            results['args'] += new_args
            results['arg_names'] += new_arg_names
        elif type(idx) is NumpyVar or issubclass(type(idx), tf.Variable):
            key = cls._generate_unique_argnames([idx.short_name])[0]
            var_idx = f"[{key}]"
            results['args'].append(idx)
            results['arg_names'].append(key)
        elif type(idx) is str or "int" in str(type(idx)) or "float" in str(type(idx)):
            for arg in args[2:]:
                if hasattr(arg, 'short_name') and arg.short_name in idx:
                    name_tmp = arg.short_name
                    if name_tmp in cls.arg_names:
                        arg.short_name = f"{name_tmp}_{cls.arg_names[name_tmp]}"
                        idx = idx.replace(name_tmp, arg.short_name)
            var_idx = f"[{idx}]"
        elif type(idx) in (list, tuple):
            var_idx = f"{list(idx)}"
        elif hasattr(idx, 'shape'):
            key = cls._generate_unique_argnames(["idx"])[0]
            var_idx = f"[{key}]"
            results['args'].append(idx)
            results['arg_names'].append(key)
        else:
            raise ValueError(f'Index type not understood: {idx}. Please consider another form of variable indexing.')

        # setup function head
        #####################

        # beginning
        if decorator:
            code_gen.add_code_line(decorator)
            code_gen.add_linebreak()
        code_gen.add_code_line(f"def {name}(")

        # variable and index
        for arg in results['arg_names']:
            code_gen.add_code_line(f"{arg},")

        # remaining arguments
        code_gen, results, results_args, results_arg_names, _ = cls._process_args(code_gen, args[2:], results)

        # end of function head
        code_gen.code[-1] = code_gen.code[-1][:-1]
        code_gen.add_code_line("):")
        results['call'] = code_gen.generate()
        code_gen.add_linebreak()
        code_gen.add_indent()

        # apply index to variable and return variable
        #############################################

        return_line = f"{var}{var_idx}"
        code_gen.add_code_line(f"return {return_line}")
        results['return_val'] = return_line

        # generate op
        results['func'] = code_gen.generate()

        return results


###########################
# backend wrapper classes #
###########################


class NumpyBackend(object):
    """Wrapper to numpy.

    Parameters
    ----------
    ops
        Additional operations this backend instance can perform, defined as key-value pairs.
    dtypes
        Additional data-types this backend instance can use, defined as key-value pairs.

    """

    def __init__(self,
                 ops: Optional[Dict[str, str]] = None,
                 dtypes: Optional[Dict[str, object]] = None,
                 name: str = 'net_0',
                 float_default_type: str = 'float32',
                 jit_compile: bool = False
                 ) -> None:
        """Instantiates tensorflow backend, i.e. a tensorflow graph.
        """

        super().__init__()

        # define operations and datatypes of the backend
        ################################################

        # base math operations
        self.ops = {"+": {'name': "numpy_add", 'call': "np.add"},
                    "-": {'name': "numpy_subtract", 'call': "np.subtract"},
                    "*": {'name': "numpy_multiply", 'call': "np.multiply"},
                    "/": {'name': "numpy_divide", 'call': "np.divide"},
                    "%": {'name': "numpy_modulo", 'call': "np.mod"},
                    "^": {'name': "numpy_power", 'call': "np.power"},
                    "**": {'name': "numpy_power_float", 'call': "np.float_power"},
                    "@": {'name': "numpy_dot", 'call': "np.dot"},
                    ".T": {'name': "numpy_transpose", 'call': "np.transpose"},
                    ".I": {'name': "numpy_invert", 'call': "np.invert"},
                    ">": {'name': "numpy_greater", 'call': "np.greater"},
                    "<": {'name': "numpy_less", 'call': "np.less"},
                    "==": {'name': "numpy_equal", 'call': "np.equal"},
                    "!=": {'name': "numpy_not_equal", 'call': "np.not_equal"},
                    ">=": {'name': "numpy_greater_equal", 'call': "np.greater_equal"},
                    "<=": {'name': "numpy_less_equal", 'call': "np.less_equal"},
                    "=": {'name': "assign", 'call': "="},
                    "+=": {'name': "assign_add", 'call': "+="},
                    "-=": {'name': "assign_subtract", 'call': "-="},
                    "*=": {'name': "assign_multiply", 'call': "*="},
                    "/=": {'name': "assign_divide", 'call': "/="},
                    "neg": {'name': "negative", 'call': "neg_one"},
                    "sin": {'name': "numpy_sin", 'call': "np.sin"},
                    "cos": {'name': "numpy_cos", 'call': "np.cos"},
                    "tan": {'name': "numpy_tan", 'call': "np.tan"},
                    "atan": {'name': "numpy_atan", 'call': "np.arctan"},
                    "abs": {'name': "numpy_abs", 'call': "np.abs"},
                    "sqrt": {'name': "numpy_sqrt", 'call': "np.sqrt"},
                    "sq": {'name': "numpy_square", 'call': "np.square"},
                    "exp": {'name': "numpy_exp", 'call': "np.exp"},
                    "max": {'name': "numpy_max", 'call': "np.max"},
                    "min": {'name': "numpy_min", 'call': "np.min"},
                    "argmax": {'name': "numpy_transpose", 'call': "np.argmax"},
                    "argmin": {'name': "numpy_argmin", 'call': "np.argmin"},
                    "round": {'name': "numpy_round", 'call': "np.round"},
                    "sum": {'name': "numpy_sum", 'call': "np.sum"},
                    "mean": {'name': "numpy_mean", 'call': "np.mean"},
                    "concat": {'name': "numpy_concatenate", 'call': "np.concatenate"},
                    "reshape": {'name': "numpy_reshape", 'call': "np.reshape"},
                    "shape": {'name': "numpy_shape", 'call': "np.shape"},
                    "dtype": {'name': "numpy_dtype", 'call': "np.dtype"},
                    'squeeze': {'name': "numpy_squeeze", 'call': "np.squeeze"},
                    'expand': {'name': 'numpy_expand', 'call': "np.expand_dims"},
                    "roll": {'name': "numpy_roll", 'call': "np.roll"},
                    "cast": {'name': "numpy_cast", 'call': "np.asarray"},
                    "randn": {'name': "numpy_randn", 'call': "np.randn"},
                    "ones": {'name': "numpy_ones", 'call': "np.ones"},
                    "zeros": {'name': "numpy_zeros", 'call': "np.zeros"},
                    "range": {'name': "numpy_arange", 'call': "np.arange"},
                    "softmax": {'name': "pyrates_softmax", 'call': "pr_softmax"},
                    "sigmoid": {'name': "pyrates_sigmoid", 'call': "pr_sigmoid"},
                    "tanh": {'name': "numpy_tanh", 'call': "np.tanh"},
                    "index": {'name': "pyrates_index", 'call': "pyrates_index"},
                    "mask": {'name': "pyrates_mask", 'call': "pr_mask"},
                    "group": {'name': "pyrates_group", 'call': "pr_group"},
                    "asarray": {'name': "numpy_asarray", 'call': "np.asarray"},
                    "no_op": {'name': "pyrates_identity", 'call': "pr_identity"},
                    }
        if ops:
            self.ops.update(ops)

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

        self.vars = dict()
        self.layers = [[]]
        self.var_counter = {}
        self.op_counter = {}
        self.layer = 0
        self.op_indices = {}
        self._build_dir = ""
        self._float_def = self.dtypes[float_default_type]
        self.name = name
        self._rt_optimization = jit_compile
        self._base_layer = 0

    def run(self,
            steps: int,
            outputs: Dict[str, tf.Variable],
            sampling_steps: Optional[int] = None,
            out_dir: Optional[str] = None,
            profile: Optional[str] = None,
            **kwargs
            ) -> Union[Dict[str, tf.Variable], Tuple[dict, float, float]]:
        """Executes all operations in tensorflow graph for a given number of steps.

        Parameters
        ----------
        steps
            Number of graph evaluations.
        inputs
            Inputs fed into the graph.
        outputs
            Variables in the graph to store the history from.
        layers
            Indices of layers to evaluate. If None, all will be evaluated.
        sampling_steps
            Number of graph execution steps to combine into a single output step.
        sampling_layer
            Index of the layer containing sampling ops.
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

        if not sampling_steps:
            sampling_steps = 1

        # initialize session log
        if out_dir:
            # TODO : implement log files
            pass

        # initialize profiler
        if profile is None:
            profile = ''
        if 'm' in profile:
            # TODO: implement memory tracker
            time_and_memory = None
        if 't' in profile:
            t0 = t.time()

        # graph execution
        #################

        # map layers that need to be executed to compiled network structure
        imports = kwargs.pop('imports', ["import numpy as np", "from numba import jit, prange",
                                         "from pyrates.backend.funcs import *"])
        decorator = kwargs.pop('decorator', "")
        layer_run_funcs = self.compile(imports=imports, decorator=decorator)

        sampling_layer = layer_run_funcs.pop(-1)

        # simulate backend behavior for each time-step
        self._run(layer_run_funcs, sampling_layer, steps, sampling_steps)

        # output storage and clean-up
        #############################

        # store output variables in output dictionary
        for key, var in outputs.items():
            outputs[key] = var.numpy() if hasattr(var, 'numpy') else var.eval()

        # store profiling results
        if 't' in profile:
            sim_time = t.time() - t0
        else:
            sim_time = 0.
        if 'm' in profile:
            peak_memory = 'much'
        else:
            peak_memory = 0.

        if profile:
            return outputs, sim_time, peak_memory
        return outputs

    def add_var(self,
                vtype: str,
                name: Optional[str] = None,
                value: Optional[Any] = None,
                shape: Optional[Union[tuple, list, np.shape]] = None,
                dtype: Optional[Union[str, np.dtype]] = None,
                **kwargs
                ) -> NumpyVar:
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
        PyRatesVar
            Handle for the numpy variable.

        """

        # processs input arguments
        ##########################

        # extract variable scope
        scope = kwargs.pop('scope', None)
        if (scope or name):
            if scope:
                name = f'{scope}/{name}'
            if name in self.var_counter:
                name_old = name
                name = f"{name}_{self.var_counter[name]}"
                self.var_counter[name_old] += 1
            else:
                self.var_counter[name] = 1

        # create variable
        #################

        var, name = self._create_var(vtype=vtype, dtype=dtype, shape=shape, value=value, name=name)
        self.vars[name] = var
        return var

    def add_op(self,
               op_name: str,
               *args,
               **kwargs
               ) -> Union[PyRatesOp, NumpyVar]:
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
        Union[PyRatesOp, PyRatesVar]
            Handle for the lambda-numpy function

        """

        # process input arguments
        #########################

        # extract scope
        scope = kwargs.pop('scope', None)

        # extract graph dependencies
        dependencies = kwargs.pop('dependencies', [])

        # extract operator decorator
        decorator = kwargs.pop('decorator', None)

        name = kwargs.pop('name', None)
        if name and scope:
            name = f'{scope}/{name}'
        elif scope:
            name = f'{scope}/assign' if '=' in op_name else f"{scope}/{self.ops[op_name]['name']}"
        if name in self.op_counter:
            name_old = name
            name = f"{name}_{self.op_counter[name]}"
            self.op_counter[name_old] += 1
        else:
            self.op_counter[name] = 0

        if dependencies:
            found = False
            for dep in dependencies:
                if dep.name == name:
                    found = True
                    break
            if found:
                self.add_layer()

        # create operation
        ##################

        # generate op
        try:
            op = self._create_op(op_name, *args, decorator=decorator)
            in_shape = []
            expand_ops = ('@', 'index', 'concat', 'expand', 'stack', 'group', 'asarray')
            if op_name not in expand_ops:
                for arg in args:
                    if hasattr(arg, 'shape'):
                        in_shape.append(sum(tuple(arg.shape)))
                    elif type(arg) in (tuple, list):
                        in_shape.append(sum(arg))
                    else:
                        in_shape.append(1)
                if hasattr(op, 'shape') and (sum(tuple(op.shape)) > max(in_shape)):
                    arg1, arg2 = self.broadcast(args[0], args[1])
                    op = self._create_op(op_name, arg1, arg2, *args[2:], decorator=decorator)
        except (TypeError, ValueError):
            if type(args[0]) in (tuple, list):
                args_tmp = self.broadcast(args[0][0], args[0][1])
                args_tmp = (args_tmp[0], args_tmp[1]) + tuple(args[0][2:])
                args_new = args_tmp + args[1:]
            else:
                args_tmp = self.broadcast(args[0], args[1])
                args_new = args_tmp + args[2:]
            op = self._create_op(op_name, *args_new, decorator=decorator)

        # remove op inputs from layers (need not be evaluated anymore)
        for op_name_tmp in op.input_ops:
            idx_1, idx_2 = self.op_indices[op_name_tmp]
            self._set_op(None, idx_1, idx_2)

        # add op to the graph
        if op.constant:
            new_var = op.eval()
            if hasattr(new_var, 'shape'):
                name = f'{name}_evaluated'
                return self.add_var(vtype='constant', name=name, value=new_var)
            else:
                return new_var

        self._add_op(op)
        self.op_indices[op.short_name] = (self.layer, len(self.get_current_layer())-1)
        return op

    def add_layer(self, to_beginning=False) -> None:
        """Adds a new layer to the backend and sets cursor to that layer.

        Parameters
        ----------

        Returns
        -------
        None

        """

        if to_beginning:
            self.layers = [[]] + self.layers
            self._base_layer += 1
            self.layer = -self._base_layer
        else:
            self.layer = len(self.layers)
            self.layers.append([])

    def next_layer(self):
        if self.layer == len(self.layers)-1:
            self.add_layer()
        else:
            self.layer += 1

    def previous_layer(self):
        if self.layer == -self._base_layer:
            self.add_layer(to_beginning=True)
        else:
            self.layer -= 1

    def remove_layer(self, idx):
        self.layers.pop(self._base_layer + idx)
        if idx <= self._base_layer:
            self._base_layer -= 1

    def top_layer(self):
        self.layer = len(self.layers)-1

    def bottom_layer(self):
        self.layer = -self._base_layer

    def get_current_layer(self):
        return self.layers[self._base_layer + self.layer]

    def clear(self):
        """Resets all tensorflow operations on the graph.
        """
        self.vars.clear()
        self.layers = [[]]
        self.op_counter = 0
        self.var_counter = 0
        self.layer = 0
        rmtree(self._build_dir)

    def get_var(self, name, updated=True):
        if updated:
            return self.vars[name]
        else:
            try:
                return self.vars[f'{name}_old']
            except KeyError:
                return self.vars[name]

    def get_layer(self, idx):
        return self.layers[self._base_layer + idx]

    def eval_var(self, var):
        return self.vars[var].eval()

    def compile(self, build_dir=None, imports=None, decorator=None):

        if not imports:
            imports = []

        # remove empty layers and operators
        new_layer_idx = 0
        for layer_idx, layer in enumerate(self.layers.copy()):
            for op in layer.copy():
                if op is None:
                    layer.pop(layer.index(op))
            if len(layer) == 0:
                self.layers.pop(new_layer_idx)
            else:
                new_layer_idx += 1

        # create directory in which to store layer scripts
        orig_path = os.getcwd()
        if build_dir:
            os.mkdir(build_dir)
        dir_name = f"{build_dir}/pyrates_build" if build_dir else "pyrates_build"
        try:
            os.mkdir(dir_name)
        except FileExistsError:
            pass
        os.chdir(dir_name)
        build_dir = os.getcwd()
        try:
            os.mkdir(self.name)
        except FileExistsError:
            rmtree(self.name)
            os.mkdir(self.name)
        for key in sys.modules.copy():
            if self.name in key:
                del sys.modules[key]
        os.chdir(self.name)
        net_dir = os.getcwd()
        self._build_dir = net_dir

        # write layer operations to files and import the respective functions from file
        for i, layer in enumerate(self.layers):
            l_dir = f"layer_{i}"
            try:
                os.mkdir(l_dir)
            except FileExistsError:
                pass
            os.chdir(l_dir)
            for j, op in enumerate(layer):
                if type(op) is not tuple:
                    op_fname = f"op_{j}"
                    with open(f"{op_fname}.py", 'w') as f:
                        for imp in imports:
                            f.write(f"{imp}\n")
                        f.write(op.func)
                        f.close()
            os.chdir(net_dir)

        # write layer run functions and import them for execution
        sys.path.insert(1, build_dir)
        sys.path.insert(2, net_dir)
        with open(f"__init__.py", 'w') as net_init:
            net_init.close()
        net_module = import_module(self.name)
        layer_runs = []
        for i, layer in enumerate(self.layers):
            layer_ops = []
            os.chdir(f"layer_{i}")
            with open(f"__init__.py", 'w') as layer_init:
                for j in range(len(layer)):
                    layer_init.write(f"from .op_{j} import *\n")
                layer_init.close()
            os.chdir(net_dir)
            t.sleep(0.1)
            reload(net_module)
            for j, op in enumerate(layer):
                op_module = import_module(f".op_{j}", package=f"{self.name}.layer_{i}")
                layer_ops.append(getattr(op_module, op.short_name))
            run_func, run_args = self._generate_layer_run(layer, i, layer_ops, decorator=decorator, imports=imports)
            with open(f"layer_{i}_run.py", 'w') as f_run:
                f_run.write(run_func)
                f_run.close()
            with open(f"__init__.py", 'w') as net_init:
                for j in range(i+1):
                    net_init.write(f"from .layer_{j} import *\n")
                    net_init.write(f"from .layer_{j}_run import *\n")
                net_init.close()
            t.sleep(0.1)
            reload(net_module)
            layer_runs.append((getattr(net_module, f"run_l{i}"), run_args))
        sys.path.pop(2)
        sys.path.pop(1)
        os.chdir(orig_path)

        return layer_runs

    def eval(self, ops):
        return [op.eval() for op in ops]

    def broadcast(self, op1: Any, op2: Any, **kwargs) -> tuple:
        """Tries to match the shapes of op1 and op2 such that op can be applied. Then applies op to op1 and op2.

        Parameters
        ----------
        op1
            First argument to the operation.
        op2
            Second argument to the operation.
        return_ops
            If true, the adjusted arguments (op1 and op2) are returned.
        kwargs
            Additional keyword arguments to be passed to the backend.

        Returns
        -------
        tp.Union[tuple, tp.Any]
            Output of op applied to op1 and op2. If return_ops, op1 and op2 are also returned.

        """

        # try to match shapes
        if not self._compare_shapes(op1, op2):

            # try removing singleton dimensions from op1/op2
            op1_tmp, op2_tmp = self._match_shapes(op1, op2, adjust_second=True)
            if not self._compare_shapes(op1_tmp, op2_tmp):
                op1_tmp, op2_tmp = self._match_shapes(op1_tmp, op2_tmp, adjust_second=False)
            if self._compare_shapes(op1_tmp, op2_tmp):
                op1, op2 = op1_tmp, op2_tmp

        return op1, op2

    def apply_idx(self, var: Any, idx: str, update: Optional[Any] = None, *args):
        if update:
            return self.add_op('=', var, update, idx, *args)
        else:
            return self.add_op('index', var, idx, *args)

    def stack_vars(self, *vars, **kwargs):
        return self.add_op('asarray', vars)

    @staticmethod
    def eval_layer(layer):
        return [func(*args) for func, args in layer]

    def _set_op(self, op, idx1, idx2):
        self.layers[self._base_layer+idx1][idx2] = op

    def _add_op(self, op, layer=None):
        if layer is None:
            self.get_current_layer().append(op)
        else:
            self.get_layer(layer).append(op)

    def _run(self, layers, sampling_layer, steps, sampling_steps):
        if sampling_layer is None:
            for step in range(steps):
                for func, args in layers:
                    func(*args)
        else:
            sampling_func, sampling_args = sampling_layer
            for step in range(steps):
                if step % sampling_steps == 0:
                    sampling_func(*sampling_args)
                for func, args in layers:
                    func(*args)

    # def _run_inp(self, layers, sampling_layer, inputs, steps, sampling_steps):
    #     if sampling_layer is None:
    #         for step, inp in zip(range(steps), inputs):
    #             for key, val in inp.items():
    #                 self.vars[key][:] = val
    #             for func, args in layers:
    #                 func(*args)
    #     else:
    #         for step, inp in zip(range(steps), inputs):
    #             sampling_func, sampling_args = sampling_layer
    #             for key, val in inp.items():
    #                 self.vars[key][:] = val
    #             if step % sampling_steps == 0:
    #                 sampling_func(*sampling_args)
    #             for func, args in layers:
    #                 func(*args)

    def _match_shapes(self, op1: Any, op2: Any, adjust_second: bool = True) -> tuple:
        """Re-shapes op1 and op2 such that they can be combined via mathematical operations.

        Parameters
        ----------
        op1
            First operator.
        op2
            Second operator.
        adjust_second
            If true, the second operator will be re-shaped according to the shape of the first operator. If false,
            it will be done the other way around.

        Returns
        -------
        tuple
            The re-shaped operators.

        """

        if not hasattr(op1, 'shape'):
            op1 = self.add_var(vtype='constant', value=op1)
        if not hasattr(op2, 'shape'):
            op2 = self.add_var(vtype='constant', value=op2)

        if adjust_second:
            op_adjust = op2
            op_target = op1
        else:
            op_adjust = op1
            op_target = op2

        if len(op_target.shape) > len(op_adjust.shape) and 1 in op_target.shape:

            # add axis to op_adjust
            target_shape = op_target.shape
            idx = list(target_shape).index(1)
            op_adjust = self.add_op('expand', op_adjust, idx)

        elif (len(op_adjust.shape) > len(op_target.shape) and 1 in op_adjust.shape) or \
                (len(op_target.shape) == 2 and len(op_adjust.shape) == 2 and op_target.shape[1] != op_adjust.shape[0]
                 and 1 in op_adjust.shape):

            # cut singleton dimension from op_adjust
            old_shape = list(op_adjust.shape)
            idx = old_shape.index(1)
            op_adjust = self.add_op('squeeze', op_adjust, idx)

        if adjust_second:
            return op_target, op_adjust
        return op_adjust, op_target

    def _create_var(self, vtype, dtype, shape, value, name):
        return NumpyVar(vtype=vtype, dtype=dtype, shape=shape, value=value, name=name, backend=self)

    def _create_op(self, op, *args, decorator=None):
        if op in ["=", "+=", "-=", "*=", "/="]:
            if decorator is None and self._rt_optimization:
                decorator = self._optimize_decorator(op, args)
            return PyRatesAssignOp(self.ops[op]['call'], self.ops[op]['name'], decorator, *args)
        elif op is "index":
            return PyRatesIndexOp(self.ops[op]['call'], self.ops[op]['name'], "", *args)
        else:
            if op is "cast":
                args = list(args)
                for dtype in self.dtypes:
                    if dtype in str(args[1]):
                        args[1] = self.dtypes[dtype]
                        break
                args = tuple(args)
            return PyRatesOp(self.ops[op]['call'], self.ops[op]['name'], "", *args)

    def _optimize_decorator(self, op, args):
        decorators = ["", "@jit", "@jit(fastmath=True)" , "@jit(nogil=True)",
                      "@jit(nogil=True, fastmath=True)", "@jit(nopython=True)", "@jit(nopython=True, parallel=True)",
                      "@jit(nopython=True, fastmath=True)", "@jit(nopython=True, fastmath=True, parallel=True)",
                      "@jit(nopython=True, fastmath=True, parallel=True, nogil=True)",
                      "@jit(nopython=True, fastmath=True, nogil=True)", "@jit(nopython=True, nogil=True)"]
        decorator_times = [np.infty for _ in range(len(decorators))]
        args = deepcopy(args)
        for i, dec in enumerate(decorators):
            try:
                operator = self._create_op(op, *args, decorator=dec)
                operator.eval()
                if hasattr(operator, 'call'):
                    decorator_times[i] = timeit.timeit(lambda: operator.eval(), number=10)
                else:
                    decorator_times[i] = 0.
                    break
            except Exception:
                continue

        return decorators[np.argmin(decorator_times)]

    def _generate_layer_run(self, layer, idx, func_calls, decorator, imports):

        # set up file head (imports)
        code_gen = CodeGen()
        for imp in imports:
            code_gen.add_code_line(imp)
            code_gen.add_linebreak()
        code_gen.add_code_line(f"n_ops = {len(layer)}")
        code_gen.add_linebreak()

        # set up function head
        if decorator:
            code_gen.add_code_line(decorator)
            code_gen.add_linebreak()
        code_gen.add_code_line(f"def run_l{idx}(")
        op_args_str, op_args, op_names = [], [], []
        for i, (op, func) in enumerate(zip(layer, func_calls)):
            code_gen.add_code_line(f"{op.short_name},")
            arg_str = op.call.split('(')[-1]
            arg_str = arg_str.split(')')[0]
            code_gen.add_code_line(f"{arg_str},")
            op_names.append(op.short_name)
            op_args_str.append(arg_str)
            op_args.append(func)
            op_args += op.args
        code_gen.code[-1] = code_gen.code[-1][:-1]
        code_gen.add_code_line('):')
        code_gen.add_linebreak()
        code_gen.add_indent()

        # set up function body
        code_gen = self._generate_layer_run_body(code_gen, op_names, op_args_str)

        return code_gen.generate(), tuple(op_args)

    @staticmethod
    def _generate_layer_run_body(code_gen, ops: List[str], op_args: List[str]):
        for op, args in zip(ops, op_args):
            code_gen.add_code_line(f"{op}({args})")
            code_gen.add_linebreak()
        return code_gen

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
            elif len(op1.shape) == 0 or len(op2.shape) == 0:
                return True
            else:
                return False
        else:
            return True


class TensorflowBackend(NumpyBackend):
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
                 name: str = 'net_0',
                 float_default_type: str = 'float32',
                 use_device: str = 'cpu'
                 ) -> None:
        """Instantiates tensorflow backend, i.e. a tensorflow graph.
        """

        if use_device == 'cpu':
            device = '/cpu:0'
        elif use_device == 'gpu':
            device = '/gpu:0'
        else:
            device = use_device
        self.default_device = device

        super().__init__(ops, dtypes, name, float_default_type)

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

    def run(self,
            steps: int,
            outputs: Dict[str, tf.Variable],
            sampling_steps: Optional[int] = None,
            out_dir: Optional[str] = None,
            profile: Optional[str] = None,
            **kwargs
            ) -> Union[Dict[str, tf.Variable], Tuple[dict, float, float]]:
        imports = kwargs.pop('imports', ["import tensorflow as tf", "from pyrates.backend.funcs import *"])
        kwargs['imports'] = imports
        decorator = kwargs.pop('decorator', "@tf.function")
        kwargs['decorator'] = decorator
        return super().run(steps, outputs, sampling_steps, out_dir, profile, **kwargs)

    def broadcast(self, op1: Any, op2: Any, **kwargs) -> tuple:
        """Tries to match the shapes of op1 and op2 such that op can be applied. Then applies op to op1 and op2.

        Parameters
        ----------
        op1
            First argument to the operation.
        op2
            Second argument to the operation.
        return_ops
            If true, the adjusted arguments (op1 and op2) are returned.
        kwargs
            Additional keyword arguments to be passed to the backend.

        Returns
        -------
        tp.Union[tuple, tp.Any]
            Output of op applied to op1 and op2. If return_ops, op1 and op2 are also returned.

        """

        # match data types
        if not self._compare_dtypes(op1, op2):
            op1, op2 = self._match_dtypes(op1, op2)

        return super().broadcast(op1, op2, **kwargs)

    def stack_vars(self, *vars, **kwargs):
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
        if not decorator:
            decorator = "@tf.function"
        if op in ["=", "+=", "-=", "*=", "/="]:
            if len(args) > 2 and hasattr(args[2], 'shape'):
                if op == "=":
                    op = "update"
                elif op == "+=":
                    op = "update_add"
                else:
                    op = "update_sub"
                args = self._process_update_args(*args)
            return PyRatesAssignOp(self.ops[op]['call'], self.ops[op]['name'], decorator, *args)
        if op is "index":
            if hasattr(args[1], 'dtype') and 'bool' in str(args[1].dtype):
                return PyRatesOp(self.ops['mask']['call'], self.ops['mask']['name'], decorator, *args)
            if hasattr(args[1], 'shape') or type(args[1]) in (list, tuple):
                try:
                    return PyRatesOp(self.ops['gather']['call'], self.ops['gather']['name'], decorator, *args)
                except (ValueError, IndexError):
                    args = self._process_idx_args(*args)
                    return PyRatesOp(self.ops['gather_nd']['call'], self.ops['gather_nd']['name'], decorator, *args)
            return PyRatesIndexOp(self.ops[op]['call'], self.ops[op]['name'], decorator, *args)
        if op is "cast":
            args = list(args)
            for dtype in self.dtypes:
                if dtype in str(args[1]):
                    args[1] = self.dtypes[dtype]
                    break
            args = tuple(args)
        return PyRatesOp(self.ops[op]['call'], self.ops[op]['name'], decorator, *args)

    def _process_update_args(self, var, update, idx):
        """Preprocesses the index and a variable update to match the variable shape.

        Parameters
        ----------
        var
        update
        idx

        Returns
        -------
        tuple
            Preprocessed index and re-shaped update.

        """

        # pre-process args
        ###################

        shape = var.shape

        # match shape of index and update to shape
        ##########################################

        if idx.shape[0] != update.shape[0]:
            if update.shape[0] == 1 and len(update.shape) == 1:
                idx = self.add_op('reshape', idx, tuple(idx.shape) + (1,))
            elif idx.shape[0] == 1:
                update = self.add_op('reshape', update, (1,) + tuple(update.shape))
            else:
                raise ValueError(f'Invalid indexing. Operation of shape {shape} cannot be updated with updates of '
                                 f'shapes {update.shape} at locations indicated by indices of shape {idx.shape}.')

        if len(idx.shape) < 2:
            idx = self.add_op('reshape', idx, tuple(idx.shape) + (1,))

        shape_diff = len(shape) - idx.shape[1]
        if shape_diff < 0:
            raise ValueError(f'Invalid index shape. Operation has shape {shape}, but received an index of '
                             f'length {idx.shape[1]}')

        # manage shape of updates
        update_dim = 0
        if len(update.shape) > 1:
            update_dim = len(update.shape[1:])
            if update_dim > shape_diff:
                singleton = -1 if update.shape[-1] == 1 else list(update.shape).index(1)
                update = self.add_op('squeeze', update, singleton)
                if len(update.shape) > 1:
                    update_dim = len(update.shape[1:])

        # manage shape of idx
        shape_diff = int(shape_diff) - update_dim
        if shape_diff > 0:
            indices = []
            indices.append(idx)
            for i in range(shape_diff):
                indices.append(self.add_op('zeros', tuple(idx.shape), idx.dtype))
            idx = self.add_op('concat', indices, 1)

        return var, update, idx

    def _process_idx_args(self, var, idx):
        """Preprocesses the index to a variable.

        Parameters
        ----------
        var
        idx

        Returns
        -------
        tuple
            Variable and preprocessed index.

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
        """

        Parameters
        ----------
        op1
        op2

        Returns
        -------

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

    @staticmethod
    def _compare_shapes(op1: Any, op2: Any) -> bool:
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


##############################################
# code generator class for backend functions #
##############################################

class CodeGen:
    """Generates code
    """

    def __init__(self):
        self.code = []
        self.lvl = 0

    def generate(self):
        return "".join(self.code)

    def add_code_line(self, code_str):
        self.code.append("    " * self.lvl + code_str)

    def add_linebreak(self):
        self.code.append("\n")

    def add_indent(self):
        self.lvl += 1

    def remove_indent(self):
        if self.lvl == 0:
            raise(SyntaxError("internal error in code generator"))
        self.lvl -= 1
