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
from .funcs import *
from .parser import CodeGen, var_in_expression, extract_var
from .computegraph import ComputeGraph

# external imports
from typing import Optional, Dict, List, Union, Any
import os
import sys
from shutil import rmtree
import warnings
import numpy as np


# Helper Functions
##################


def sort_equations(lhs_vars: list, rhs_expressions: list) -> tuple:

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
    # stuck = False
    # while lhs_vars_old:
    #     v_tmp, indexed = extract_var(lhs_vars[0])
    #     found = False
    #     for idx, (lhs_var, expr_tmp) in enumerate(zip(lhs_vars, rhs_expressions)):
    #         var, indexed = extract_var(lhs_var)
    #         if var_in_expression(v_tmp, expr_tmp):
    #             if not var_in_expression(v_tmp, lhs_var):
    #                 found = True
    #                 break
    #             elif stuck:
    #                 indexed = True
    #                 found = True
    #                 break
    #         elif var_in_expression(v_tmp, lhs_var) and (stuck or indexed):
    #             found = True
    #             indexed = True
    #             break
    #     if found:
    #         lhs_var = lhs_vars_old.pop(idx)
    #         vars_new.append(lhs_var)
    #         expr = expressions_old.pop(idx)
    #         expressions_new.append(expr)
    #         if not indexed and var not in defined_vars:
    #             defined_vars.append(var)
    #         stuck = False
    #     elif stuck:
    #         raise ValueError('Unable to find a well-defined order of the non-differential equations in the system.')
    #     else:
    #         stuck = True
    #
    #     lhs_vars = lhs_vars_old
    #     rhs_expressions = expressions_old

    return vars_new[::-1], expressions_new[::-1], [v for v in all_vars if v not in defined_vars], defined_vars


#################################
# classes for backend variables #
#################################


class BaseVar(np.ndarray):
    """Base class for adding variables to the PyRates compute graph. Creates a numpy array with additional attributes
    for variable identification/retrieval from graph. Should be used as parent class for custom variable classes.

    Parameters
    ----------
    vtype
        Type of the variable. Can be either `constant` or `state_variable`. Constant variables are necessary to perform
        certain graph optimizations previous to run time.
    dtype
        Data-type of the variable. For valid data-types, check the documentation of the backend in use.
    shape
        Shape of the variable.
    value
        Value of the variable. If scalar, please provide the shape in addition.
    name
        Full name of the variable (including the node and operator it belongs to).
    short_name
        Name of the variable excluding its node and operator.

    Returns
    -------
    BaseVar
        Instance of BaseVar.
    """

    def __new__(cls, vtype: str, backend: Any, name: str, dtype: Optional[str] = None,
                shape: Optional[tuple] = None, value: Optional[Any] = None, squeeze: bool = True):
        """Creates new instance of BaseVar.
        """

        # check whether necessary arguments were provided
        if all([arg is None for arg in [shape, value, dtype]]):
            raise ValueError('Either `value` or `shape` and `dtype` need to be provided')

        # get shape
        if not shape:
            shape = value.shape if hasattr(value, 'shape') else np.shape(value)

        # get data type
        if not dtype:
            if hasattr(value, 'dtype'):
                dtype = value.dtype
            else:
                try:
                    dtype = np.dtype(value)
                except TypeError:
                    dtype = type(value)
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
        if vtype == 'state_var' and 1 in tuple(shape) and squeeze:
            idx = tuple(shape).index(1)
            shape = list(shape)
            shape.pop(idx)
            shape = tuple(shape)
        if callable(value):
            obj = value
        else:
            try:
                # normal variable
                value = cls._get_value(value, dtype, shape)
                if squeeze:
                    value = cls.squeeze(value)
                obj = cls._get_var(value, name, dtype)
            except TypeError:
                # list of callables
                obj = cls._get_var(value, name, dtype)

        # store additional attributes on variable object
        obj.short_name = name.split('/')[-1]
        if not hasattr(obj, 'name'):
            obj.name = name
        else:
            name = obj.name
        obj.vtype = vtype

        return obj, name

    def numpy(self):
        """Returns current value of BaseVar.
        """
        try:
            return self[:]
        except IndexError:
            return self

    def reshape(self, shape: tuple, **kwargs):

        obj = super().reshape(shape, **kwargs)
        if hasattr(self, 'name'):
            obj.name = self.name
        if hasattr(self, 'short_name'):
            obj.short_name = self.short_name
        if hasattr(self, 'vtype'):
            obj.vtype = self.vtype
        return obj

    @staticmethod
    def _get_value(value, dtype, shape):
        """Defines initial value of variable.
        """
        if value is None:
            return np.zeros(shape=shape, dtype=dtype)
        elif not hasattr(value, 'shape'):
            if type(value) is list:
                return np.asarray(value, dtype=dtype).reshape(shape)
            else:
                return np.zeros(shape=shape, dtype=dtype) + value
        elif shape:
            if value.shape == shape:
                return value
            elif sum(shape) < sum(value.shape):
                return value.squeeze()
            else:
                idx = ",".join("None" if s == 1 else ":" for s in shape)
                return eval(f'value[{idx}]')
        else:
            return np.asarray(value, dtype=dtype)

    @staticmethod
    def squeeze(var, axis=None):
        return var.squeeze(axis)

    def _is_equal_to(self, v: np.ndarray):
        if self.short_name == v.short_name and self.shape == v.shape:
            if sum(v.shape) > 1:
                return all(self.flatten() != v.flatten())
            else:
                return self != v

    @classmethod
    def _get_var(cls, value, name, dtype):
        """Creates new numpy array from BaseVar.
        """
        return np.array(value).view(cls)

    def __deepcopy__(self, memodict={}):
        obj = super().__deepcopy__(memodict)
        if not hasattr(obj, 'name') and hasattr(self, 'name'):
            obj.name = self.name
        if not hasattr(obj, 'short_name') and hasattr(self, 'short_name'):
            obj.short_name = self.short_name
        if not hasattr(obj, 'vtype') and hasattr(self, 'vtype'):
            obj.vtype = self.vtype
        return obj

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(str(self))

    @staticmethod
    def __subclasscheck__(subclass):
        if np.ndarray.__subclasscheck__(subclass):
            return True
        else:
            return interp1d.__subclasscheck__(subclass)


#######################################
# classes for backend functionalities #
#######################################


class BaseBackend(object):
    """

    """

    create_backend_var = BaseVar

    def __init__(self,
                 ops: Optional[Dict[str, str]] = None,
                 dtypes: Optional[Dict[str, object]] = None,
                 name: str = 'net_0',
                 float_default_type: str = 'float32',
                 imports: Optional[List[str]] = None,
                 build_dir: str = None,
                 compute_graph: ComputeGraph = None,
                 **kwargs
                 ) -> None:
        """Instantiates the standard, numpy-based backend, i.e. a compute graph with numpy operations.
        """

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

        # initialize compute graph
        self.graph = compute_graph if compute_graph else ComputeGraph(**kwargs)

        # further attributes
        self._var_map = dict()
        self._float_def = self.dtypes[float_default_type]
        self.name = name
        self._imports = ["from numpy import *", "from pyrates.backend.funcs import *"]
        if imports:
            for imp in imports:
                if imp not in self._imports:
                    self._imports.append(imp)
        self._input_names = []
        self.type = 'numpy'
        self._orig_dir = None
        self._build_dir = None
        self._func_name = None
        self._file_name = None

        # create build dir
        self._orig_dir = os.getcwd()
        self._build_dir = f"{build_dir}/{self.name}" if build_dir else ""
        if build_dir:
            os.makedirs(build_dir, exist_ok=True)
            try:
                os.mkdir(self._build_dir)
            except FileExistsError:
                rmtree(self._build_dir)
                os.mkdir(self._build_dir)
            sys.path.append(self._build_dir)
        else:
            sys.path.append(self._orig_dir)

    def add_var(self,
                name: str,
                vtype: str,
                value: Optional[Any] = None,
                shape: Optional[Union[tuple, list, np.shape]] = None,
                dtype: Optional[Union[str, np.dtype]] = None,
                **kwargs
                ) -> tuple:
        """Adds a variable to the backend.

        Parameters
        ----------
        vtype
            Variable type. Can be
                - `state_var` for variables that can change over time.
                - `constant` for non-changing variables.
        name
            Name of the variable.
        value
            Value of the variable. Not needed for placeholders.
        shape
            Shape of the variable.
        dtype
            Datatype of the variable.
        kwargs
            Additional keyword arguments passed to `computegraph.ComputeGraph.add_var`.

        Returns
        -------
        tuple
            (1) variable name, (2) dictionary with all variable information.

        """

        # extract variable scope
        scope = kwargs.pop('scope', None)
        if scope:
            label = f'{scope}/{name}'
        else:
            label = name

        # if variable already exists, return it
        try:
            return name, self.get_var(label, get_key=False)
        except KeyError:
            pass

        # create variable
        var, label = self.create_backend_var(vtype=vtype, dtype=dtype, shape=shape, value=value, name=label,
                                             backend=self, squeeze=kwargs.pop('squeeze', True))

        # add variable to compute graph
        name, cg_var = self.graph.add_var(label=name, value=var, vtype=vtype, **kwargs)

        # save to dict
        self._var_map[label] = name

        return name, cg_var

    def add_op(self,
               inputs: Union[list, tuple],
               name: str,
               **kwargs
               ) -> tuple:
        """Add operation to the backend.

        Parameters
        ----------
        inputs
            List with the names of all compute graph nodes that should enter as input in this operation.
        name
            Key of the operation. If it is a key of `backend.ops`, the function call at `backend.ops[name]['func']` will
             be used.
        kwargs
            Additional keyword arguments passed to `computegraph.ComputeGraph.add_op`.

        Returns
        -------
        tuple
            (1) The key for extracting the operator from the compute graph,
            (2) the dictionary of the compute graph operator node.

        """

        # extract operator scope
        scope = kwargs.pop('scope', None)
        if scope:
            name = f'{scope}/{name}'

        # add operator to compute graph
        return self.graph.add_op(inputs, label=name, **kwargs)

    def get_var(self, name: str, get_key: bool = False, **kwargs) -> Union[dict, str]:
        """Retrieve variable from graph.

        Parameters
        ----------
        name
            Identifier of the variable.
        get_key
            If true, only the name of the variable in the compute graph is returned.

        Returns
        -------
        Union[dict, str]
            Variable dictionary from graph.

        """

        # extract operator scope
        scope = kwargs.pop('scope', None)
        label = f'{scope}/{name}' if scope else name
        try:
            return self._var_map[label] if get_key else self.graph.nodes[self._var_map[label]]
        except KeyError:
            if get_key:
                return name
            return self.graph.nodes[name]

    def run(self, T: float, dt: float, dts: float = None, outputs: dict = None, solver: str = None,
            in_place: bool = True, func_name: str = None, file_name: str = None, compile_kwargs: dict = None, **kwargs
            ) -> dict:

        # preparations
        ##############

        if not compile_kwargs:
            compile_kwargs = dict()
        if not func_name and solver:
            func_name = 'rhs_func'
        if not file_name and solver:
            file_name = 'pyrates_rhs_func'
        if not dts:
            dts = dt
        self._func_name = func_name
        self._file_name = file_name

        # network specs
        run_info = self._compile(in_place=in_place, func_name=func_name, file_name=file_name, **compile_kwargs)
        state_vec = self.get_var(run_info['state_vec'])['value']
        rhs = self.ops['cast']['func'](self.ops['zeros']['func'](shape=tuple(state_vec.shape)))

        # simulation specs
        steps = int(np.round(T / dt))
        store_steps = int(np.round(T / dts))
        state_rec = np.zeros((store_steps, state_vec.shape[0]))
        times = np.arange(0, T, dts) if dts else np.arange(0, T, dt)

        # perform simulation
        ####################

        if solver:

            # numerical integration via a system update function generated from the compute graph
            rhs_func = run_info['func']
            func_args = run_info['func_args'][3:]
            args = tuple([self.get_var(arg)['value'] for arg in func_args])

            if solver == 'euler':

                # perform integration via standard Euler method
                store_step = int(np.round(dts/dt))
                state_rec = self._euler_integration(steps, store_step, state_vec, state_rec, dt, rhs_func, rhs, *args)

            else:

                from scipy.integrate import solve_ivp

                # solve via scipy's ode integration function
                fun = lambda t, y: rhs_func(t, y, rhs, *args)
                t = 0.0
                kwargs['t_eval'] = times
                results = solve_ivp(fun=fun, t_span=(t, T), y0=state_vec, first_step=dt, **kwargs)
                state_rec = results['y'].T

        else:

            # numerical integration by directly traversing the compute graph at each integration step
            # TODO: add mathematical operation to graph that increases time in steps of dt
            store_step = int(np.round(dts / dt))
            idx = 0
            for step in range(steps):
                if step % store_step == 0:
                    state_rec[idx, :] = state_vec
                    idx += 1
                state_vec += dt * self.execute_graph()

        # reduce state recordings to requested state variables
        final_results = {}
        for key, var in outputs.items():
            idx = run_info['old_state_vars'].index(var)
            idx = run_info['vec_indices'][idx]
            if type(idx) is tuple and idx[1] - idx[0] == 1:
                idx = (idx[0],)
            elif type(idx) is int:
                idx = (idx,)
            final_results[key] = state_rec[:, idx] if len(idx) == 1 else state_rec[:, idx[0]:idx[1]]
        final_results['time'] = times

        return final_results

    def execute_graph(self):
        """Executes all computations inside the compute graph once.
        """
        for v in self.non_state_vars:
            self.graph.eval_node(self.get_var(v, get_key=True))
        return np.concatenate([self.graph.eval_node(self.get_var(v, get_key=True)) for v in self.state_vars])

    def clear(self) -> None:
        """Deletes build directory and removes all compute graph nodes
        """

        # delete compute graph nodes
        nodes = [n for n in self.graph.nodes]
        for n in nodes:
            self.graph.remove_subgraph(n)
        self._var_map.clear()

        # remove files and directories that have been created during simulation process
        if self._build_dir:
            rmtree(f"{self._orig_dir}/{self._build_dir}")
        else:
            try:
                os.remove(f"{self._orig_dir}/{self._file_name}.py")
            except FileNotFoundError:
                pass
        if self._func_name in sys.modules:
            del sys.modules[self._func_name]
        if self._file_name in sys.modules:
            del sys.modules[self._file_name]

    def _compile(self, in_place: bool = True, func_name: str = None, file_name: str = None, **kwargs) -> dict:

        # finalize compute graph
        self.graph = self.graph.compile(in_place=in_place)

        # create state variable vector and state variable update vector
        ###############################################################

        vars, indices, updates, old_state_vars = [], [], [], []
        idx = 0
        for var, update in self.graph.var_updates['DEs'].items():

            # extract left-hand side and right-hand side nodes from graph
            lhs, rhs, shape = self._process_var_update(var, update)
            vars.append(lhs), updates.append(rhs)

            # store information of the original, non-vectorized state variable
            old_state_vars.append(var)
            indices.append((idx, idx+shape) if shape > 1 else idx)
            idx += shape

        # add vectorized state variables and updates to the backend
        state_vec = self.ops['concat']['func'](vars, axis=0)
        rhs_vec = self.ops['concat']['func'](updates, axis=0)
        state_var_key, state_var = self.add_var(vtype='state_var', name='state_vec', value=state_vec, squeeze=False)
        rhs_var_key, rhs_var = self.add_var(vtype='state_var', name='state_vec_update', value=np.zeros_like(rhs_vec))

        return_dict = {'old_state_vars': old_state_vars, 'state_vec': state_var_key, 'vec_indices': indices}

        if func_name or file_name:

            if not func_name:
                func_name = 'rhs_func'

            code_gen = CodeGen()

            # generate function body with all equations and assignments
            func_args, code_gen_tmp = self._graph_to_str(rhs_indices=indices, state_var=state_var_key,
                                                         rhs_var=rhs_var_key)
            func_body = code_gen_tmp.generate()

            # generate function head
            func_args, code_gen = self._generate_func_head(code_gen=code_gen, state_var=state_var_key,
                                                           func_args=func_args, func_name=func_name,
                                                           imports=self._imports)

            # add lines from function body after function head
            code_gen.add_code_line(func_body)

            # generate function tail
            code_gen = self._generate_func_tail(code_gen=code_gen, rhs_var=rhs_var_key)

            # finalize function string
            func_str = code_gen.generate()

            # write function string to file
            func = self._generate_func(func_name=func_name, file_name=file_name, func_str=func_str,
                                       build_dir=self._build_dir, **kwargs)
            return_dict['func'] = func
            return_dict['func_args'] = func_args

        return return_dict

    def _graph_to_str(self, code_gen: CodeGen = None, rhs_indices: list = None, state_var: str = None,
                      rhs_var: str = None):

        # preparations
        if code_gen is None:
            code_gen = CodeGen()
        backend_funcs = {key: val['str'] for key, val in self.ops.items()}
        code_gen.add_linebreak()

        # extract state variable from state vector if necessary
        rhs_indices_str = []
        if rhs_indices:

            for idx, var in zip(rhs_indices, self.graph.var_updates['DEs']):
                idx_str = f'{idx}' if type(idx) is int else f'{idx[0]}:{idx[1]}'
                code_gen.add_code_line(f"{var} = {state_var}[{idx_str}]")
                rhs_indices_str.append(idx_str)

            code_gen.add_linebreak()

        # get equation string and argument list for each non-DE node at the end of the compute graph hierarchy
        func_args2, delete_args1, code_gen = self._generate_update_equations(code_gen,
                                                                             nodes=self.graph.var_updates['non-DEs'],
                                                                             funcs=backend_funcs)

        # get equation string and argument list for each DE node at the end of the compute graph hierarchy
        func_args1, delete_args2, code_gen = self._generate_update_equations(code_gen,
                                                                             nodes=self.graph.var_updates['DEs'],
                                                                             rhs_var=rhs_var, indices=rhs_indices_str,
                                                                             funcs=backend_funcs)

        # remove unnecessary function arguments
        func_args = func_args1 + func_args2
        for arg in delete_args1 + delete_args2:
            while arg in func_args:
                func_args.pop(func_args.index(arg))

        return func_args, code_gen

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
                    code_gen.add_code_line(f"{rhs_var}[{idx}] = {expr}")

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
                    code_gen.add_code_line(f"{target_var}[{idx}] = {expr}")

            else:

                # non-DE update stored in a single variable
                for target_var, expr in zip(var_names, expressions):
                    code_gen.add_code_line(f"{target_var} = {expr}")

        code_gen.add_linebreak()

        return func_args, defined_vars, code_gen

    def _process_var_update(self, var: str, update: str) -> tuple:

        # extract nodes
        var_info = self.get_var(var)
        update_info = self.get_var(update)

        # extract common shape
        lhs = var_info['value']
        if not lhs.shape:
            lhs = np.reshape(lhs, (1,))
        rhs = self.graph.eval_node(update)
        if not rhs.shape:
            rhs = np.reshape(rhs, (1,))
        s1, s2 = self.get_var_shape(lhs), self.get_var_shape(rhs)
        if s1 == s2:
            return lhs, rhs, s1
        raise ValueError(
            f"Shapes of state variable {var} and its right-hand side update {update_info['expr']} do not"
            " match.")

    @staticmethod
    def _generate_func_head(func_name: str, code_gen: CodeGen = None, state_var: str = None, func_args: list = None,
                            imports: list = None):

        if code_gen is None:
            code_gen = CodeGen()
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
    def _generate_func_tail(code_gen: CodeGen = None, rhs_var: str = None):

        if code_gen is None:
            code_gen = CodeGen()

        code_gen.add_code_line(f"return {rhs_var}")
        code_gen.remove_indent()

        return code_gen

    @staticmethod
    def _generate_func(file_name: str, func_name: str, func_str: str, build_dir: str, decorator: Any = None,
                       **kwargs):

        if file_name:

            # save rhs function to file
            if build_dir:
                file_name = f'{build_dir}/{file_name}'
            with open(f'{file_name}.py', 'w') as f:
                f.writelines(func_str)
                f.close()

            # import function from file
            exec(f"from {file_name} import {func_name}", globals())
            rhs_eval = globals().pop(func_name)

        else:

            # generate function from string
            exec(func_str, globals())
            rhs_eval = globals().pop(func_name)

        # apply function decorator
        if decorator:
            rhs_eval = decorator(rhs_eval, **kwargs)

        return rhs_eval

    @classmethod
    def _expr_to_str(cls, expr: Any) -> str:
        expr_str = str(expr)
        for arg in expr.args:
            expr_str = expr_str.replace(str(arg), cls._expr_to_str(arg))
        while 'pr_base_index(' in expr_str:
            # replace `index` calls with brackets-based indexing
            start = expr_str.find('pr_base_index(')
            end = expr_str[start:].find(')') + 1
            expr_str = expr_str.replace(expr_str[start:start+end], f"{expr.args[0]}[{expr.args[1]}]")
        while 'pr_2d_index(' in expr_str:
            # replace `index` calls with brackets-based indexing
            start = expr_str.find('pr_2d_index(')
            end = expr_str[start:].find(')') + 1
            expr_str = expr_str.replace(expr_str[start:start + end], f"{expr.args[0]}[{expr.args[1]}, {expr.args[2]}]")
        while 'pr_range_index(' in expr_str:
            # replace `index` calls with brackets-based indexing
            start = expr_str.find('pr_range_index(')
            end = expr_str[start:].find(')') + 1
            expr_str = expr_str.replace(expr_str[start:start + end], f"{expr.args[0]}[{expr.args[1]}:{expr.args[2]}]")
        while 'pr_axis_index(' in expr_str:
            # replace `index` calls with brackets-based indexing
            start = expr_str.find('pr_axis_index(')
            end = expr_str[start:].find(')') + 1
            if len(expr.args) == 1:
                expr_str = expr_str.replace(expr_str[start:start + end], f"{expr.args[0]}[:]")
            else:
                idx = f','.join([':' for _ in range(expr.args[2])])
                idx = f'{idx},{expr.args[1]}'
                expr_str = expr_str.replace(expr_str[start:start + end], f"{expr.args[0]}[{idx}]")
        while 'pr_identity(' in expr_str:
            # replace `no_op` calls with first argument to the function call
            start = expr_str.find('pr_identity(')
            end = expr_str[start:].find(')') + 1
            expr_str = expr_str.replace(expr_str[start:start+end], f"{expr.args[0]}")
        return expr_str

    @staticmethod
    def get_var_shape(v):
        if not v.shape:
            v = np.reshape(v, (1,))
        return v.shape[0]

    @staticmethod
    def _euler_integration(steps, store_step, state_vec, state_rec, dt, rhs_func, rhs, *args):
        idx = 0
        for step in range(steps):
            if step % store_step == 0:
                state_rec[idx, :] = state_vec
                idx += 1
            state_vec += dt * rhs_func(step, state_vec, rhs, *args)
        return state_rec
