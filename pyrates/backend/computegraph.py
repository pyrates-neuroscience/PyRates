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

"""This module provides the backend class that should be used to set up any backend in pyrates.
"""

# external imports
from typing import Any, Callable, Union, Iterable, Optional
from networkx import MultiDiGraph
from sympy import Symbol, Expr, Function
from copy import deepcopy
import numpy as np

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


#########################
# compute graph classes #
#########################


# numpy-based node class
class ComputeNode:
    """Base class for adding variables to the compute graph. Creates a numpy array with additional attributes
    for variable identification/retrieval from graph. Should be used as parent class for custom variable classes.

    Parameters
    ----------
    name
        Full name of the variable in the original NetworkGraph (including the node and operator it belongs to).
    dtype
        Data-type of the variable. For valid data-types, check the documentation of the backend in use.
    shape
        Shape of the variable.
    """

    __slots__ = ["name", "symbol", "dtype", "shape", "_value"]

    def __init__(self, name: str, symbol: Union[Symbol, Expr, Function], dtype: Optional[str] = None,
                 shape: Optional[str] = None):
        """Instantiates a basic node of a ComputeGraph instance.
        """

        self.name = name
        self.symbol = symbol
        self.dtype = dtype
        self.shape = shape
        self._value = self._get_value(shape=shape, dtype=dtype)

        # # check whether necessary arguments were provided
        # if all([arg is None for arg in [shape, value, dtype]]):
        #     raise ValueError('Either `value` or `shape` and `dtype` need to be provided')
        #
        # # get shape
        # if not shape:
        #     shape = value.shape if hasattr(value, 'shape') else np.shape(value)
        #
        # # get data type
        # if not dtype:
        #     if hasattr(value, 'dtype'):
        #         dtype = value.dtype
        #     else:
        #         try:
        #             dtype = np.dtype(value)
        #         except TypeError:
        #             dtype = type(value)
        # dtype = dtype.name if hasattr(dtype, 'name') else str(dtype)
        # if dtype in backend.dtypes:
        #     dtype = backend.dtypes[dtype]
        # else:
        #     for dtype_tmp in backend.dtypes:
        #         if dtype_tmp in dtype:
        #             dtype = backend.dtypes[dtype_tmp]
        #             break
        #     else:
        #         dtype = backend._float_def
        #         warnings.warn(f'WARNING! Unknown data type of variable {name}: {dtype}. '
        #                       f'Datatype will be set to default type: {dtype}.')
        #
        # # create variable
        # if vtype == 'state_var' and 1 in tuple(shape) and squeeze:
        #     idx = tuple(shape).index(1)
        #     shape = list(shape)
        #     shape.pop(idx)
        #     shape = tuple(shape)
        # if callable(value):
        #     obj = value
        # else:
        #     try:
        #         # normal variable
        #         value = cls._get_value(value, dtype, shape)
        #         if squeeze:
        #             value = cls.squeeze(value)
        #         obj = cls._get_var(value, name, dtype)
        #     except TypeError:
        #         # list of callables
        #         obj = cls._get_var(value, name, dtype)
        #
        # # store additional attributes on variable object
        # obj.short_name = name.split('/')[-1]
        # if not hasattr(obj, 'name'):
        #     obj.name = name
        # else:
        #     name = obj.name
        # obj.vtype = vtype
        #
        # return obj, name

    def reshape(self, shape: tuple, **kwargs):

        self._value = self.value.reshape(shape, **kwargs)
        return self

    def squeeze(self, axis=None):
        self._value = self.value.squeeze(axis=axis)
        return self

    @property
    def value(self):
        """Returns current value of BaseVar.
        """
        return self._value

    def _is_equal_to(self, v):
        for attr in self.__slots__:
            if not hasattr(v, attr) or getattr(v, attr) != getattr(self, attr):
                return False
        return True

    @staticmethod
    def _get_value(value: Optional[Union[list, np.ndarray]] = None, dtype: Optional[str] = None,
                   shape: Optional[tuple] = None):
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

    def __deepcopy__(self, memodict: dict):
        node = ComputeNode(name=self.name, symbol=self.symbol, dtype=self.dtype, shape=self.shape)
        node._value = node._value[:]
        return node

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(str(self))


class ComputeVar(ComputeNode):
    """Class for variables and vector-valued constants in the ComputeGraph.
    """

    __slots__ = super().__slots__ + ["vtype"]

    def __init__(self, name: str, symbol: Union[Symbol, Expr, Function], vtype: str, dtype: Optional[str] = None,
                 shape: Optional[str] = None, value: Optional[Union[list, np.ndarray]] = None):

        # set attributes
        super().__init__(name=name, symbol=symbol, dtype=dtype, shape=shape)
        self.vtype = vtype

        # adjust variable value
        self._value = self._get_value(value=value, shape=shape, dtype=dtype)


class ComputeOp(ComputeNode):
    """Class for ComputeGraph nodes that represent mathematical operations.
    """

    __slots__ = super().__slots__ + ["func", "expr"]

    def __init__(self, name: str, symbol: Union[Symbol, Expr, Function], func: Callable, expr: Expr,
                 dtype: Optional[str] = None, shape: Optional[str] = None):

        # set attributes
        super().__init__(name=name, symbol=symbol, dtype=dtype, shape=shape)
        self.func = func
        self.expr = expr


# networkx-based graph class
class ComputeGraph(MultiDiGraph):
    """Creates a compute graph where nodes are all constants and variables of the network and edges are the mathematical
    operations linking those variables/constants together to form equations.
    """

    def __init__(self, backend: str, **kwargs):

        super().__init__()

        # choose a backend
        if backend == 'tensorflow':
            from pyrates.backend.tensorflow_backend import TensorflowBackend
            backend = TensorflowBackend
        elif backend == 'fortran':
            from pyrates.backend.fortran_backend import FortranBackend
            backend = FortranBackend
        elif backend == 'PyAuto' or backend == 'pyauto':
            from pyrates.backend.fortran_backend import PyAutoBackend
            backend = PyAutoBackend
        else:
            from pyrates.backend.base_backend import BaseBackend
            backend = BaseBackend

        # set attributes
        self.backend = backend(**kwargs)
        self.var_updates = {'DEs': dict(), 'non-DEs': dict()}
        self._eq_nodes = []

    def add_var(self, label: str, value: Any, vtype: str, **kwargs):

        unique_label = self._generate_unique_label(label)
        var = ComputeVar(value=value, vtype=vtype, **kwargs)
        super().add_node(unique_label, node=var)
        return unique_label, self.nodes[unique_label]['node']

    def add_op(self, inputs: Union[list, tuple], label: str, expr: Expr, func: Callable, **kwargs):

        # add target node that contains result of operation
        unique_label = self._generate_unique_label(label)
        op = ComputeOp(func=func, expr=expr, **kwargs)
        super().add_node(unique_label, node=op)

        # add edges from source nodes to target node
        for i, v in enumerate(inputs):
            super().add_edge(v, unique_label, key=i)

        return unique_label, self.nodes[unique_label]['node']

    def add_var_update(self, var: str, update: str, differential_equation: bool = False):

        # store mapping between left-hand side variable and right-hand side update
        if differential_equation:
            self.var_updates['DEs'][var] = update
        else:
            self.var_updates['non-DEs'][var] = update

        # remember var and update node to ensure that they are not pruned during compilation
        self._eq_nodes.extend([var, update])

    def eval_graph(self):

        for n in self.var_updates['non-DEs'].values():
            self.eval_subgraph(n)
        return self.eval_nodes(self.var_updates['DEs'].values())

    def eval_nodes(self, nodes: Iterable):

        return [self.eval_node(n) for n in nodes]

    def eval_node(self, n):

        inputs = [self.eval_node(inp) for inp in self.predecessors(n)]
        if 'func' in self.nodes[n]:
            return self.nodes[n]['func'](*tuple(inputs))
        return self.nodes[n]['value']

    def eval_subgraph(self, n):

        inputs = []
        input_nodes = [node for node in self.predecessors(n)]
        for inp in input_nodes:
            inputs.append(self.eval_subgraph(inp))
            self.remove_node(inp)

        node = self.nodes[n]
        if inputs:
            node['value'] = node['func'](*tuple(inputs))

        return node['value']

    def remove_subgraph(self, n):

        for inp in self.predecessors(n):
            self.remove_subgraph(inp)
        self.remove_node(n)

    def compile(self, in_place: bool = True):

        G = self if in_place else deepcopy(self)

        # evaluate constant-based operations
        out_nodes = [node for node, out_degree in G.out_degree if out_degree == 0]
        for node in out_nodes:

            # process inputs of node
            for inp in G.predecessors(node):
                if G.nodes[inp]['vtype'] == 'constant':
                    G.eval_subgraph(inp)

            # evaluate node if all its inputs are constants
            if all([G.nodes[inp]['vtype'] == 'constant' for inp in G.predecessors(node)]):
                G.eval_subgraph(node)

        # remove unconnected nodes and constants from graph
        G._prune()

        # broadcast all variable shapes to a common number of dimensions
        # for node in [node for node, out_degree in G.out_degree if out_degree == 0]:
        #     G.broadcast_op_inputs(node, squeeze=False)

        return G

    def broadcast_op_inputs(self, n, squeeze: bool = False, target_shape: tuple = None, depth=0, max_depth=20,
                            target_dtype: np.dtype = np.float32):

        try:

            # attempt to perform graph operation
            if target_shape:
                raise ValueError
            return self.eval_node(n)

        except (ValueError, IndexError, TypeError) as e:

            # TODO: re-work broadcasting. Issue: broadcasting needs to be applied to all variables that may be affected
            #  by it across all network equations
            if depth > max_depth:
                raise e

            # collect inputs to operator node
            inputs, nodes, shapes = [], [], []
            for inp in self.predecessors(n):

                # ensure the whole operation tree of input is broadcasted to matching shapes
                inp_eval = self.broadcast_op_inputs(inp, depth=depth+1)

                inputs.append(inp_eval)
                nodes.append(inp)
                shapes.append(len(inp_eval.shape) if hasattr(inp_eval, 'shape') else 1)

            if isinstance(e, ValueError):

                # broadcast shapes of inputs
                if not target_shape:
                    target_shape = inputs[np.argmax(shapes)].shape
                for i in range(len(inputs)):

                    inp_eval = inputs[i]

                    # get new shape of input
                    new_shape = self._broadcast_shapes(target_shape, inp_eval.shape, squeeze=squeeze)

                    # reshape input
                    if new_shape != inp_eval.shape:
                        if 'func' in self.nodes[nodes[i]]:
                            self.broadcast_op_inputs(nodes[i], squeeze=squeeze, target_shape=new_shape, depth=depth+1)
                        else:
                            inp_eval = inp_eval.reshape(new_shape)
                            self.nodes[nodes[i]]['value'] = inp_eval

                if n in self.var_updates['non-DEs']:
                    return self.broadcast_op_inputs(self.var_updates['non-DEs'][n], squeeze=True, depth=depth+1)
                return self.broadcast_op_inputs(n, squeeze=True, depth=depth+1)

            elif isinstance(e, TypeError):

                for node, var in zip(nodes, inputs):
                    var = var.astype(target_dtype)
                    self.nodes[node]['value'] = var
                return self.broadcast_op_inputs(n, depth=depth+1)

            else:

                # broadcast shape of variable that an index should be applied to
                if not target_shape:
                    target_shape = inputs[0].shape + (1,)

                # get new shape of indexed variable
                inp_eval = inputs[0]
                new_shape = self._broadcast_shapes(target_shape, inp_eval.shape, squeeze=squeeze)

                # update right-hand side of non-DE instead of the target variable, if suitable
                if nodes[0] in self.var_updates['non-DEs']:
                    return self.broadcast_op_inputs(self.var_updates['non-DEs'][nodes[0]], target_shape=new_shape,
                                                    depth=depth + 1)

                # reshape variable
                if new_shape != inp_eval.shape:
                    if 'func' in self.nodes[nodes[0]]:
                        self.broadcast_op_inputs(nodes[0], squeeze=squeeze, target_shape=new_shape, depth=depth + 1)
                    else:
                        inp_eval = inp_eval.reshape(new_shape)
                        self.nodes[nodes[0]]['value'] = inp_eval

                return self.broadcast_op_inputs(n, squeeze=True, depth=depth + 1)

    def node_to_expr(self, n: str, **kwargs) -> tuple:

        expr_args = []
        try:
            expr = self.nodes[n]['expr']
            expr_info = {self.nodes[inp]['symbol']: self.node_to_expr(inp, **kwargs) for inp in self.predecessors(n)}
            for expr_old, (args, expr_new) in expr_info.items():
                expr = expr.replace(expr_old, expr_new)
                expr_args.extend(args)
            # TODO: check beforehand, whether next step is necessary (e.g. indicate existence of undefined functions)
            for expr_old, expr_new in kwargs.items():
                expr = expr.replace(Function(expr_old), Function(expr_new))
            return expr_args, expr
        except KeyError:
            if self.nodes[n]['vtype'] in ['constant', 'input']:
                expr_args.append(n)
            return expr_args, self.nodes[n]['symbol']

    def _prune(self):

        # remove all subgraphs that contain constants only
        for n in [node for node, out_degree in self.out_degree if out_degree == 0]:
            if self.nodes[n]['vtype'] == 'constant' and n not in self._eq_nodes:
                self.remove_subgraph(n)

        # remove all unconnected nodes
        for n in [node for node, out_degree in self.out_degree if out_degree == 0]:
            if self.in_degree(n) == 0 and n not in self._eq_nodes:
                self.remove_node(n)

    def _generate_unique_label(self, label: str):

        if label in self.nodes:
            label_split = label.split('_')
            try:
                new_label = "_".join(label_split[:-1] + [f"{int(label_split[-1])+1}"])
            except ValueError:
                new_label = f"{label}_0"
            return self._generate_unique_label(new_label)
        else:
            return label

    @staticmethod
    def _broadcast_shapes(s1: tuple, s2: tuple, squeeze: bool):

        new_shape = []
        for j, s in enumerate(s1):
            if squeeze:
                try:
                    if s2[j] == s1[j] or s2[j] == 1:
                        new_shape.append(s1[j])
                    else:
                        new_shape.append(1)
                except IndexError:
                    pass
            else:
                try:
                    if s2[j] == s1[j] or s1[j] == 1:
                        new_shape.append(s2[j])
                    else:
                        new_shape.append(1)
                except IndexError:
                    new_shape.append(1)

        return tuple(new_shape)
