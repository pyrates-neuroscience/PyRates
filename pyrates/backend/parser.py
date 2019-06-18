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

"""This module provides parser classes and functions to parse string-based equations into symbolic representations of
operations.
"""

# external imports
from pyparsing import Literal, CaselessLiteral, Word, Combine, Optional, \
    ZeroOrMore, Forward, nums, alphas, ParserElement
from numbers import Number
import math
import typing as tp
import numpy as np

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


# expression parsers (lhs/rhs of an equation)
#############################################


class ExpressionParser(ParserElement):
    """Base class for parsing mathematical expressions from a string format into a symbolic representation of the
    mathematical operation expressed by it.

    Parameters
    ----------
    expr_str
        Mathematical expression in string format.
    args
        Dictionary containing all variables and functions needed to evaluate the expression.
    lhs
        If True, parser will treat `expr_str` as left-hand side of an equation, if False as right-hand side.
    solve
        If True, the parser will treat the left hand side as a differential variable that needs to be integrated over
        time.
    assign_add
        If True, the right hand side will be added to the current value of left hand side. If False, the left hand side
        value will be replaced with the right hand side.
    kwargs
        Additional keyword arguments to be passed to the backend functions.

    Attributes
    ----------
    lhs
        Boolean, indicates whether expression is left-hand side or right-hand side of an equation
    args
        Dictionary containing the variables of an expression
    solve
        Only relevant for lhs expressions. If True, lhs will be treated as a first-order ordinary differential equation.
    expr_str
        String representation of the mathematical expression
    expr
        Symbolic (Pyparsing-based) representation of mathematical expression.
    expr_stack
        List representation of the syntax tree of the (parsed) mathematical expression.
    expr_list
        List representation of the mathematical expression.
    op
        Operator for calculating the mathematical expression (symbolic representation).
    _op_tmp
        Helper variable for building `op`.
    ops
        Dictionary containing all mathematical operations that are allowed for a specific instance of the
        `ExpressionParser` (e.g. +, -, *, /).
    funcs
        Dictionary containing all additional functions that can be used within mathematical expressions with a specific
        instance of the `ExpressionParser` (e.g. sum(), reshape(), float32()).
    dtypes
        Dictionary containing all data-types that can be used within mathematical expressions with a specific instance
        of the `ExpressionParser` (e.g. float32, bool, int32).

    """

    def __init__(self, expr_str: str, args: dict, backend, lhs: bool = False, solve=False, assign_add=False,
                 **kwargs) -> None:
        """Instantiates expression parser.
        """

        # call super init
        #################

        super().__init__()

        # bind attributes to instance
        #############################

        # input arguments
        self.lhs = lhs
        self.args = args.copy()
        self.backend = backend
        self.parser_kwargs = kwargs
        self.solve = solve
        self.assign = '+=' if assign_add else '='
        self.constant_counter = 0

        # check whether the all important fields exist in args
        if 'updates' not in self.args.keys():
            self.args['updates'] = {}
        if 'vars' not in self.args.keys():
            self.args['vars'] = {}
        if 'inputs' not in self.args.keys():
            self.args['inputs'] = {}
        if 'lhs_evals' not in self.args.keys():
            self.args['lhs_evals'] = []

        # add functions from args dictionary to backend, if passed
        for key, val in self.args.items():
            if callable(val):
                self.backend.ops[key] = val

        # additional attributes
        self.expr_str = expr_str
        self.expr = None
        self.expr_stack = []
        self.expr_list = []
        self.expr_op = None

        # define algebra
        ################

        if not self.expr:

            # general symbols
            point = Literal(".")
            comma = Literal(",")
            colon = Literal(":")
            e = CaselessLiteral("E")
            pi = CaselessLiteral("PI")

            # parentheses
            par_l = Literal("(")
            par_r = Literal(")").setParseAction(self._push_first)
            idx_l = Literal("[")
            idx_r = Literal("]")

            # basic mathematical operations
            plus = Literal("+")
            minus = Literal("-")
            mult = Literal("*")
            div = Literal("/")
            mod = Literal("%")
            dot = Literal("@")
            exp_1 = Literal("^")
            exp_2 = Combine(mult + mult)
            transp = Combine(point + Literal("T"))
            inv = Combine(point + Literal("I"))

            # numeric types
            num_float = Combine(Word("-" + nums, nums) +
                                Optional(point + Optional(Word(nums))) +
                                Optional(e + Word("-" + nums, nums)))
            num_int = Word("-" + nums, nums)

            # variables and functions
            name = Word(alphas, alphas + nums + "_$")
            func_name = Combine(name + par_l, adjacent=True)

            # math operation groups
            op_add = plus | minus
            op_mult = mult | div | dot | mod
            op_exp = exp_1 | exp_2 | inv | transp

            # logical operations
            greater = Literal(">")
            less = Literal("<")
            equal = Combine(Literal("=") + Literal("="))
            unequal = Combine(Literal("!") + Literal("="))
            greater_equal = Combine(Literal(">") + Literal("="))
            less_equal = Combine(Literal("<") + Literal("="))

            # logical operations group
            op_logical = greater_equal | less_equal | unequal | equal | less | greater

            # pre-allocations
            self.expr = Forward()
            exponential = Forward()
            index_multiples = Forward()

            # basic organization units
            index_start = idx_l.setParseAction(self._push_first)
            index_end = idx_r.setParseAction(self._push_first)
            index_comb = colon.setParseAction(self._push_first)
            arg_comb = comma.setParseAction(self._push_first)

            # basic computation unit
            atom = (func_name + Optional(par_l.suppress() + self.expr.suppress() +
                                         ZeroOrMore((arg_comb.suppress() + self.expr.suppress() +
                                                     Optional(arg_comb.suppress()))) +
                                         par_r.suppress() + Optional(arg_comb)) +
                    Optional(self.expr.suppress() + ZeroOrMore((arg_comb.suppress() + self.expr.suppress())))
                    + par_r.suppress() | name | pi | e | num_float | num_int
                    ).setParseAction(self._push_neg_or_first) | \
                   (par_l.setParseAction(self._push_last) + self.expr.suppress() + par_r
                    ).setParseAction(self._push_neg)

            # apply indexing to atoms
            indexed = atom + ZeroOrMore((index_start + index_multiples + index_end))
            index_base = (self.expr.suppress() | index_comb)
            index_full = index_base + ZeroOrMore((index_comb + index_base)) + ZeroOrMore(index_comb)
            index_multiples << index_full + ZeroOrMore((arg_comb + index_full))

            # hierarchical relationships between mathematical and logical operations
            boolean = indexed + Optional((op_logical + indexed).setParseAction(self._push_first))
            exponential << boolean + ZeroOrMore((op_exp + Optional(exponential)).setParseAction(self._push_first))
            factor = exponential + ZeroOrMore((op_mult + exponential).setParseAction(self._push_first))
            self.expr << factor + ZeroOrMore((op_add + factor).setParseAction(self._push_first))

        # extract symbols and operations from expression string
        self.expr_list = self.expr.parseString(self.expr_str)

    def parse_expr(self):
        """Parses string-based expression.
        """

        # check whether parsing was successful
        expr_str = self.expr_str
        for sub_str in sorted(self.expr_stack, key=len)[::-1]:
            if sub_str == 'E':
                sub_str = 'e'
            expr_str = expr_str.replace(sub_str, "")
        expr_str = expr_str.replace(" ", "")
        expr_str = expr_str.replace("(", "")
        expr_str = expr_str.replace(")", "")
        expr_str = expr_str.replace("-", "")
        if len(expr_str) > 0:
            raise ValueError(f"Error while parsing expression: {self.expr_str}. {expr_str} could not be parsed.")

        # turn expression into operation
        if self.lhs:
            self.parse(self.expr_stack[:])
        else:
            self.args['rhs'] = self.parse(self.expr_stack[:])

        return self.args

    def _push_first(self, strg, loc, toks):
        """Push tokens in first-to-last order to expression stack.
        """
        self.expr_stack.append(toks[0])

    def _push_neg(self, strg, loc, toks):
        """Push negative one multiplier if on first position in toks.
        """
        if toks and toks[0] == '-':
            self.expr_stack.append('-one')

    def _push_neg_or_first(self, strg, loc, toks):
        """Push all tokens to expression stack at once (first-to-last).
        """
        if toks and toks[0] == '-':
            self.expr_stack.append('-one')
        else:
            self.expr_stack.append(toks[0])

    def _push_last(self, strg, loc, toks):
        """Push tokens in last-to-first order to expression stack.
        """
        self.expr_stack.append(toks[-1])

    def parse(self, expr_stack: list) -> tp.Any:
        """Parse elements in expression stack to operation.

        Parameters
        ----------
        expr_stack
            Ordered list with expression variables and operations.

        Returns
        -------
        tp.Any
            Parsed expression stack element (object type depends on the backend).

        """

        # get next operation from stack
        op = expr_stack.pop()

        # check type of operation
        #########################

        if op == '-one':

            # multiply expression by minus one
            #op1, op2 = self.backend.broadcast(self.parse(expr_stack), -1, **self.parser_kwargs)
            self.expr_op = self.backend.add_op('*', self.parse(expr_stack), -1, **self.parser_kwargs)

        elif op in "+-**/^@<=>=!==":

            # collect elements to combine
            op2 = self.parse(expr_stack)
            op1 = self.parse(expr_stack)

            # combine elements via mathematical/boolean operator
            #op1, op2 = self.backend.broadcast(op1, op2, **self.parser_kwargs)
            self.expr_op = self.backend.add_op(op, op1, op2, **self.parser_kwargs)

        elif ".T" == op or ".I" == op:

            # transpose/invert expression
            self.expr_op = self.backend.add_op(op, self.parse(expr_stack), **self.parser_kwargs)

        elif op == "]":

            # parse indices
            indices = []
            while len(expr_stack) > 0 and expr_stack[-1] != "[":
                index = []
                while len(expr_stack) > 0 and expr_stack[-1] not in ",[":
                    if expr_stack[-1] == ":":
                        index.append(expr_stack.pop())
                    else:
                        try:
                            int(expr_stack[-1])
                            index.append(expr_stack.pop())
                        except ValueError:
                            lhs = self.lhs
                            self.lhs = False
                            index.append(self.parse(expr_stack))
                            self.lhs = lhs
                indices.append(index[::-1])
                if expr_stack[-1] == ",":
                    expr_stack.pop()
            expr_stack.pop()

            # build string-based representation of idx
            if 'idx' not in self.args.keys():
                self.args['idx'] = {}
            idx = ""
            i = 0
            for index in indices[::-1]:
                for ind in index:
                    if type(ind) == str:
                        idx += ind
                    elif isinstance(ind, Number):
                        idx += f"{ind}"
                    else:
                        self.args['idx'][f'idx_var_{i}'] = ind
                        idx += f"idx_var_{i}"
                    i += 1
                idx += ","
            idx = idx[0:-1]

            # extract variable
            if self.lhs:
                op = expr_stack[-1]
                if op in self.args['updates']:
                    op_to_idx = self.args['updates'][op]
                else:
                    self.lhs = False
                    op_to_idx = self.parse([op])
                    self.lhs = True
                self.args['lhs_evals'].append(op)
            else:
                op_to_idx = self.parse(expr_stack)

            # apply index
            op_idx = self.apply_idx(op_to_idx, idx, **self.parser_kwargs)
            self.args['updates'][op] = op_idx
            self.expr_op = op_idx

        elif op == "PI":

            # return float representation of pi
            self.expr_op = math.pi

        elif op == "E":

            # return float representation of e
            self.expr_op = math.e

        elif op in self.args['inputs']:

            # extract input variable from args dict and move it to the vars collection
            self.args['vars'][op] = self.args['inputs'].pop(op)
            if self.lhs:
                self.parse([op])
            else:
                self.expr_op = self.args['vars'][op]

        elif op in self.args['vars']:

            if self.lhs:

                if self.solve:

                    # parse dt
                    self.lhs = False
                    dt = self.parse(['dt'])
                    self.lhs = True

                    # get variables
                    var = self.args['vars'][op]
                    var_name = f'{op}_old'
                    old_var = self.args['vars'][var_name]

                    # calculate update of differential equation
                    var_update = self.update(old_var, self.args.pop('rhs'), dt, **self.parser_kwargs)
                    #var, var_update = self.backend.broadcast(var, var_update, **self.parser_kwargs)
                    self.args['updates'][op] = self.backend.add_op(self.assign, var, var_update, **self.parser_kwargs)
                    self.args['lhs_evals'].append(op)
                    self.expr_op = self.args['updates'][op]

                else:

                    # update variable according to rhs
                    var = self.args['updates'][op] if op in self.args['updates'] else self.args['vars'][op]
                    #var, var_update = self.backend.broadcast(var, self.args.pop('rhs'), **self.parser_kwargs)
                    self.args['updates'][op] = self.backend.add_op(self.assign, var, self.args.pop('rhs'),
                                                                   **self.parser_kwargs)
                    self.args['lhs_evals'].append(op)
                    self.expr_op = self.args['updates'][op]

            else:

                # extract constant/variable from args dict
                self.expr_op = self.args['vars'][f'{op}_old'] if f'{op}_old' in self.args['vars'] \
                    else self.args['vars'][op]

        elif op in self.args['updates']:

            self.expr_op = self.args['updates'][op]

        elif any(["float" in op, "bool" in op, "int" in op, "complex" in op]):

            expr_stack.pop(-1)

            # extract data type
            try:
                self.expr_op = self.backend.add_op('cast', self.parse(expr_stack), op[0:-1], **self.parser_kwargs)
            except AttributeError:
                raise AttributeError(f"Datatype casting error in expression: {self.expr_str}. "
                                     f"{op[0:-1]} is not a valid data-type for this parser.")

        elif op[-1] == "(":

            expr_stack.pop(-1)

            # parse arguments
            args = []
            while len(expr_stack) > 0:
                args.append(self.parse(expr_stack))
                if len(expr_stack) == 0 or expr_stack[-1] != ",":
                    break
                else:
                    expr_stack.pop()

            # apply function to arguments
            try:
                if len(args) == 1:
                    self.expr_op = self.backend.add_op(op[0:-1], args[0], **self.parser_kwargs)
                else:
                    self.expr_op = self.backend.add_op(op[0:-1], *tuple(args[::-1]), **self.parser_kwargs)
            except KeyError:
                raise KeyError(
                    f"Undefined function in expression: {self.expr_str}. {op[0:-1]} needs to be provided "
                    f"in arguments dictionary.")

        elif op == ")":

            # check whether expression in parenthesis is a group of arguments to a function
            start_par = expr_stack.index("(")
            if "," in expr_stack[start_par:]:

                args = []
                while True:
                    args.append(self.parse(expr_stack))
                    if expr_stack[-1] == ",":
                        expr_stack.pop(-1)
                    elif expr_stack[-1] == "(":
                        expr_stack.pop(-1)
                        break
                    else:
                        break
                self.expr_op = args[::-1]

            else:

                self.expr_op = self.parse(expr_stack)
                expr_stack.pop(-1)

        elif any([op == "True", op == "true", op == "False", op == "false"]):

            # return boolean
            self.expr_op = True if op in "Truetrue" else False

        elif "." in op:

            self.expr_op = float(op)

        elif op.isnumeric():

            self.expr_op = int(op)

        elif op[0].isalpha():

            if self.lhs:

                # add new variable to arguments that represents rhs op
                rhs = self.args.pop('rhs')
                shape = rhs.shape if rhs.shape else (1,)
                new_var = self.backend.add_var(vtype='state_var', name=op, value=0.,
                                               shape=shape, dtype=rhs.dtype, **self.parser_kwargs)
                self.args['vars'][op] = new_var
                #new_var, rhs = self.backend.broadcast(new_var, rhs, **self.parser_kwargs)
                self.args['updates'][op] = self.backend.add_op(self.assign, new_var, rhs, **self.parser_kwargs)
                self.args['lhs_evals'].append(op)
                self.expr_op = self.args['updates'][op]

            else:

                raise ValueError(f"Undefined variable detected in expression: {self.expr_str}. {op} was not found "
                                 f"in the respective arguments dictionary.")

        else:

            raise ValueError(f"Undefined operation detected in expression: {self.expr_str}. {op} cannot be "
                             f"interpreted by this parser.")

        return self.expr_op

    def apply_idx(self, op: tp.Any, idx: tp.Any, **kwargs) -> tuple:
        """Apply index idx to operation op.

        Parameters
        ----------
        op
            Operation to be indexed.
        idx
            Index to op.
        kwargs
            Additional keyword arguments to be passed to the indexing functions.

        Returns
        -------
        tp.Any
            Result of applying idx to op.

        """

        kwargs.update(self.parser_kwargs)

        # do some initial checks
        if self.lhs and self.solve:
            raise ValueError(f'Indexing of differential equations is currently not supported. Please consider '
                             f'changing equation {self.expr_str}.')

        # get update
        update = self.args.pop('rhs', None)

        # get constants/variables that are part of the index
        args = []
        i = 0
        if idx in self.args['idx']:
            idx = self.args['idx'].pop(idx)
        if type(idx) is str:
            idx_old = idx
            idx = []
            for idx_tmp in idx_old.split(','):
                for idx_tmp2 in idx_tmp.split(':'):
                    idx.append(idx_tmp2)
                    if idx_tmp2 in self.args['idx']:
                        idx_var = self.args['idx'].pop(idx_tmp2)
                        if not hasattr(idx_var, 'short_name'):
                            idx_var.short_name = idx_tmp2
                            i += 1
                        else:
                            idx[-1] = idx_var.short_name
                        args.append(idx_var)
                    idx.append(':')
                idx.pop(-1)
                idx.append(',')
            idx.pop(-1)
            idx = "".join(idx)

        return self.backend.apply_idx(op, idx, update, *tuple(args))

    def update(self, var_old, var_delta, dt, **kwargs):
        """Solves single step of a differential equation.

        Parameters
        ----------
        var_old
        var_delta
        dt
        kwargs

        Returns
        -------

        """

        kwargs.update(self.parser_kwargs)
        var_delta, dt = self.backend.broadcast(var_delta, dt, **kwargs)
        var_update = self.backend.add_op('*', var_delta, dt, **kwargs)
        #var_old, var_update = self.backend.broadcast(var_old, var_update, **kwargs)
        return self.backend.add_op('+', var_old, var_update, **kwargs)


def parse_equation_list(equations: list, equation_args: dict, backend: tp.Any, **kwargs) -> dict:
    """Parses a list of equations into the backend.

    Parameters
    ----------
    equations
        Collection of equations that should be evaluated together.
    equation_args
        Key-value pairs of arguments needed for parsing the equations.
    backend
        Backend instance to parse the equations into.
    kwargs
        Additional keyword arguments to be passed to the backend.

    Returns
    -------
    dict
        The updated equations args (in-place manipulation of all variables in equation_args happens during
        equation parsing).

    """

    # preprocess equations and equation arguments
    #############################################

    if 'inputs' not in equation_args:
        equation_args['inputs'] = {}

    # preprocess equations
    left_hand_sides, right_hand_sides, update_types = preprocess_equations(equations, solver='euler')

    # go through pre-processed equations and add new variables for old values of state variables
    for lhs in left_hand_sides:

        # get key of lhs variable
        lhs_var = lhs.split('[')[0]
        lhs_var = lhs_var.replace(' ', '')

        # see whether lhs variable is a state variable. If yes, add variable to argument dictionary (for old value)
        for key, var in equation_args['vars'].copy().items():
            if key == lhs_var and '_old' not in key:
                if type(var) is dict:
                    var_dict = var.copy()
                    if 'value' in var_dict and hasattr(var_dict['value'], 'shape'):
                        var_dict['value'] = np.asarray(var_dict['value'].tolist())
                elif callable(var):
                    var_tmp = var()
                    var_dict = {'vtype': 'state_var',
                                'dtype': var_tmp.dtype,
                                'shape': var_tmp.shape,
                                'value': 0.}
                else:
                    var_dict = {'vtype': 'state_var',
                                'dtype': var.dtype,
                                'shape': var.shape,
                                'value': 0.}
                equation_args['vars'].update(parse_dict({f'{key}_old': var_dict}, backend=backend, **kwargs))

    # parse equations
    #################

    for lhs, rhs, add_assign in zip(left_hand_sides, right_hand_sides, update_types):
        equation_args = parse_equation(lhs, rhs, equation_args, backend, assign_add=add_assign, **kwargs)

    return equation_args


def parse_equation(lhs: str, rhs: str, equation_args: dict, backend: tp.Any, solve: bool = False,
                   assign_add: bool = False, **kwargs) -> dict:
    """Parses lhs and rhs of an equation.

    Parameters
    ----------
    lhs
        Left hand side of an equation.
    rhs
        Right hand side of an equation.
    equation_args
        Dictionary containing all variables and functions needed to evaluate the expression.
    backend
        Backend instance to parse the equation into.
    solve
        If true, the left hand side will be treated as a differential variable that needs to be integrated over time.
    assign_add
        If true, the right hand side will be added to the current value of the left hand side instead of replacing it.
    kwargs
        Additional keyword arguments to be passed to the backend.

    Returns
    -------
    dict
        The updated equation_args dictionary (variables were manipulated in place during equation parsing).

    """

    # parse arguments
    #################

    args_tmp = {}
    for key, arg in equation_args['vars'].items():
        if type(arg) is dict and 'vtype' in arg.keys():
            args_tmp[key] = arg
    args_tmp = parse_dict(args_tmp, backend, **kwargs)
    equation_args['vars'].update(args_tmp)

    # parse equation
    ################

    # parse rhs
    rhs_parser = ExpressionParser(expr_str=rhs, args=equation_args, backend=backend, **kwargs)
    equation_args = rhs_parser.parse_expr()

    # parse lhs
    lhs_parser = ExpressionParser(expr_str=lhs, args=equation_args, lhs=True, solve=solve, backend=backend,
                                  assign_add=assign_add, **kwargs)

    return lhs_parser.parse_expr()


def parse_dict(var_dict: dict, backend, **kwargs) -> dict:
    """Parses a dictionary with variable information and creates backend variables from that information.

    Parameters
    ----------
    var_dict
        Contains key-value pairs for each variable that should be translated into the backend graph.
        Each value is a dictionary again containing the variable information (needs at least a field for `vtype`).
    backend
        Backend instance that the variables should be added to.
    kwargs
        Additional keyword arguments to be passed to the backend.

    Returns
    -------
    dict
        Key-value pairs with the backend variable names and handles.

    """

    var_dict_new = {}

    # go through dictionary items and instantiate variables
    #######################################################

    for var_name, var in var_dict.items():

        # instantiate variable
        if var['vtype'] == 'raw':
            var_dict_new[var_name] = var['value']
        else:
            var.update(kwargs)
            var_dict_new[var_name] = backend.add_var(name=var_name, **var)

    return var_dict_new


def preprocess_equations(eqs: list, solver: str) -> tuple:
    """Turns differential equations into simple algebraic equations using a certain solver scheme.

    Parameters
    ----------
    eqs
        Collection of equations to be pre-processed.
    solver
        Type of the solver.

    Returns
    -------
    tuple
        Contains left hand sides, right hand sides and left hand side update modes (= or +=)
    """

    # collect equation specifics
    ############################

    lhs_col, rhs_col, add_assign_col = [], [], []
    de_lhs_col, de_rhs_col, lhs_var_col = [], [], []

    for eq in eqs:

        # split equation into lhs and rhs and check update type
        if ' += ' in eq:
            lhs, rhs = eq.split(' += ')
            add_assign = True
        else:
            lhs, rhs = eq.split(' = ')
            add_assign = False

        # for the left-hand side, check whether it includes a differential operator
        if "d/dt" in lhs:
            diff_eq = True
            lhs_split = lhs.split('*')
            lhs = "".join(lhs_split[1:])
        else:
            diff_eq = False

        # get key of lhs variable
        lhs_var = lhs.split('[')[0]
        lhs_var = lhs_var.replace(' ', '')

        # store equation specifics
        if diff_eq:
            if add_assign:
                raise ValueError(f'Wrong assignment method for equation: {eq}. '
                                 f'A differential equation cannot be combined with a `+=` assign.')
            de_lhs_col.append(lhs)
            de_rhs_col.append(rhs)
            lhs_var_col.append(lhs_var)
        else:
            lhs_col.append(lhs)
            rhs_col.append(rhs)
            add_assign_col.append(add_assign)

    # solve differential equations
    ##############################

    eqs_new = []

    if solver == 'euler':

        # use explicit forward euler
        for lhs, rhs in zip(de_lhs_col, de_rhs_col):
            eqs_new.append(f"{lhs} += dt * ({rhs})")

    elif solver == 'rk23':

        # use second-order runge-kutta solver
        k1_col, k2_col, rhs_new = [], [], []
        for lhs, rhs, lhs_var in zip(de_lhs_col, de_rhs_col, lhs_var_col):
            k1_col.append(f"{lhs_var}_k1 = {rhs}")
            k1_col.append(f"{lhs_var}_k1_added = ({lhs_var} + dt * {lhs_var}_k1 * 2/3)")
            for lhs_var_tmp in lhs_var_col:
                rhs = replace(rhs, lhs_var_tmp, f"{lhs_var_tmp}_k1_added")
            k2_col.append(f"{lhs_var}_k2 = {rhs}")
            k2_col.append(f"{lhs} += dt * (0.25 * {lhs_var}_k1 + 0.75 * {lhs_var}_k2)")
        eqs_new = k1_col + k2_col

    elif solver == 'mp':

        # use second-order runge-kutta solver
        k1_col, k2_col, rhs_new = [], [], []
        for lhs, rhs, lhs_var in zip(de_lhs_col, de_rhs_col, lhs_var_col):
            k1_col.append(f"{lhs_var}_k1 = {rhs}")
            k1_col.append(f"{lhs_var}_k1_added = ({lhs_var} + dt * {lhs_var}_k1 * 0.5)")
            for lhs_var_tmp in lhs_var_col:
                rhs = replace(rhs, lhs_var_tmp, f"{lhs_var_tmp}_k1_added")
            k2_col.append(f"{lhs_var}_k2 = {rhs}")
            k2_col.append(f"{lhs} += dt * {lhs_var}_k2")
        eqs_new = k1_col + k2_col

    else:

        raise ValueError(f'Wrong solver type: {solver}. '
                         f'Please check the docstring of this function for available solvers.')

    # preprocess the newly added differential equation updates
    ##########################################################

    if eqs_new:
        lhs_tmp, rhs_tmp, add_assign_tmp = preprocess_equations(eqs_new, solver)
        lhs_col += lhs_tmp
        rhs_col += rhs_tmp
        add_assign_col += add_assign_tmp

    return lhs_col, rhs_col, add_assign_col


def replace(eq: str, term: str, replacement: str) -> str:
    """Replaces a term in an equation with a replacement term.

    Parameters
    ----------
    eq
        Equation that includes the term.
    term
        Term that should be replaced.
    replacement
        Replacement for all occurences of term.

    Returns
    -------
    str
        The updated equation.

    """

    # define follow-up operations/signs that are allowed to follow directly after term in eq
    allowed_follow_ops = '+=*/^<>=!.%@[]():, '

    # replace every proper appearance of term in eq with replacement
    ################################################################

    eq_new = ""
    idx = eq.find(term)

    # go through all appearances of term in eq
    while idx != -1:

        # get idx of sign that follows after term
        idx_follow_op = idx+len(term)

        # if it is an allowed sign, replace term, else not
        if (idx_follow_op < len(eq) and eq[idx_follow_op] in allowed_follow_ops) or idx_follow_op == len(eq):
            eq_new += f"{eq[:idx]} {replacement}"
        else:
            eq_new += f"{eq[:idx_follow_op]}"

        # jump to next appearance of term in eq
        eq = eq[idx_follow_op:]
        idx = eq.find(term)

    # add rest of eq to new eq
    eq_new += eq

    return eq_new
