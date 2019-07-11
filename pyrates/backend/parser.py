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

import math
import typing as tp
from copy import deepcopy
from numbers import Number

# external imports
from pyparsing import Literal, CaselessLiteral, Word, Combine, Optional, \
    ZeroOrMore, Forward, nums, alphas, ParserElement

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
    backend
        Backend instance in which to parse all variables and operations.
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

    def __init__(self, expr_str: str, args: dict, backend: tp.Any, **kwargs) -> None:
        """Instantiates expression parser.
        """

        # call super init
        #################

        super().__init__()

        # bind attributes to instance
        #############################

        # input arguments
        self.vars = args.copy()
        self.backend = backend
        self.parser_kwargs = kwargs

        self.lhs, self.rhs, self._diff_eq, self._assign_type, self.lhs_key = self._preprocess_expr_str(expr_str)

        # add functions from args dictionary to backend, if passed
        for key, val in args.items():
            if callable(val):
                self.backend.ops[key] = val

        # additional attributes
        self.expr_str = expr_str
        self.expr = None
        self.expr_stack = []
        self.expr_list = []
        self.op = None
        self._finished_rhs = False
        self.rhs_eval = None
        self._instantaneous = kwargs.pop('instantaneous', False)

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

    def parse_expr(self):
        """Parses string-based expression.
        """

        # extract symbols and operations from equations right-hand side
        self.expr_list = self.expr.parseString(self.rhs)
        self._check_parsed_expr(self.rhs)

        # parse rhs into backend
        rhs = self.parse(self.expr_stack[:])

        if self.lhs_key in self.vars and not self._instantaneous:

            # create new variable for lhs update
            name = f"{self.lhs_key}_delta"
            i = 0
            name = f"{name}_{i}"
            while name in self.vars:
                i += 1
                name[-1] = i
            dtype = rhs.dtype if hasattr(rhs, 'dtype') else self.backend._float_def
            shape = rhs.shape if hasattr(rhs, 'shape') else ()

            # assign rhs to new variable
            delta = self.backend.add_var('state_var', name=name, dtype=dtype, shape=shape, **self.parser_kwargs)
            self.backend.add_op('=', delta, rhs)
            self.rhs = delta

        else:

            self.rhs = rhs

        self.rhs_eval = rhs
        self.clear()
        self._finished_rhs = True

        # extract symbols and operations from left-hand side
        self.expr_list = self.expr.parseString(self.lhs)
        self._check_parsed_expr(self.lhs)

        # parse lhs into backend
        self.lhs = self._update_lhs()

        return self.lhs, self.rhs, self.vars

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
            self.op = self.backend.add_op('*', self.parse(expr_stack), -1, **self.parser_kwargs)

        elif op in ["*=", "/=", "+=", "-=", "="]:

            # collect rhs
            op1 = self.parse(expr_stack)

            # collect lhs
            indexed_lhs = True if "]" in expr_stack else False
            op2 = self.parse(expr_stack)

            # combine elements via mathematical/boolean operator
            if indexed_lhs:
                self.op = self._apply_idx(op=op2[0], idx=op2[1], update=op1, update_type=op, **self.parser_kwargs)
            else:
                self.op = self.backend.add_op(op, op2, op1, **self.parser_kwargs)

        elif op in "+-/**^@<=>=!==":

            # collect elements to combine
            op2 = self.parse(expr_stack)
            op1 = self.parse(expr_stack)

            # combine elements via mathematical/boolean operator
            self.op = self.backend.add_op(op, op1, op2, **self.parser_kwargs)

        elif ".T" == op or ".I" == op:

            # transpose/invert expression
            self.op = self.backend.add_op(op, self.parse(expr_stack), **self.parser_kwargs)

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
                            tmp = self._finished_rhs
                            self._finished_rhs = False
                            index.append(self.parse(expr_stack))
                            self._finished_rhs = tmp
                indices.append(index[::-1])
                if expr_stack[-1] == ",":
                    expr_stack.pop()
            expr_stack.pop()

            # build string-based representation of idx
            if 'idx' not in self.vars.keys():
                self.vars['idx'] = {}
            idx = ""
            i = 0
            for index in indices[::-1]:
                for ind in index:
                    if type(ind) == str:
                        idx += ind
                    elif isinstance(ind, Number):
                        idx += f"{ind}"
                    else:
                        self.vars['idx'][f'idx_var_{i}'] = ind
                        idx += f"idx_var_{i}"
                    i += 1
                idx += ","
            idx = idx[0:-1]

            # extract variable and apply idx if its a rhs variable. Else return variable and index
            if self._finished_rhs:
                op = expr_stack.pop(-1)
                if op in self.vars:
                    op_to_idx = self.vars[op]
                else:
                    op_to_idx = self.parse([op])
                self.op = (op_to_idx, idx)
            else:
                op_to_idx = self.parse(expr_stack)
                op_idx = self._apply_idx(op_to_idx, idx, **self.parser_kwargs)
                self.op = op_idx

        elif op == "PI":

            # return float representation of pi
            self.op = math.pi

        elif op == "E":

            # return float representation of e
            self.op = math.e

        elif op in self.vars:

            # extract constant/variable from args dict
            self.op = self.vars[op]

        elif any(["float" in op, "bool" in op, "int" in op, "complex" in op]):

            expr_stack.pop(-1)

            # extract data type
            try:
                self.op = self.backend.add_op('cast', self.parse(expr_stack), op[0:-1], **self.parser_kwargs)
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
                    self.op = self.backend.add_op(op[0:-1], args[0], **self.parser_kwargs)
                else:
                    self.op = self.backend.add_op(op[0:-1], *tuple(args[::-1]), **self.parser_kwargs)
            except KeyError:
                raise KeyError(
                    f"Undefined function in expression: {self.expr_str}. {op[0:-1]} needs to be provided "
                    f"in arguments dictionary.")

        elif op == ")":

            # check whether expression in parenthesis is a group of arguments to a function
            start_par = -1
            found_end = 0
            while found_end < 1:
                if "(" in expr_stack[start_par]:
                    found_end += 1
                if ")" in expr_stack[start_par]:
                    found_end -= 1
                start_par -= 1
            if "," in expr_stack[start_par+1:]:

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
                self.op = args[::-1]

            else:

                self.op = self.parse(expr_stack)
                expr_stack.pop(-1)

        elif any([op == "True", op == "true", op == "False", op == "false"]):

            # return boolean
            self.op = True if op in "Truetrue" else False

        elif "." in op:

            self.op = float(op)

        elif op.isnumeric():

            self.op = int(op)

        elif op[0].isalpha():

            if self._finished_rhs:

                if op == 'rhs':
                    self.op = self.rhs
                else:
                    shape = self.rhs_eval.shape if hasattr(self.rhs_eval, 'shape') else ()
                    dtype = self.rhs_eval.dtype if hasattr(self.rhs_eval, 'dtype') else type(self.rhs_eval)
                    self.op = self.backend.add_var(vtype='state_var', name=op, shape=shape, dtype=dtype)
                self.vars[op] = self.op

            else:

                raise ValueError(f"Undefined variable detected in expression: {self.expr_str}. {op} was not found "
                                 f"in the respective arguments dictionary.")

        else:

            raise ValueError(f"Undefined operation detected in expression: {self.expr_str}. {op} cannot be "
                             f"interpreted by this parser.")

        return self.op

    def clear(self):
        """Clears expression list and stack.
        """
        self.expr_list.clear()
        self.expr_stack.clear()

    def _update_lhs(self):
        """Applies update to left-hand side of equation. For differential equations, different solving schemes are
        available.
        """

        # extract update parameters
        solver = self.parser_kwargs.pop('solver', 'euler')
        diff_eq = self._diff_eq

        # update left-hand side of equation
        ###################################

        if diff_eq:

            if solver == 'euler':

                # use explicit forward euler
                self.backend.next_layer()
                op = self.parse(self.expr_stack + ['dt', 'rhs', '*', '+='])
                self.backend.previous_layer()

            else:

                raise ValueError(f'Wrong solver type: {solver}. '
                                 f'Please check the docstring of this function for available solvers.')

        elif self._instantaneous:

            # simple instantaneous update
            op = self.parse(self.expr_stack + ['rhs', self._assign_type])

        else:

            # simple non-instantaneous update
            self.backend.next_layer()
            op = self.parse(self.expr_stack + ['rhs', self._assign_type])
            self.backend.previous_layer()

        return op

    def _preprocess_expr_str(self, expr: str) -> tuple:
        """Turns differential equations into simple algebraic equations using a certain solver scheme and extracts
        left-hand side, right-hand side and update type of the equation.

        Parameters
        ----------
        expr
            Equation in string format.

        Returns
        -------
        tuple
            Contains left hand side, right hand side and left hand side update type
        """

        # collect equation specifics
        ############################

        # split equation into lhs and rhs and assign type
        lhs, rhs, assign_type = split_equation(expr)

        if not assign_type:
            return self._preprocess_expr_str(f"x = {expr}")

        # for the left-hand side, check whether it includes a differential operator
        if "d/dt" in lhs:
            diff_eq = True
            lhs_split = lhs.split('*')
            lhs = "".join(lhs_split[1:])
        elif "'" in lhs:
            diff_eq = True
            lhs = lhs.replace("'", "")
        elif "d" in lhs and "/dt" in lhs:
            diff_eq = True
            lhs = lhs.split('/dt')[0]
            lhs = lhs.replace("d", "", count=1)
        else:
            diff_eq = False

        # get clean name of lhs
        lhs_key = lhs.split('[')[0]
        lhs_key = lhs_key.replace(' ', '')
        lhs = lhs.replace(' ', '')

        # store equation specifics
        if diff_eq and assign_type != '=':
            raise ValueError(f'Wrong assignment method for equation: {expr}. '
                             f'A differential equation cannot be combined with an assign type other than `=`.')

        return lhs, rhs, diff_eq, assign_type, lhs_key

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

    def _apply_idx(self, op: tp.Any, idx: tp.Any, update: tp.Optional[tp.Any] = None,
                   update_type: tp.Optional[str] = None, **kwargs) -> tp.Any:
        """Apply index idx to operation op.

        Parameters
        ----------
        op
            Operation to be indexed.
        idx
            Index to op.
        update
            Update to apply to op at idx.
        update_type
            Type of left-hand side update (e.g. `=` or `+=`).
        kwargs
            Additional keyword arguments to be passed to the indexing functions.

        Returns
        -------
        tp.Any
            Result of applying idx to op.

        """

        kwargs.update(self.parser_kwargs)

        # get constants/variables that are part of the index
        args = []
        if idx in self.vars['idx']:
            idx = self.vars['idx'].pop(idx)
        if type(idx) is str:
            idx_old = idx
            idx = []
            for idx_tmp in idx_old.split(','):
                for idx_tmp2 in idx_tmp.split(':'):
                    idx.append(idx_tmp2)
                    if idx_tmp2 in self.vars['idx']:
                        idx_var = self.vars['idx'].pop(idx_tmp2)
                        if not hasattr(idx_var, 'short_name'):
                            if hasattr(idx_var, 'shape') and tuple(idx_var.shape):
                                idx_var = idx_var[0]
                            idx[-1] = f"{idx_var}"
                        else:
                            idx[-1] = idx_var.short_name
                            args.append(idx_var)
                    idx.append(':')
                idx.pop(-1)
                idx.append(',')
            idx.pop(-1)
            idx = "".join(idx)

        return self.backend.apply_idx(op, idx, update, update_type, *tuple(args))

    def _check_parsed_expr(self, expr_str) -> None:
        """check whether parsing of expression string was successful.

        Parameters
        ----------
        expr_str
            Expression that has been attempted to be parsed.
        """
        expr_str
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

    @staticmethod
    def _compare(x, y):
        """Checks whether x and y are equal or not.
        """
        test = x == y
        if hasattr(test, 'shape'):
            test = test.any()
        return test


def parse_equation_system(equations: list, equation_args: dict, backend: tp.Any, **kwargs) -> dict:
    """Parses a list of equations into the backend.

    Parameters
    ----------
    equations
        Collection of equations that describe the dynamics of the nodes and edges.
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

    solver = kwargs.pop('solver', 'euler')
    update_num = 0

    if solver == 'midpoint':

        print("EXPERIMENTAL FEATURE WARNING: Using the midpoint method for solving differential equations is not fully "
              "tested yet and should be used with caution.")

        # first rhs evaluation
        update_num += 1
        state_vars = {key: var for key, var in equation_args.items()
                      if (type(var) is dict) and ('vtype' in var) and (var['vtype'] == 'state_var')}
        state_vars_orig = dict(state_vars.items())
        equations_tmp, state_vars = update_lhs(deepcopy(equations), state_vars, update_num, state_vars_orig)
        equation_args.update(state_vars)
        updates, equation_args = parse_equations(equations=equations_tmp, equation_args=equation_args, backend=backend,
                                                 **kwargs)
        backend.add_layer()

        # second rhs evaluation + combination of the two
        updates.update({key: arg for key, arg in equation_args.items() if 'inputs' in key})
        equations, _ = update_rhs(equations, updates, update_num, "(var_placeholder + 0.5*update_placeholder)")
        _, equation_args = parse_equations(equations=equations, equation_args=equation_args, backend=backend, **kwargs)

    elif solver == 'rk23':

        print("EXPERIMENTAL FEATURE WARNING: Using the Runge-Kutta 23 algorithm for solving differential equations is "
              "not fully tested yet and should be used with caution.")

        # first rhs evaluation
        update_num += 1
        state_vars = {key: var for key, var in equation_args.items()
                      if (type(var) is dict) and ('vtype' in var) and (var['vtype'] == 'state_var')}
        state_vars_orig = dict(state_vars.items())
        equations_tmp, state_vars = update_lhs(deepcopy(equations), state_vars, update_num, state_vars_orig)
        for var_orig in state_vars_orig.copy().keys():
            if not any([var_orig in var for var in state_vars]):
                state_vars_orig.pop(var_orig)
        equation_args.update(state_vars)
        updates, equation_args = parse_equations(equations=equations_tmp, equation_args=equation_args, backend=backend,
                                                 **kwargs)
        backend.add_layer()

        # second rhs evaluation
        updates.update({key: arg for key, arg in equation_args.items() if 'inputs' in key})
        equations_tmp, updates = update_rhs(deepcopy(equations), updates, update_num,
                                                                       "(var_placeholder + 0.5*update_placeholder)")
        #equation_args = update_equation_args(equation_args, updates)
        state_vars = {key: var for key, var in equation_args.items()
                      if any([orig_key in key for orig_key in state_vars_orig])}
        update_num += 1
        equations_tmp, state_vars = update_lhs(equations_tmp, state_vars, update_num, state_vars_orig)
        equation_args.update(state_vars)
        updates_new, equation_args = parse_equations(equations=equations_tmp, equation_args=equation_args,
                                                     backend=backend, **kwargs)
        backend.add_layer()

        # third rhs evaluation
        updates.update({key: arg for key, arg in equation_args.items() if 'inputs' in key})
        updates.update(updates_new)
        equations, updates = update_rhs(equations, updates, update_num,
                                        "(var_placeholder - var_placeholder_1 + 2*update_placeholder)")
        #equation_args = update_equation_args(equation_args, updates)
        state_vars = {key: var for key, var in equation_args.items()
                      if any([orig_key in key for orig_key in state_vars_orig])}
        update_num += 1
        equations, state_vars = update_lhs(equations, state_vars, update_num, state_vars_orig)
        equation_args.update(state_vars)
        updates, equation_args = parse_equations(equations=equations, equation_args=equation_args, backend=backend,
                                                 **kwargs)
        backend.add_layer()

        # combination of 3 rhs evaluations a'la rk23
        #equation_args = update_equation_args(equation_args, updates)
        equations = []
        for var_info in state_vars_orig:
            node, op, var = var_info.split('/')
            equations.append((f'{var} += ({var}_upd_1 + 4.*{var}_upd_2 + {var}_upd_3)/6.', f"{node}/{op}"))
        _, equation_args = parse_equations(equations=[equations], equation_args=equation_args, backend=backend,
                                           edge_layer=False, **kwargs)

    elif solver == 'euler':

        _, equation_args = parse_equations(equations=equations, equation_args=equation_args, backend=backend, **kwargs)

    else:

        valid_solvers = ['euler', 'midpoint', 'rk23']
        raise ValueError(f'Invalid solver: {solver}. Valid differential equation solvers are: {valid_solvers}.')

    return equation_args


def parse_equations(equations: list, equation_args: dict, backend: tp.Any, **kwargs
                    ) -> tuple:
    """

    Parameters
    ----------
    equations
        List of equations to parse into the backend.
    equation_args
        Dictionary with arguments needed for the equations.
    backend
        Backend instance.

    Returns
    -------
    tuple

    """
    updates = {}

    for layer in equations:

        coupled = is_coupled(layer)

        for eq, scope in layer:

            # parse arguments
            #################

            # extract operator variables from equation args
            op_args = {key.split('/')[-1]: var for key, var in equation_args.items() if scope in key}
            inputs = op_args['inputs'] if 'inputs' in op_args else {}
            unprocessed_inputs = []
            for key, inp in inputs.items():
                inp_tmp = equation_args[inp]
                op_args[key] = inp_tmp
                if type(inp_tmp) is dict and 'vtype' in inp_tmp:
                    unprocessed_inputs.append(key)

            # parse operator variables in backend
            args_tmp = {}
            for key, arg in op_args.items():
                if type(arg) is dict and 'vtype' in arg:
                    args_tmp[key] = arg
            args_tmp = parse_dict(args_tmp, backend, scope=scope, **kwargs)
            op_args.update(args_tmp)
            op_args['dt'] = equation_args['all/all/dt']

            # parse equation
            ################

            diff_eq = is_diff_eq(eq)

            if not diff_eq and not coupled:

                parser = ExpressionParser(expr_str=eq, args=op_args, backend=backend, scope=scope, instantaneous=True,
                                          **kwargs.copy())
                parser.parse_expr()
            else:
                parser = ExpressionParser(expr_str=eq, args=op_args, backend=backend, scope=scope, instantaneous=False,
                                          **kwargs.copy())
                parser.parse_expr()
                if diff_eq:
                    updates[f"{scope}/{parser.lhs_key}"] = parser.vars[parser.lhs_key]

            # udpate equations args
            #######################

            for key, var in parser.vars.items():
                if key != "inputs" and key != "rhs" and key != "dt":
                    equation_args[f"{scope}/{key}"] = var

            for key, inp in inputs.items():
                if key in unprocessed_inputs:
                    equation_args[inp] = parser.vars[key]

        backend.add_layer()

    return updates, equation_args


def update_rhs(equations: list, equation_args: dict, update_num: int, update_str: str) -> tuple:
    """

    Parameters
    ----------
    equations
    equation_args
    update_num
    update_str

    Returns
    -------
    tuple

    """

    # collect variable updates from earlier rhs evaluations
    var_updates = {}
    if update_num > 1:
        for key, arg in equation_args.items():
            node, op, var = key.split('/')
            if f"_upd_{update_num}" in var:
                var = var.replace(f"_upd_{update_num}", "")
            if "_upd_" in var:
                idx = int(var[-1])
                var_tmp = var[:-6]
                if var_tmp in var_updates:
                    var_updates[var_tmp].append((f'var_placeholder_{idx}', var))
                else:
                    var_updates[var_tmp] = [(f'var_placeholder_{idx}', var)]

    # integrate previous rhs evaluations into rhs equations
    updated_args = {}
    while equation_args:

        key, arg = equation_args.popitem()

        if 'inputs' in key:

            for var, arg_tmp in arg.copy().items():
                if f"_upd_{update_num}" in var:
                    var = var.replace(f"_upd_{update_num}", "")
                if "_upd_" not in var:
                    new_var = f"{var}_upd_{update_num}"
                    arg_tmp = arg_tmp.split('/')
                    arg_tmp[-1] = f"{arg_tmp[-1]}_upd_{update_num}"
                    arg_tmp = "/".join(arg_tmp)
                    if arg_tmp in equation_args or arg_tmp in updated_args:
                        arg[new_var] = arg_tmp
                        replace_str = update_str.replace('update_placeholder', new_var)
                        if var in var_updates:
                            for placeholder, var_tmp in var_updates[var]:
                                replace_str = replace_str.replace(placeholder, var_tmp)
                        else:
                            for i in range(1, update_num):
                                replace_str = replace_str.replace(f'var_placeholder_{i}', f'{var}_upd_{i}')
                        replace_str = replace_str.replace('var_placeholder', var)
                        for i, layer in enumerate(equations.copy()):
                            for j, (eq, scope) in enumerate(layer):
                                lhs, rhs, assign = split_equation(eq)
                                if replace_str not in rhs:
                                    rhs = replace(rhs, var, replace_str)
                                equations[i][j] = (f"{lhs} {assign} {rhs}", scope)

        else:

            node, op, var = key.split('/')
            if f"_upd_{update_num}" in var:
                var = var.replace(f"_upd_{update_num}", "")
            if "_upd_" in var:
                updated_args[f"{node}/{op}/{var}"] = arg
            else:
                new_var = f"{var}_upd_{update_num}"
                updated_args[f"{node}/{op}/{new_var}"] = arg
                replace_str = update_str.replace('update_placeholder', new_var)
                if var in var_updates:
                    for placeholder, var_tmp in var_updates[var]:
                        replace_str = replace_str.replace(placeholder, var_tmp)
                replace_str = replace_str.replace('var_placeholder', var)
                for i, layer in enumerate(equations.copy()):
                    for j, (eq, scope) in enumerate(layer):
                        lhs, rhs, assign = split_equation(eq)
                        if replace_str not in rhs:
                            rhs = replace(rhs, var, replace_str)
                        equations[i][j] = (f"{lhs} {assign} {rhs}", scope)

    return equations, updated_args


def update_lhs(equations: list, equation_args: dict, update_num: int, var_dict: dict) -> tuple:
    """

    Parameters
    ----------
    equations
    equation_args
    update_num
    rhs_scalar
    var_dict

    Returns
    -------
    tuple

    """

    updated_args = {}
    while equation_args:
        key, arg = equation_args.popitem()
        node, op, var = key.split('/')
        if "_upd_" in var and f"_upd_{update_num-1}" not in var:
            updated_args[key] = arg
        else:
            var = var.replace(f"_upd_{update_num-1}", "")
            new_var = f"{var}_upd_{update_num}"
            add_to_args = False
            for i, layer in enumerate(equations.copy()):
                for j, (eq, scope) in enumerate(layer):
                    if node in scope and op in scope:
                        lhs, rhs, _ = split_equation(eq)
                        if var in lhs and new_var in replace(lhs, var, new_var):
                            de = False
                            if "d/dt" in lhs:
                                de = True
                                add_to_args = True
                                lhs = lhs.replace("d/dt", "")
                                lhs = lhs.replace("*", "")
                                lhs = lhs.replace(" ", "")
                            elif "'" in lhs:
                                de = True
                                add_to_args = True
                                lhs = lhs.replace("'", "")
                                lhs = lhs.replace(" ", "")
                            if de:
                                lhs = replace(lhs, var, new_var)
                                equations[i][j] = (f"{lhs} = dt * ({rhs})", scope)
            if add_to_args:
                for var_key, var in var_dict.copy().items():
                    if var_key == key or f"{var_key}_upd_" in key:
                        arg = var
                        break
                updated_args[f"{node}/{op}/{new_var}"] = arg

    return equations, updated_args


def update_equation_args(args: dict, updates: dict) -> dict:
    """

    Parameters
    ----------
    args
    updates

    Returns
    -------

    """
    args_new = {}

    for key, arg in args.items():
        if key in updates:
            args_new[key] = updates[key]
        else:
            args_new[key] = arg

    inputs = [key for key in args if 'inputs' in key]
    for inp in inputs:
        for in_key, in_map in args[inp].copy().items():
            for upd in updates:
                if in_map in upd:
                    args_new[inp].update({upd.split('/')[-1]: upd})
                    break

    return args_new


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


def split_equation(expr):
    """

    Parameters
    ----------
    expr

    Returns
    -------

    """
    assign_types = ['+=', '-=', '*=', '/=']
    not_assign_types = ['<=', '>=', '==', '!=']
    lhs, rhs, assign_type, found_assign_type = "", "", "", False
    for assign_type in assign_types:
        if assign_type in expr:
            if f' {assign_type} ' in expr:
                lhs, rhs = expr.split(f' {assign_type} ', maxsplit=1)
            elif f' {assign_type}' in expr:
                lhs, rhs = expr.split(f' {assign_type}', maxsplit=1)
            elif f'{assign_type} ' in expr:
                lhs, rhs = expr.split(f'{assign_type} ', maxsplit=1)
            else:
                lhs, rhs = expr.split(assign_type, maxsplit=1)
            found_assign_type = True
            break
        elif '=' in expr:
            assign_type = '='
            assign = True
            for not_assign_type in not_assign_types:
                if not_assign_type in expr:
                    expr_tmp = expr.replace(not_assign_type, '')
                    if '=' not in expr_tmp:
                        assign = False
            if assign:
                if f' = ' in expr:
                    lhs, rhs = expr.split(f' = ', maxsplit=1)
                elif f' {assign_type}' in expr:
                    lhs, rhs = expr.split(f' =', maxsplit=1)
                elif f'{assign_type} ' in expr:
                    lhs, rhs = expr.split(f'= ', maxsplit=1)
                else:
                    lhs, rhs = expr.split(f"=", maxsplit=1)
                found_assign_type = True
                break

    if not found_assign_type:
        return lhs, rhs, False
    return lhs, rhs, assign_type


def replace(eq: str, term: str, replacement: str, rhs_only=False, lhs_only=False) -> str:
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
        replaced = False
        if (idx_follow_op < len(eq) and eq[idx_follow_op] in allowed_follow_ops) or idx_follow_op == len(eq):
            eq_part = eq[:idx]
            if (rhs_only and "=" in eq_part) or (lhs_only and "=" not in eq_part) or (not rhs_only and not lhs_only):
                eq_new += f"{eq_part} {replacement}"
                replaced = True
        if not replaced:
            eq_new += f"{eq[:idx_follow_op]}"

        # jump to next appearance of term in eq
        eq = eq[idx_follow_op:]
        idx = eq.find(term)

    # add rest of eq to new eq
    eq_new += eq

    return eq_new


def is_diff_eq(eq: str) -> bool:
    """

    Parameters
    ----------
    eq

    Returns
    -------

    """

    lhs, rhs, _ = split_equation(eq)
    if "d/dt" in lhs:
        de = True
    elif "'" in lhs:
        de = True
    elif "dt_test" in replace(rhs, "dt", "dt_test"):
        de = True
    else:
        de = False

    return de


def is_coupled(eqs: list) -> bool:
    """

    Parameters
    ----------
    eqs

    Returns
    -------

    """

    lhs_col, rhs_col, scope_col = [], [], []
    for eq, scope in eqs:
        lhs, rhs, _ = split_equation(eq)
        lhs_col.append(lhs)
        rhs_col.append(rhs)
        scope_col.append(scope)

    for lhs, lhs_scope in zip(lhs_col, scope_col):
        lhs = lhs.replace(" ", "")
        lhs = lhs.replace("d/dt", "")
        lhs = lhs.replace("*", "")
        lhs = lhs.replace("'", "")
        if "[" in lhs:
            l_idx = lhs.index("[")
            r_idx = lhs.index("]")
            lhs = lhs.replace(lhs[l_idx:r_idx+1], "")
        for rhs, rhs_scope in zip(rhs_col, scope_col):
            if "__test_string__" in replace(rhs, lhs, "__test_string__") and rhs_scope == lhs_scope:
                return True

    return False
