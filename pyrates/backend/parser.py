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

# external _imports
import math
import typing as tp
from numbers import Number

import numpy as np
# from pyparsing import Literal, CaselessLiteral, Word, Combine, Optional, \
#     ZeroOrMore, Forward, nums, alphas, ParserElement
from sympy import Expr, Symbol, lambdify, sympify

# pyrates internal _imports
from pyrates.backend.computegraph import ComputeGraph, ComputeNode

# meta infos
__author__ = "Richard Gast"
__status__ = "development"

###############################################
# expression parsers (lhs/rhs of an equation) #
###############################################


class Algebra:

    def __init__(self, **kwargs) -> None:
        """Instantiates expression parser.
        """

        # call super method
        super().__init__()
        self.algebra = kwargs.pop('algebra', None)

        # define algebra
        ################

        # if not self.algebra:
        #
        #     # general symbols
        #     point = Literal(".")
        #     comma = Literal(",")
        #     colon = Literal(":")
        #     e = CaselessLiteral("E")
        #     pi = CaselessLiteral("PI")
        #
        #     # parentheses
        #     par_l = Literal("(")
        #     par_r = Literal(")").setParseAction(self._push_first)
        #     idx_l = Literal("[")
        #     idx_r = Literal("]")
        #
        #     # basic mathematical operations
        #     plus = Literal("+")
        #     minus = Literal("-")
        #     mult = Literal("*")
        #     div = Literal("/")
        #     mod = Literal("%")
        #     dot = Literal("@")
        #     exp_1 = Literal("^")
        #     exp_2 = Combine(mult + mult)
        #     transp = Combine(point + Literal("T"))
        #     inv = Combine(point + Literal("I"))
        #
        #     # numeric types
        #     num_float = Combine(Word("-" + nums, nums) +
        #                         Optional(point + Optional(Word(nums))) +
        #                         Optional(e + Word("-" + nums, nums)))
        #     num_int = Word("-" + nums, nums)
        #
        #     # variables and functions
        #     name = Word(alphas, alphas + nums + "_$")
        #     func_name = Combine(name + par_l, adjacent=True)
        #
        #     # math operation groups
        #     op_add = plus | minus
        #     op_mult = mult | div | dot | mod
        #     op_exp = exp_1 | exp_2 | inv | transp
        #
        #     # logical operations
        #     greater = Literal(">")
        #     less = Literal("<")
        #     equal = Combine(Literal("=") + Literal("="))
        #     unequal = Combine(Literal("!") + Literal("="))
        #     greater_equal = Combine(Literal(">") + Literal("="))
        #     less_equal = Combine(Literal("<") + Literal("="))
        #
        #     # logical operations group
        #     op_logical = greater_equal | less_equal | unequal | equal | less | greater
        #
        #     # pre-allocations
        #     self.algebra = Forward()
        #     exponential = Forward()
        #     index_multiples = Forward()
        #
        #     # basic organization units
        #     index_start = idx_l.setParseAction(self._push_first)
        #     index_end = idx_r.setParseAction(self._push_first)
        #     index_comb = colon.setParseAction(self._push_first)
        #     arg_comb = comma.setParseAction(self._push_first)
        #     arg_tuple = par_l + ZeroOrMore(self.algebra.suppress() + Optional(arg_comb)) + par_r
        #     func_arg = arg_tuple | self.algebra.suppress()
        #
        #     # basic computation unit
        #     atom = (func_name + Optional(func_arg.suppress()) + ZeroOrMore(arg_comb.suppress() + func_arg.suppress()) +
        #             par_r.suppress() | name | pi | e | num_float | num_int).setParseAction(self._push_neg_or_first) | \
        #            (par_l.setParseAction(self._push_last) + self.algebra.suppress() + par_r).setParseAction(self._push_neg)
        #
        #     # apply indexing to atoms
        #     indexed = (Optional(minus) + atom).setParseAction(self._push_neg) + \
        #               ZeroOrMore((index_start + index_multiples + index_end))
        #     index_base = (self.algebra.suppress() | index_comb)
        #     index_full = index_base + ZeroOrMore((index_comb + index_base)) + ZeroOrMore(index_comb)
        #     index_multiples << index_full + ZeroOrMore((arg_comb + index_full))
        #
        #     # hierarchical relationships between mathematical and logical operations
        #     boolean = indexed + Optional((op_logical + indexed).setParseAction(self._push_first))
        #     exponential << boolean + ZeroOrMore((op_exp + Optional(exponential)).setParseAction(self._push_first))
        #     factor = exponential + ZeroOrMore((op_mult + exponential).setParseAction(self._push_first))
        #     expr = factor + ZeroOrMore((op_add + factor).setParseAction(self._push_first))
        #     self.algebra << expr

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
        """Push neg one multipler to expression stack if on first position in toks, else push toks from first-to-last.
        """
        if toks and toks[0] == '-':
            self.expr_stack.append('-one')
        else:
            self.expr_stack.append(toks[0])

    def _push_last(self, strg, loc, toks):
        """Push tokens in last-to-first order to expression stack.
        """
        self.expr_stack.append(toks[-1])


class ExpressionParser:
    """Class for parsing mathematical expressions from a string format into a symbolic representation of the
    mathematical operation expressed by it.

    Parameters
    ----------
    expr_str
        Mathematical expression in string format.
    args
        Dictionary containing all variables and functions needed to evaluate the expression.
    cg
        ComputeGraph instance in which to parse all variables and operations.
        See `pyrates.backend.computegraph.ComputeGraph` for a full documentation of its methods and attributes.
    parsing_method
        Name of the parsing method to use. Valid options: `sympy` for a sympy-based parser, `pyrates` for the
        pyrates-internal, pyparsing-based parser.

    Attributes
    ----------
    lhs
        Boolean, indicates whether expression is left-hand side or right-hand side of an equation
    rhs
        PyRatesOp for the evaluation of the right-hand side of the equation
    vars
        Dictionary containing the variables of an expression
    expr_str
        String representation of the mathematical expression
    expr_stack
        List representation of the syntax tree of the (parsed) mathematical expression.
    parse_func
        Function that will be used to deduce an operation stack/tree from a given expression.
    cg
        Passed ComputeGraph instance into which the expression will be parsed.

    """

    _constant_counter = 0

    def __init__(self, expr_str: str, args: dict, cg: ComputeGraph, def_shape: tp.Optional[tuple] = None,
                 parsing_method: str = 'sympy') -> None:
        """Instantiates expression parser.
        """

        # main attributes
        self.vars = args.copy()
        self.expr_str = expr_str
        self.expr_stack = []
        self.cg = cg
        self._def_shape = def_shape

        # preprocess the expression string
        self.lhs, self.rhs, self._diff_eq, self._assign_type, self.lhs_key = self._preprocess_expr_str(expr_str)

        # define the parsing function
        if parsing_method == 'sympy':
            self.parse_func = sympify
        elif parsing_method == 'pyrates':
            a = Algebra()
            self.parse_func = a.parseString
        else:
            raise ValueError(f'Invalid identifier for the parsing method: {parsing_method}.')

    def parse_expr(self) -> dict:
        """Parses string-based mathematical expression/equation.

        Returns
        -------
        dict
            Variables of the parsed equation.
        """

        # extract symbols and operations from equations right-hand side
        self.expr_stack = self.parse_func(self.rhs)
        if self.expr_stack.is_number:
            c = f"dummy_constant_{self._constant_counter}"
            expr = f"no_op({c})"
            self.vars[c] = {'vtype': 'input',
                            'value': float(self.expr_stack),
                            'shape': ()}
            self.expr_stack = self.parse_func(expr)
            self._constant_counter += 1
        if self.expr_stack.is_symbol:
            self.expr_stack = self.parse_func(f"no_op({self.rhs})")

        # parse rhs into backend
        self.rhs = self._parse_stack(self.expr_stack)

        # extract symbols and operations from left-hand side
        self.expr_stack = self.parse_func(self.lhs)

        # parse lhs into backend
        self._update_lhs()

        return self.vars
    
    def _parse_stack(self, expr: Expr):

        if expr.args:

            # parse variables as nodes into compute graph
            inputs, func_args, old_args = [], [], []
            for arg in self._sort_expr_args(expr.args):

                if isinstance(arg, Symbol):

                    # case I: variables/constants
                    var = self.vars[arg.name]
                    if isinstance(var, ComputeNode):

                        # if parsed already, retrieve label from existing variable
                        label = var.name

                    else:

                        # if not parsed already, parse variable into backend
                        label, var = self.cg.add_var(label=arg.name, def_shape=self._def_shape, **var)
                        self.vars[arg.name] = var

                else:

                    # case II: mathematical expressions
                    label, var = self._parse_stack(arg)

                if isinstance(var, ComputeNode):

                    # store input to mathematical expression, if it is not a simple scalar
                    inputs.append(label)
                    func_args.append(var.symbol)
                    old_args.append(arg)

            # replace names of old expression arguments with new variable symbols
            replacements = {old: new for old, new in zip(old_args, func_args) if old != new}
            if replacements:
                expr = replace_in_expr(expr, replacements)

            # create callable function of the operation
            label = expr.func.__name__
            try:
                v_tmp = self.cg.get_var(func_args[0].name)
                op = self.cg.get_op(label, shape=v_tmp.shape, dtype=v_tmp.dtype)
                backend_funcs = {label: op['func']}
            except (KeyError, IndexError):
                backend_funcs = dict()
            func = lambdify(func_args, expr=expr, modules=[backend_funcs, "numpy"])

            # parse mathematical operation into compute graph
            return self.cg.add_op(inputs, label=label, expr=expr, func=func)

        else:

            # for simple scalar constants, return empty string and dict
            return "", dict()
        
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
            self.vars['x'] = {'vtype': 'variable', 'value': 0.0, 'dtype': 'float', 'shape': ()}
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
        lhs_key = lhs.split('(')[0]
        lhs_key = lhs_key.replace(' ', '')
        lhs = lhs.replace(' ', '')

        # store equation specifics
        if diff_eq and assign_type != '=':
            raise ValueError(f'Wrong assignment method for equation: {expr}. '
                             f'A differential equation cannot be combined with an assign type other than `=`.')

        return lhs, rhs, diff_eq, assign_type, lhs_key
    
    def _update_lhs(self):
        """Applies update to left-hand side of equation. For differential equations, different solving schemes are
        available.
        """

        # update left-hand side of equation
        ###################################

        # receive left-hand side variable information
        lhs_key = self.lhs_key
        if self.expr_stack.is_symbol:

            # retrieve variable information
            v = self.vars[self.lhs_key]

            # create backend state variable if it does not exist already
            if not isinstance(v, ComputeNode):
                _, v = self.cg.add_var(label=lhs_key, def_shape=self._def_shape, **v)

        else:

            # parse left-hand side indexing operation
            lhs_key, v = self._parse_stack(self.expr_stack)

        # create mapping between left-hand side and right-hand side of the equation
        self.cg.add_var_update(v.name, self.rhs[0], differential_equation=self._diff_eq)
        if lhs_key in self.vars:
            self.vars[lhs_key] = v

    @staticmethod
    def _sort_expr_args(args: tuple) -> list:

        # sort arguments from longest to shortest expression
        arg_lengths = [len(str(arg)) for arg in args]
        args_sorted = [args[idx] for idx in np.argsort(arg_lengths)[::-1]]

        # add arguments that need to be treated with priority
        args_final = []
        while args_sorted:

            arg = args_sorted.pop(0)

            # check whether the position between arguments should be swapped
            prioritize = False
            if isinstance(arg, Expr):
                for arg_tmp in args_sorted:
                    if len(arg_tmp.find(arg)):
                        prioritize = True
                        break

            if prioritize:
                idx = args_sorted.index(arg_tmp)
                args_final.append(args_sorted.pop(idx))
            else:
                args_final.append(arg)

        return args_final

################################
# helper classes and functions #
################################


def parse_equations(equations: list, equation_args: dict, cg: ComputeGraph, def_shape: tuple, **kwargs) -> dict:
    """Parses a system (list) of equations into the backend. Transforms differential equations into the appropriate set
    of right-hand side evaluations that can be solved later on.

    Parameters
    ----------
    equations
        Collection of equations that describe the dynamics of the nodes and edges.
    equation_args
        Key-value pairs of arguments needed for parsing the equations.
    cg
        ComputeGraph instance that all equations will be parsed into.
    def_shape
        Default shape of variables that are scalar. Can either be `(1,)` or `()`.
    kwargs
        Additional keyword arguments to be passed to the backend methods.

    Returns
    -------
    dict
        The updated equations args (in-place manipulation of all variables in equation_args happens during
        equation parsing).

    """

    for eq, scope in equations:

        # parse arguments
        #################

        # extract operator variables from equation args
        op_args = {}
        in_vars = []
        update_vars = {}
        for key, var in equation_args.copy().items():

            if scope in key:

                var_name = key.split('/')[-1]

                if var_name == 'inputs':

                    # extract inputs from other variable scopes
                    for in_key, inp in var.items():

                        # check whether input variable has been passed properly
                        if inp not in equation_args:
                            raise KeyError(inp)

                        # extract input variable
                        inp_tmp = equation_args[inp]

                        # add input variable to operator arguments
                        op_args[in_key] = inp_tmp

                        # remember to update the variable entry in the variable collection later
                        update_vars[inp] = in_key

                        # add variable to operator inputs
                        in_vars.append(in_key)

                elif var_name not in in_vars:

                    # change the variable type of input variables that did not receive any extrinsic input
                    try:
                        if var['vtype'] == 'input' and var_name not in equation_args['inputs']:
                            var['vtype'] = 'constant'
                    except KeyError:
                        var['vtype'] = 'constant'
                    except TypeError:
                        pass

                    # include variable information in operator arguments
                    op_args[var_name] = var

                    # remember to update the variable entry in the variable collection later
                    update_vars[key] = var_name

        # parse equation
        ################

        instantaneous = is_diff_eq(eq) is False

        # initialize parser
        parser = ExpressionParser(expr_str=eq, args=op_args, cg=cg, def_shape=def_shape, **kwargs)

        # parse expression into compute graph
        variables = parser.parse_expr()

        # store newly created backend variables
        for full_key, var_key in update_vars.items():
            if full_key in equation_args:
                equation_args[full_key] = variables[var_key]

    return equation_args


def split_equation(expr: str) -> tuple:
    """Splits an equation string into a left-hand side, right-and side and an assign type.

    Parameters
    ----------
    expr
        Equation string. Should contain a left-hand side and a right-hand side, separated by some form of assign symbol.

    Returns
    -------
    tuple
        left-hand side string, right-hand side string, assign operation string.
    """

    # define assign types and explicit non-assign types
    assign_types = ['+=', '-=', '*=', '/=']
    not_assign_types = ['<=', '>=', '==', '!=']

    lhs, rhs, assign_type, found_assign_type = "", "", "", False

    # look for assign types in expression
    for assign_type in assign_types:

        if assign_type in expr:

            # split expression via assign symbol
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

            # assume standard assign
            assign_type = '='
            assign = True

            # check if `=` symbol marks an assign operation or not
            for not_assign_type in not_assign_types:
                if not_assign_type in expr:
                    expr_tmp = expr.replace(not_assign_type, '')
                    if '=' not in expr_tmp:
                        assign = False

            if assign:

                # split equation via `=` symbol
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


def replace(eq: str, term: str, replacement: str, rhs_only: tp.Optional[bool] = False,
            lhs_only: tp.Optional[bool] = False) -> str:
    """Replaces a term in an equation with a replacement term (save replacement).

    Parameters
    ----------
    eq
        Equation that includes the term.
    term
        Term that should be replaced.
    replacement
        Replacement for all occurences of term.
    rhs_only
        If True, replacements will only be performed in right-hand side of the equation.
    lhs_only
        IF True, replacements will only be performed in left-hand side of the equation.

    Returns
    -------
    str
        The updated equation.
    """

    # define follow-up operations/signs that are allowed to follow directly after term in eq
    allowed_follow_ops = '-+=*/^<>=!.%@[]():, '

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
        if ((idx_follow_op < len(eq) and eq[idx_follow_op] in allowed_follow_ops) and
           (idx == 0 or eq[idx-1] in allowed_follow_ops)) or \
                (idx_follow_op == len(eq) and eq[idx-1] in allowed_follow_ops):
            eq_part = eq[:idx]
            if (rhs_only and "=" in eq_part) or (lhs_only and "=" not in eq_part) or (not rhs_only and not lhs_only):
                eq_new += f"{eq_part}{replacement}"
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
    """Checks whether `eq` is a differential equation or not.

    Parameters
    ----------
    eq
        Equation string.

    Returns
    -------
    bool
        True, if `eq` is a differential equation.

    """

    lhs, rhs, _ = split_equation(eq)
    if "d/dt" in lhs:
        de = True
    elif "'" in lhs:
        de = True
    elif "dt_test" in replace(rhs, "step_size", "dt_test"):
        de = True
    else:
        de = False

    return de


def var_in_expression(var: str, expr: str) -> bool:

    if var == expr:
        return True

    # define follow-up operations/signs that are allowed to follow directly after term in eq
    follow_ops = '+-=*/^<>=!.%@[]():, '

    start = 0
    n = len(expr)
    exists_in_expr = False
    while start < n:

        try:

            # find variable in string and extract the follow-up sign
            idx = expr[start:].find(var)
            if idx == -1:
                break
            idx += start
            idx_next = idx + len(var)
            next_sign = expr[idx_next]
            prev_sign = expr[idx - 1]

            # decide whether variable actually exists in expression
            exists_in_expr = ((idx_next < n and next_sign in follow_ops) and
                              (idx == 0 or prev_sign in follow_ops)) or (idx_next == n and prev_sign in follow_ops)

            if exists_in_expr:
                break
            start = idx_next

        except IndexError:

            break

    return exists_in_expr


def extract_var(var: str) -> tuple:
    if "[" in var:
        return var.split("[")[0], True
    return var, False


def get_unique_label(label: str, labels: list) -> str:
    while label in labels:
        try:
            idx = int(label[-1]) + 1
            label = label[:-1] + str(idx)
        except ValueError:
            label = f"{label}0"
    return label


def replace_in_expr(expr: Expr, replacements: dict):
    expr = expr.subs(replacements, simultaneous=True)
    for arg_old in replacements:
        if expr.count(arg_old):
            expr = expr.replace(arg_old, replacements[arg_old])
    return expr
