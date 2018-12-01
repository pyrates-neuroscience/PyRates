"""This module provides parser classes and functions to parse string-based equations into symbolic representations of
operations.
"""

# external imports
from pyparsing import Literal, CaselessLiteral, Word, Combine, Optional, \
    ZeroOrMore, Forward, nums, alphas, ParserElement
from numbers import Number
import math
import tensorflow as tf
import typing as tp
from copy import copy
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

    Methods
    -------
    parse_expr
        Checks whether `expr_str` was successfully parsed into `expr_stack` and translates `expr_stack` into an
        operation `op` representing the evaluation of the full expression.
    parse
        Parses next element of `expr_stack` into a symbolic representation `expr_op` (type of representation depends on
        the functions, operations and data-types defined in `funcs`, `ops` and `dtypes`). Is called by `parse_expr`.
    push_first
        Helper function for building up `expr_stack`.
        Pushes first element of a set of symbolic representations to `expr_stack`.
    push_last
        Helper function for building up `expr_stack`.
        Pushes last element of a set of symbolic representations to `expr_stack`.
    push_negone
        Helper function for building up `expr_stack`.
        Pushes `-1` to `expr_stack`.
    push_all
        Helper function for building up `expr_stack`.
        Pushes all elements of a set of symbolic representations to `expr_stack`.
    push_all_reverse
        Helper function for building up `expr_stack`.
        Pushes all elements of a set of symbolic representations to `expr_stack` in reverse order.

    References
    ----------

    """

    lhs_count = 0

    def __init__(self, expr_str: str, args: dict, backend, lhs: bool = False, solve=False, **kwargs) -> None:
        """Instantiates expression parser.
        """

        # call super init
        #################

        super().__init__()

        # bind attributes to instance
        #############################

        # input arguments
        self.lhs = lhs
        self.args = args
        self.backend = backend
        self.parser_kwargs = kwargs
        self.solve = solve

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
            par_r = Literal(")")
            idx_l = Literal("[")
            idx_r = Literal("]")

            # numeric types
            num_float = Combine(Word("+-" + nums, nums) +
                                Optional(point + Optional(Word(nums))) +
                                Optional(e + Word("+-" + nums, nums)))
            num_int = Word("+-" + nums, nums)

            # variables and functions
            name = Word(alphas, alphas + nums + "_$")
            func_name = Combine(name + par_l, adjacent=True)

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
            index_start = idx_l.setParseAction(self.push_first)
            index_end = idx_r.setParseAction(self.push_first)
            index_comb = colon.setParseAction(self.push_first)
            arg_comb = comma.setParseAction(self.push_first)

            # basic computation unit
            atom = (Optional("-") + (func_name + self.expr.suppress() + ZeroOrMore((arg_comb.suppress() +
                                                                                    self.expr.suppress()))
                                     + par_r | pi | e | name | num_float | num_int
                                     ).setParseAction(self.push_first)
                    ).setParseAction(self.push_negone) | \
                   (par_l.suppress() + self.expr.suppress() + par_r.suppress()).setParseAction(self.push_negone)

            # apply indexing to atoms
            indexed = atom + ZeroOrMore((index_start + index_multiples + index_end))
            index_base = (self.expr.suppress() | index_comb)
            index_full = index_base + ZeroOrMore((index_comb + index_base)) + ZeroOrMore(index_comb)
            index_multiples << index_full + ZeroOrMore((arg_comb + index_full))

            # hierarchical relationships between mathematical and logical operations
            boolean = indexed + Optional((op_logical + indexed).setParseAction(self.push_first))
            exponential << boolean + ZeroOrMore((op_exp + Optional(exponential)).setParseAction(self.push_first))
            factor = exponential + ZeroOrMore((op_mult + exponential).setParseAction(self.push_first))
            self.expr << factor + ZeroOrMore((op_add + factor).setParseAction(self.push_first))

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

    def push_first(self, strg, loc, toks):
        """Push tokens in first-to-last order to expression stack.
        """
        self.expr_stack.append(toks[0])

    def push_negone(self, strg, loc, toks):
        """Push negative one multiplier if on first position in toks.
        """
        if toks and toks[0] == '-':
            self.expr_stack.append('-one')

    def push_all(self, strg, loc, toks):
        """Push all tokens to expression stack at once (first-to-last).
        """
        for t in toks:
            self.expr_stack.append(t)

    def push_all_reverse(self, strg, loc, toks):
        """Push all tokens to expression stack at once (last-to-first).
        """
        for t in range(len(toks)-1, -1, -1):
            self.expr_stack.append(toks[t])

    def push_last(self, strg, loc, toks):
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
        type.Any

        """

        # get next operation from stack
        op = expr_stack.pop()

        # check type of operation
        #########################

        if op == '-one':

            # multiply expression by minus one
            self.expr_op = self.backend.add_op('neg', self.parse(expr_stack), **self.parser_kwargs)

        elif op in "+-**/^@<=>=!==":

            # collect elements to combine
            op2 = self.parse(expr_stack)
            op1 = self.parse(expr_stack)

            # combine elements via mathematical/boolean operator
            self.expr_op = self.broadcast(op, op1, op2, **self.parser_kwargs)

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
                        try:
                            self.args['idx'][f'var_{i}'] = ind.__copy__()
                        except AttributeError:
                            self.args['idx'][f'var_{i}'] = copy(ind)
                        idx += f"var_{i}"
                    i += 1
                idx += ","
            idx = idx[0:-1]

            # extract variable and apply index
            if self.lhs:
                self.lhs = False
                op_to_idx = self.parse(expr_stack)
                self.lhs = True
                self.args['updates'][op] = self.apply_idx(op_to_idx, idx, **self.parser_kwargs)
                self.args['lhs_evals'].append(op)
                self.expr_op = self.args['updates'][op]
            else:
                op_to_idx = self.parse(expr_stack)
                self.expr_op = self.apply_idx(op_to_idx, idx, **self.parser_kwargs)

        elif op == "PI":

            # return float representation of pi
            self.expr_op = math.pi

        elif op == "E":

            # return float representation of e
            self.expr_op = math.e

        elif f'{op}_old' in self.args['inputs'].keys():

            if self.lhs:

                if self.solve:

                    # parse dt
                    self.lhs = False
                    dt = self.parse(['dt'])
                    self.lhs = True

                    # calculate update of differential equation
                    var_update = self.update(self.args['inputs'][f'{op}_old'], self.args.pop('rhs'), dt,
                                             **self.parser_kwargs)
                    self.args['updates'][op] = self.backend.add_op('=', self.args['vars'][op], var_update,
                                                                   **self.parser_kwargs)
                    self.args['lhs_evals'].append(op)
                    self.expr_op = self.args['updates'][op]

                else:

                    # update variable according to rhs
                    self.args['updates'][op] = self.broadcast('=',
                                                              self.args['vars'][op],
                                                              self.args.pop('rhs'),
                                                              **self.parser_kwargs)
                    self.args['lhs_evals'].append(op)
                    self.expr_op = self.args['updates'][op]

            else:

                # extract state variable from previous time-step from args dict
                self.expr_op = self.args['inputs'][f'{op}_old']

        elif op in self.args['inputs'].keys():

            # extract input variable from args dict
            self.expr_op = self.args['inputs'][op]

        elif op in self.args['vars'].keys():

            if self.lhs:

                if self.solve:

                    # parse dt
                    self.lhs = False
                    dt = self.parse(['dt'])
                    self.lhs = True

                    # get variables
                    var = self.args['vars'][op]
                    var_name = f'{op}_old'
                    old_var = self.args['inputs'][var_name]

                    # calculate update of differential equation
                    var_update = self.update(old_var, self.args.pop('rhs'), dt, **self.parser_kwargs)
                    self.args['updates'][op] = self.backend.add_op('=', var, var_update, **self.parser_kwargs)
                    self.args['lhs_evals'].append(op)
                    self.expr_op = self.args['updates'][op]

                else:

                    # update variable according to rhs
                    self.args['updates'][op] = self.broadcast('=',
                                                              self.args['vars'][op],
                                                              self.args.pop('rhs'),
                                                              **self.parser_kwargs)
                    self.args['lhs_evals'].append(op)
                    self.expr_op = self.args['updates'][op]

            else:

                # extract constant/variable from args dict
                self.expr_op = self.args['vars'][op]

        elif any(["float" in op, "bool" in op, "int" in op, "complex" in op]):

            # extract data type
            try:
                self.expr_op = self.backend.add_op('cast', self.parse(expr_stack), op[0:-1], **self.parser_kwargs)
            except AttributeError:
                raise AttributeError(f"Datatype casting error in expression: {self.expr_str}. "
                                     f"{op[0:-1]} is not a valid data-type for this parser.")

        elif op[-1] == "(":

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

        elif any([op == "True", op == "true", op == "False", op == "false"]):

            # return boolean
            self.expr_op = True if op in "Truetrue" else False

        elif "." in op:

            # return float
            i = 0
            while i < 1e7:
                try:
                    arg_tmp = self.backend.add_var(type='constant', name=f'op_{i}', value=float(op), shape=(),
                                                   dtype=self.backend.dtypes['float32'], **self.parser_kwargs)
                    break
                except (ValueError, KeyError) as e:
                    i += 1
            else:
                raise e

            self.expr_op = arg_tmp

        elif op.isnumeric():

            # return integer
            i = 0
            while i < 1e7:
                try:
                    arg_tmp = self.backend.add_var(type='constant', name=f'op_{i}', value=int(op), shape=(),
                                                   dtype=self.backend.dtypes['int32'], **self.parser_kwargs)
                    break
                except (ValueError, KeyError) as e:
                    i += 1
            else:
                raise e

            self.expr_op = arg_tmp

        elif op[0].isalpha():

            if self.lhs:

                # add new variable to arguments that represents rhs op
                rhs = self.args.pop('rhs')
                new_var = self.backend.add_var(type='state_var', name=f'lhs_{self.lhs_count}', value=0.,
                                               shape=rhs.shape, dtype=rhs.dtype, **self.parser_kwargs)
                self.lhs_count += 1
                self.args['vars'][op] = new_var
                self.args['updates'][op] = self.broadcast('=', new_var, rhs, **self.parser_kwargs)
                self.args['lhs_evals'].append(op)
                self.expr_op = self.args['updates'][op]

            else:

                raise ValueError(f"Undefined variable detected in expression: {self.expr_str}. {op} was not found "
                                 f"in the respective arguments dictionary.")

        else:

            raise ValueError(f"Undefined operation detected in expression: {self.expr_str}. {op} cannot be "
                             f"interpreted by this parser.")

        return self.expr_op

    def broadcast(self, op, op1, op2, **kwargs):
        """Tries to match the shapes of arg1 and arg2 such that func can be applied.
        """

        kwargs.update(self.parser_kwargs)

        # get key and value of ops if they are dicts
        if type(op1) is dict:
            (op1_key, op1_val), = op1.items()
        else:
            op1_val = op1
            op1_key = None
        if type(op2) is dict:
            (op2_key, op2_val), = op2.items()
        else:
            op2_val = op2
            op2_key = None

        if not self.compare_shapes(op1_val, op2_val):

            # try removing singleton dimensions from op1/op2
            op1_val_tmp, op2_val_tmp = self.match_shapes(op1_val, op2_val, reduce=True)
            if not self.compare_shapes(op1_val_tmp, op2_val_tmp):
                op1_val_tmp, op2_val_tmp = self.match_shapes(op1_val_tmp, op2_val_tmp, reduce=False)
            if self.compare_shapes(op1_val_tmp, op2_val_tmp):
                op1_val, op2_val = op1_val_tmp, op2_val_tmp

            # if removing the singletons did not work, try vectorizing op1/op2
            else:

                if op1.shape.ndims == 0 or (op1.shape.ndims == 1 and op1.shape.dims[0].value == 0):
                    shape = self.backend.add_op('shape', op1_val, **self.parser_kwargs)
                    dtype = self.backend.add_op('dtype', op2_val, **self.parser_kwargs)
                    op2_val = self.backend.add_op('+', self.backend.add_op('zeros', shape, dtype=dtype), op2_val,
                                                  **self.parser_kwargs)
                elif op2.shape.ndims == 0 or (op2.shape.ndims == 1 and op2.shape.dims[0].value == 0):
                    shape = self.backend.add_op('shape', op2_val, **self.parser_kwargs)
                    dtype = self.backend.add_op('dtype', op1_val, **self.parser_kwargs)
                    op1_val = self.backend.add_op('+', self.backend.add_op('zeros', shape, dtype=dtype), op1_val,
                                                  **self.parser_kwargs)

        try:

            # try applying the operation with matched shapes
            return self.apply_op(op, op1_val, op2_val, op1_key, op2_key, **kwargs)

        except TypeError:

            # try to re-cast the data-types of op1/op2
            try:
                op2_val_tmp = self.backend.add_op('cast', op2_val, op1_val.dtype, **self.parser_kwargs)
                return self.apply_op(op, op1_val, op2_val_tmp, op1_key, op2_key, **kwargs)
            except TypeError:
                op1_val_tmp = self.backend.add_op('cast', op1_val, op2_val.dtype, **self.parser_kwargs)
                return self.apply_op(op, op1_val_tmp, op2_val, op1_key, op2_key, **kwargs)

        except ValueError:

            # try to match additional keyword arguments to shape of op1
            for key, var in kwargs.items():
                if hasattr(var, 'shape'):
                    _, kwargs[key] = self.match_shapes(op1_val, var, reduce=True)
            return self.apply_op(op, op1_val, op2_val, op1_key, op2_key, **kwargs)

    def match_shapes(self, op1, op2, reduce=True):
        """

        Parameters
        ----------
        op1
        op2
        reduce

        Returns
        -------

        """

        if len(op1.shape) > len(op2.shape) and 1 in op1.shape:

            if reduce:

                # cut singleton dimension from op1
                idx = list(op1.shape).index(1)
                op1 = self.backend.add_op('squeeze', op1, idx, **self.parser_kwargs)
            else:

                # reshape op2 to match the shape of op1
                target_shape = op1.shape
                if 1 in target_shape:
                    op2 = self.backend.add_op('reshape', op2, target_shape, **self.parser_kwargs)
                else:
                    idx = list(target_shape).index(op2.shape[0])
                    if idx == 0:
                        op2 = self.backend.add_op('reshape', op2, [1, op1.shape[0]], **self.parser_kwargs)
                    else:
                        op2 = self.backend.add_op('reshape', op2, [op1.shape[1], 1], **self.parser_kwargs)

        elif len(op2.shape) > len(op1.shape) and 1 in op2.shape:

            if reduce:

                # cut singleton dimension from op2
                idx = list(op2.shape).index(1)
                op2 = self.backend.add_op('squeeze', op2, idx, **self.parser_kwargs)

            else:

                # reshape op2 to match the shape of op1
                target_shape = op2.shape
                if 1 in target_shape:
                    op1 = self.backend.add_op('reshape', op1, target_shape, **self.parser_kwargs)
                else:
                    idx = list(target_shape).index(op1.shape[0])
                    if idx == 0:
                        op1 = self.backend.add_op('reshape', op1, [1, target_shape[0]], **self.parser_kwargs)
                    else:
                        op1 = self.backend.add_op('reshape', op1, [target_shape[1], 1], **self.parser_kwargs)

        return op1, op2

    def apply_op(self, op, x, y, x_key=None, y_key=None, **kwargs):
        """

        Parameters
        ----------
        op
        x
        y
        x_key
        y_key
        kwargs

        Returns
        -------

        """

        # collect arguments
        args = []
        if x_key:
            kwargs[x_key] = x
        else:
            args.append(x)
        if y_key:
            kwargs[y_key] = y
        else:
            args.append(y)

        return self.backend.add_op(op, *tuple(args), **kwargs)

    def apply_idx(self, op, idx, **kwargs):
        """Apply index to operation.
        """

        kwargs.update(self.parser_kwargs)

        # do some initial checks
        if self.lhs and self.solve:
            raise ValueError(f'Indexing of differential equations is currently not supported. Please consider '
                             f'changing equation {self.expr_str}.')

        # extract variables from index
        idx_tmp = idx.split(',')
        for i in idx_tmp:
            idx_tmp2 = i.split(':')
            for j in idx_tmp2:
                if j in self.args['idx'].keys():
                    exec(f"{j} = self.backend.add_op('squeeze', self.args['idx'].pop('{j}'), **kwargs)")

        # apply idx
        try:
            op_idx = eval(f'op[{idx}]')
        except ValueError as e:
            if self.lhs:
                op_idx = None
            else:
                try:
                    op_idx = self.backend.add_op('gather', op, eval(f"{idx}"), **kwargs)
                except ValueError:
                    try:
                        op_idx = self.backend.add_op('gather', op,
                                                     self.backend.add_op('squeeze', eval(f"{idx}"), **kwargs), **kwargs)
                    except ValueError as e:
                        raise e
        except TypeError as e:
            if locals()[idx].dtype.is_bool:
                op_idx = self.broadcast('*', op, eval(f"{idx}"), **kwargs)
            else:
                raise TypeError(f'Index is of type {locals()[idx].dtype} that does not match type {op.dtype} of the '
                                f'tensor to be indexed.')

        # return indexed variable
        if self.lhs:
            if op_idx is None:
                return self.backend.add_op('scatter_add', op, eval(f'{idx}'), updates=self.args.pop('rhs'), **kwargs)
            else:
                return self.broadcast('=', op_idx, self.args.pop('rhs'), **kwargs)
        else:
            if op_idx is None:
                raise e
            else:
                return op_idx

    def update(self, var_old, var_delta, dt, **kwargs):
        """Solves single step of a differential equation.
        """
        kwargs.update(self.parser_kwargs)
        var_update = self.broadcast('*', var_delta, dt, **kwargs)
        return self.broadcast('+', var_old, var_update, **kwargs)

    def compare_shapes(self, op1, op2):
        """

        Parameters
        ----------
        op1
        op2

        Returns
        -------

        """

        if hasattr(op1, 'shape') and hasattr(op2, 'shape'):
            if op1.shape == op2.shape:
                return True
            elif len(op1.shape) == 0:
                return True
            elif len(op2.shape) == 0:
                return True
            else:
                return False
        else:
            return True

    def compare_dtypes(self, op1, op2):
        """

        Parameters
        ----------
        op1
        op2

        Returns
        -------

        """

        if op1.dtype == op2.dtype:
            return True
        elif op1.dtype.name in op2.dtype.name:
            return True
        elif op2.dtype.name in op1.dtype.name:
            return True
        elif op1.dtype.base_dtype == op2.dtype.base_dtype:
            return True
        else:
            return False


def parse_equation_list(equations: list, equation_args: dict, backend, **kwargs) -> dict:
    """

    Parameters
    ----------
    equations
    equation_args
    backend
    kwargs

    Returns
    -------

    """

    # preprocess equations and equation arguments
    #############################################

    if 'inputs' not in equation_args:
        equation_args['inputs'] = {}

    left_hand_sides = []
    right_hand_sides = []
    diff_eq = []

    # go through all equations
    for i, eq in enumerate(equations):

        lhs, rhs = eq.split(' = ')

        # for the left-hand side, check whether it includes a differential operator
        if "d/dt" in lhs:
            lhs_split = lhs.split('*')
            lhs = ""
            for lhs_part in lhs_split[1:]:
                lhs += lhs_part
            diff_eq.append(True)
        else:
            diff_eq.append(False)

        # in case of the equations being a differential equation, introduce separate variables for
        # the old and new value of the variable at each update

        # get key of DE variable
        lhs_var = lhs.split('[')[0]
        lhs_var = lhs_var.replace(' ', '')

        for key, var in equation_args['vars'].copy().items():
            if key == lhs_var and '_old' not in key:
                if type(var) is dict:
                    equation_args['inputs'].update(parse_dict({f'{key}_old': var.copy()}, backend=backend, **kwargs))
                else:
                    var_dict = {'vtype': 'state_var', 'dtype': var.dtype.as_numpy_dtype, 'shape': var.shape,
                                'value': 0.}
                    equation_args['inputs'].update(parse_dict({f'{key}_old': var_dict}, backend=backend, **kwargs))

        # store left- and right-hand side of equation
        left_hand_sides.append(lhs)
        right_hand_sides.append(rhs)

    # parse equations
    #################

    for lhs, rhs, solve in zip(left_hand_sides, right_hand_sides, diff_eq):
        equation_args = parse_equation(lhs, rhs, equation_args, backend, solve, **kwargs)

    return equation_args


def parse_equation(lhs: str, rhs: str, equation_args: dict, backend, solve=False, **kwargs) -> dict:
    """Parses lhs and rhs of an equation.

    Parameters
    ----------
    equation
        Mathematical equation in string format.
    equation_args
        Dictionary containing all variables and functions needed to evaluate the expression.
    kwargs

    Returns
    -------
    dict

    Examples
    --------

    References
    ----------

    """

    # parse arguments into correct datatype
    #######################################

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
    lhs_parser = ExpressionParser(expr_str=lhs, args=equation_args, lhs=True, solve=solve, backend=backend, **kwargs)

    return lhs_parser.parse_expr()


def parse_dict(var_dict: dict, backend, **kwargs) -> dict:
    """Parses a dictionary with variable information and creates keras tensorflow variables from that information.

    Parameters
    ----------
    var_dict
        Contains key-value pairs for each variable that should be translated into the tensorflow graph.
        Each value is a dictionary again containing the variable information (needs at least a field for `vtype`).
    backend
    kwargs

    Returns
    -------
    Tuple
        Containing the variables and the variable names.

    """

    var_dict_tf = {}
    tf.keras.backend.manual_variable_initialization(True)

    # go through dictionary items and instantiate variables
    #######################################################

    for var_name, var in var_dict.items():

        # preprocess variable definition
        if var['value'] is None:
            var['value'] = 0.
        init_val = var['value'] if hasattr(var['value'], 'shape') else np.zeros(()) + var['value']
        dtype = getattr(tf, var['dtype']) if type(var['dtype']) is str else var['dtype']
        shape = var['shape'] if 'shape' in var.keys() else init_val.shape

        # instantiate variable
        if var['vtype'] == 'raw':
            var_dict_tf[var_name] = var['value']
        else:
            var_dict_tf[var_name] = backend.add_var(type=var['vtype'],
                                                    name=var_name,
                                                    value=init_val,
                                                    shape=shape,
                                                    dtype=dtype,
                                                    **kwargs)

    return var_dict_tf
