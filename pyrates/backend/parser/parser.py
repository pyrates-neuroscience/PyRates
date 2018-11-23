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
        Parses next element of `expr_stack` into a symbolic representation `_op_tmp` (type of representation depends on
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

        self.wait_for_idx = False
        self.expr_str = expr_str

        # additional attributes
        self.expr = None
        self.expr_stack = []
        self.expr_list = []
        self._op_tmp = None
        self.op = None

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
            exp = Literal("^")
            transp = Combine(point + Literal("T"))
            inv = Combine(point + Literal("I"))

            # math operation groups
            op_add = plus | minus
            op_mult = mult | div | dot | mod
            op_exp = exp | inv | transp

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
            self._op_tmp = self.backend.add_op('neg', self.parse(expr_stack), **self.parser_kwargs)

        elif op in "+-*/^@<=>=!==":

            # collect elements to combine
            op2 = self.parse(expr_stack)
            op1 = self.parse(expr_stack)

            # combine elements via mathematical/boolean operator
            self._op_tmp = self.broadcast(op, op1, op2)

        elif ".T" == op or ".I" == op:

            # transpose/invert expression
            self._op_tmp = self.backend.add_op(op, self.parse(expr_stack), **self.parser_kwargs)

        elif op == "]":

            # parse indices
            indices = []
            while len(expr_stack) > 0 and expr_stack[-1] != "[":
                index = []
                while len(expr_stack) > 0 and expr_stack[-1] not in ",[":
                    if expr_stack[-1] == ":":
                        index.append(expr_stack.pop())
                    else:
                        index.append(self.parse(expr_stack))
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
                op = expr_stack.pop(-1)
                op_to_idx = self.args['vars'][op]
                self.args['updates'][op] = self.apply_idx(op_to_idx, idx)
                self.args['lhs_evals'].append(op)
                self._op_tmp = self.args['updates'][op]
            else:
                op_to_idx = self.parse(expr_stack)
                self._op_tmp = self.apply_idx(op_to_idx, idx)

        elif op == "PI":

            # return float representation of pi
            self._op_tmp = math.pi

        elif op == "E":

            # return float representation of e
            self._op_tmp = math.e

        elif op in self.args['inputs'].keys():

            # extract input variable from args dict
            self._op_tmp = self.args['inputs'][op]

        elif f'{op}_old' in self.args['inputs'].keys():

            if self.lhs:

                # parse dt
                self.lhs = False
                dt = self.parse(['dt'])
                self.lhs = True

                # calculate update of differential equation
                self.args['updates'][op] = self.update(self.args['inputs'][f'{op}_old'],
                                                       self.args.pop('rhs'),
                                                       dt)
                self.args['lhs_evals'].append(op)
                self._op_tmp = self.args['updates'][op]

            else:

                # extract state variable from previous time-step from args dict
                self._op_tmp = self.args['inputs'][f'{op}_old']

        elif op in self.args['updates'].keys():

            # extract state variable from args dict
            self._op_tmp = self.args['updates'][op]
            if op in self.args['lhs_evals']:
                self.args['lhs_evals'].pop(self.args['lhs_evals'].index(op))

        elif op in self.args['vars'].keys():

            if self.lhs:

                if self.solve:

                    # parse dt
                    self.lhs = False
                    dt = self.parse(['dt'])
                    self.lhs = True

                    # get variable
                    var = self.args['vars'][op]

                    # create old var placeholder
                    var_name = f'{op}_old'
                    if var_name not in self.args['inputs'].keys():
                        old_var = parse_dict({var_name: {'vtype': 'state_var',
                                                         'dtype': var.dtype,
                                                         'value': np.zeros(var.shape)}},
                                             self.backend,
                                             **self.parser_kwargs)[var_name]
                    else:
                        old_var = self.args['inputs'][var_name]
                    self.args['inputs'][var_name] = old_var

                    # calculate update of differential equation
                    self.args['updates'][op] = self.update(old_var,
                                                           self.args.pop('rhs'),
                                                           dt)
                    self.args['lhs_evals'].append(op)
                    self._op_tmp = self.args['updates'][op]

                else:

                    # update variable according to rhs
                    self.args['updates'][op] = self.broadcast('=',
                                                              self.args['vars'][op],
                                                              self.args.pop('rhs'))
                    self.args['lhs_evals'].append(op)
                    self._op_tmp = self.args['updates'][op]

            else:

                # extract constant/variable from args dict
                self._op_tmp = op if self.wait_for_idx else self.args['vars'][op]

        elif any(["float" in op, "bool" in op, "int" in op, "complex" in op]):

            # extract data type
            try:
                self._op_tmp = self.backend.add_op('cast', self.parse(expr_stack), op, **self.parser_kwargs)
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
                    self._op_tmp = self.backend.add_op(op[0:-1], args[0], **self.parser_kwargs)
                else:
                    self._op_tmp = self.backend.add_op(op[0:-1], *tuple(args[::-1]), **self.parser_kwargs)
            except KeyError:
                raise KeyError(
                    f"Undefined function in expression: {self.expr_str}. {op[0:-1]} needs to be provided "
                    f"in arguments dictionary.")

        elif any([op == "True", op == "true", op == "False", op == "false"]):

            # return boolean
            self._op_tmp = True if op in "Truetrue" else False

        elif "." in op:

            # return float
            i = 0
            while True:
                try:
                    arg_tmp = parse_dict({f'op_{i}': {'vtype': 'constant',
                                                      'dtype': 'float32',
                                                      'shape': (),
                                                      'value': float(op)}},
                                         self.backend,
                                         **self.parser_kwargs)
                    break
                except (ValueError, KeyError):
                    i += 1

            self._op_tmp = arg_tmp[f'op_{i}']

        elif op.isnumeric():

            # return integer
            i = 0
            while True:
                try:
                    arg_tmp = parse_dict({f'op_{i}': {'vtype': 'constant',
                                                      'dtype': 'int32',
                                                      'shape': (),
                                                      'value': int(op)}},
                                         self.backend,
                                         **self.parser_kwargs)
                    break
                except (ValueError, KeyError):
                    i += 1

            self._op_tmp = arg_tmp[f'op_{i}']

        elif op[0].isalpha():

            if self.lhs:

                # add new variable to arguments that represents rhs op
                self.args['updates'][op] = self.args.pop('rhs')
                self.args['lhs_evals'].append(op)
                self._op_tmp = self.args['updates'][op]

            else:

                raise ValueError(f"Undefined variable detected in expression: {self.expr_str}. {op} was not found "
                                 f"in the respective arguments dictionary.")

        else:

            raise ValueError(f"Undefined operation detected in expression: {self.expr_str}. {op} cannot be "
                             f"interpreted by this parser.")

        return self._op_tmp

    def broadcast(self, op, op1, op2, **kwargs):
        """Tries to match the shapes of arg1 and arg2 such that func can be applied.
        """

        if 'scope' in self.parser_kwargs.keys():
            kwargs['scope'] = self.parser_kwargs['scope']

        try:
            # no broadcasting
            args = []
            if type(op1) is dict:
                (op1_key, op1_val), = op1.items()
                kwargs[op1_key] = op1_val
            else:
                args.append(op1)
            if type(op2) is dict:
                (op2_key, op2_val), = op2.items()
                kwargs[op2_key] = op2_val
            else:
                args.append(op2)
            return self.backend.add_op(op, *tuple(args), **kwargs)

        # try to broadcast arg1 and arg22 to the same shape
        except (ValueError, KeyError):

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

            try:

                # add singleton dimension to either op1 or op2
                if len(op1_val.shape) > len(op2_val.shape) and 1 in op1_val.shape:
                    target_shape = op1_val.shape
                    if 1 in target_shape:
                        op2_val = self.backend.add_op('reshape', op2_val, target_shape)
                    else:
                        idx = list(target_shape).index(op2_val.shape[0])
                        if idx == 0:
                            op2_val = self.backend.add_op('reshape', op2_val, [1, op1_val.shape[0]])
                        else:
                            op2_val = self.backend.add_op('reshape', op2_val, [op1_val.shape[1], 1])
                elif len(op2_val.shape) > len(op1_val.shape) and 1 in op2_val.shape:
                    if op == '=':
                        op2_val = self.backend.add_op('squeeze', op2_val, -1)
                    else:
                        target_shape = op2_val.shape
                        if 1 in target_shape:
                            op1_val = self.backend.add_op('reshape', op1_val, target_shape)
                        else:
                            idx = list(target_shape).index(op1_val.shape[0])
                            if idx == 0:
                                op1_val = self.backend.add_op('reshape', op1_val, [1, target_shape[0]])
                            else:
                                op1_val = self.backend.add_op('reshape', op1_val, [target_shape[1], 1])

                # try to apply function after singleton addition
                args = []
                if op1_key:
                    kwargs[op1_key] = op1_val
                else:
                    args.append(op1_val)
                if op2_key:
                    kwargs[op2_key] = op2_val
                else:
                    args.append(op2_val)
                return self.backend.add_op(op, *tuple(args), **kwargs)

            except (ValueError, KeyError):

                # transform op1 or op2 from scalar to array
                if hasattr(op1, 'shape'):
                    shape = self.backend.add_op('shape', op1_val)
                    dtype = self.backend.add_op('dtype', op2_val)
                    op2_val = self.backend.add_op('+', self.backend.add_op('zeros', shape, dtype=dtype), op2_val)
                else:
                    shape = self.backend.add_op('shape', op2_val)
                    dtype = self.backend.add_op('dtype', op1_val)
                    op1_val = self.backend.add_op('+', self.backend.add_op('zeros', shape, dtype=dtype), op1_val)

                # try to apply function after vectorization
                args = []
                if op1_key:
                    kwargs[op1_key] = op1_val
                else:
                    args.append(op1_val)
                if op2_key:
                    kwargs[op2_key] = op2_val
                else:
                    args.append(op2_val)
                return self.backend.add_op(op, *tuple(args), **kwargs)

    def apply_idx(self, op, idx):
        """Apply index to operation.
        """

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
                    exec(f"{j} = self.backend.add_op('squeeze', self.args['idx'].pop('{j}'))")

        # apply idx
        try:
            op_idx = eval(f'op[{idx}]')
        except ValueError:
            try:
                op_idx = eval(f"self.backend.add_op('gather', op, {idx})")
            except ValueError:
                op_idx = eval(f"self.backend.add_op('gather', op, self.funcs['squeeze']({idx}))")
        except TypeError:
            if locals()[idx].dtype.is_bool:
                op_idx = self.broadcast('*', op, eval(f"self.backend.add_op('cast', {idx}, op.dtype)"))
            else:
                raise TypeError(f'Index is of type {locals()[idx].dtype} that does not match type {op.dtype} of the '
                                f'tensor to be indexed.')

        # return indexed variable
        if self.lhs:
            return self.broadcast('=', op_idx, self.args.pop('rhs'))
        else:
            return op_idx

    def update(self, var_old, var_delta, dt):
        """Solves single step of a differential equation.
        """

        var_update = self.broadcast('*', var_delta, dt)
        return self.broadcast('+', var_old, var_update)


class KerasExpressionParser(ExpressionParser):
    """Expression parser that transforms expression into keras operations on a tensorflow graph.

    Parameters
    ----------
    expr_str
        See docstring of `ExpressionParser`.
    args
        See docstring of `ExpressionParser`. Each variable in args needs to be a dictionary with key-value pairs for:
            - `var`: contains the tensorflow variable.
            - `dependency`: Boolean. If True, the expression needs to wait for this variable to be calculated/updated
               before being evaluated.
    lhs
        See docstring of `ExpressionParser`.
    tf_graph
        Instance of `tensorflow.Graph`. Mathematical expression will be parsed into this graph.

    Attributes
    ----------
    tf_graph
        Instance of `tensorflow.Graph` containing a graph-representation of `expr_str`.
    ops
        Dictionary containing all mathematical operations available for this parser and their syntax. These include:
            - addition: `+`
            - subtraction: `-`
            - multiplication: `*`
            - division: `/`
            - modulo: `%`
            - exponentiation: `^`
            - matrix multiplication: `@`
            - matrix transposition: `.T`
            - matrix inversion: `.I`
            - logical greater than: `>`
            - logical less than: `<`
            - logical equal: `==`
            - logical unequal: `!=`
            - logical greater or equal: `>=`
            - logical smaller or equal: `<=`
    funcs
        Dicionary containing all additional functions available for this parser and their syntax. These include:
            - sinus: `sin()`.
            - cosinus: `cos()`.
            - tangens: `tan()`.
            - absolute: `abs()`.
            - maximum: `max()`
            - minimum: `min()`
            - index of maximum: `argmax()`
            - index of minimum: `argmin()`
            - round to next integer: `round()`. Tensorflow name: `tensorflow.to_int32()`.
            - round to certain decimal point: `roundto()`. Custom function using `tensorflow.round()`. Defined in
              `pyrates.parser.parser.py`.
            - sum over dimension(s): `sum()`. Tensorflow name: `reduce_sum()`.
            - Concatenate multiples of tensor over certain dimension: `tile()`.
            - Reshape tensor: `reshape()`.
            - Cut away dimensions of size 1: `squeeze()`.
            - Cast variable to data-type: `cast()`.
            - draw random variable from standard normal distribution: `randn()`.
              Tensorflow name: `tensorflow.random_normal`.
            - Create array filled with ones: `ones()`.
            - Create array filled with zeros: `zeros()`.
            - Apply softmax function to variable: `softmax()`. Tensorflow name: `tensorflow.nn.softmax()`.
            - Apply boolean mask to array: `boolean_mask()`.
            - Create new array with non-zero entries at certain indices: `scatter()`.
              Tensorflow name: `tensorflow.scatter_nd`
            - Add values to certain entries of tensor: 'scatter_add()'. Tensorflow name: `tensorflow.scatter_nd_add`.
            - Update values of certain tensor entries: `scatter_update()`.
              Tensorflow name: `tensorflow.scatter_nd_update`.
            - Apply tensor as index to other tensor: `array_idx()`. Tensorflow name: `tensorflow.gather_nd`.
            - Get variable from tensorflow graph or create new variable: `new_var()`:
              Tensorflow name: `tensorflow.get_variable`.
        For a detailed documentation of how to use these functions, see the tensorflow Python API.
    dtypes
        Dictionary containing all data-types available for this parser. These include:
            - float16, float32, float64
            - int16, int32, int64
            - uint16, uint32, uint64
            - complex64, complex128,
            - bool
        All of those data-types can be used inside a mathematical expression instead of using `cast()`
        (e.g. `int32(3.631)`.
    For all other attributes, see docstring of `ExpressionParser`.

    Methods
    -------
    See docstrings of `ExpressionParser` methods.

    Examples
    --------

    References
    ----------

    """

    def __init__(self, expr_str: str, args: dict, backend: tf.keras.layers.Layer, lhs: bool = False) -> None:
        """Instantiates keras expression parser.
        """

        # call super init
        #################

        super().__init__(expr_str=expr_str, args=args, backend=backend, lhs=lhs)

        # define operations and functions
        #################################

        # base math operations
        ops = {"+": tf.keras.layers.Lambda(lambda x: x[0] + x[1]),
               "-": tf.keras.layers.Lambda(lambda x: x[0] - x[1]),
               "*": tf.keras.layers.Lambda(lambda x: x[0] * x[1]),
               "/": tf.keras.layers.Lambda(lambda x: x[0] / x[1]),
               "%": tf.keras.layers.Lambda(lambda x: x[0] % x[1]),
               "^": tf.keras.layers.Lambda(lambda x: tf.keras.backend.pow(x[0], x[1])),
               "@": tf.keras.layers.Lambda(lambda x: tf.keras.backend.dot(x[0], x[1])),
               ".T": tf.keras.layers.Lambda(lambda x: tf.keras.backend.transpose(x)),
               ".I": tf.keras.layers.Lambda(lambda x: tf.matrix_inverse(x)),
               ">": tf.keras.layers.Lambda(lambda x: tf.keras.backend.greater(x[0], x[1])),
               "<": tf.keras.layers.Lambda(lambda x: tf.keras.backend.less(x[0], x[1])),
               "==": tf.keras.layers.Lambda(lambda x: tf.keras.backend.equal(x[0], x[1])),
               "!=": tf.keras.layers.Lambda(lambda x: tf.keras.backend.not_equal(x[0], x[1])),
               ">=": tf.keras.layers.Lambda(lambda x: tf.keras.backend.greater_equal(x[0], x[1])),
               "<=": tf.keras.layers.Lambda(lambda x: tf.keras.backend.less_equal(x[0], x[1])),
               "=": tf.keras.backend.update
               }
        self.ops.update(ops)

        # additional functions
        funcs = {"sin": tf.keras.layers.Lambda(lambda x: tf.keras.backend.sin(x)),
                 "cos": tf.keras.layers.Lambda(lambda x: tf.keras.backend.cos(x)),
                 "tanh": tf.keras.layers.Lambda(lambda x: tf.keras.backend.tanh(x)),
                 "abs": tf.keras.layers.Lambda(lambda x: tf.keras.backend.abs(x)),
                 "sqrt": tf.keras.layers.Lambda(lambda x: tf.keras.backend.sqrt(x)),
                 "sq": tf.keras.layers.Lambda(lambda x: tf.keras.backend.square(x)),
                 "exp": tf.keras.layers.Lambda(lambda x: tf.keras.backend.exp(x)),
                 "max": tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x)),
                 "min": tf.keras.layers.Lambda(lambda x: tf.keras.backend.min(x)),
                 "argmax": tf.keras.layers.Lambda(lambda x: tf.keras.backend.argmax(x)),
                 "argmin": tf.keras.layers.Lambda(lambda x: tf.keras.backend.argmin(x)),
                 "round": tf.keras.layers.Lambda(lambda x: tf.keras.backend.round(x)),
                 "roundto": tf.keras.layers.Lambda(lambda x: tf.keras.backend.round(x[0] * 10**x[1]) / 10**x[1]),
                 "sum": tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x[0], *x[1])
                                               if type(x) is list else tf.keras.backend.sum(x)),
                 "concat": tf.keras.layers.Lambda(lambda x: tf.keras.backend.concatenate(x[0], *x[1])
                                                  if type(x[0]) is list else tf.keras.backend.concatenate(x)),
                 "reshape": tf.keras.layers.Lambda(lambda x: tf.keras.backend.reshape(x[0], x[1])
                                                   if type(x) is list else tf.keras.backend.reshape(x)),
                 "shape": tf.keras.backend.shape,
                 "dtype": tf.keras.backend.dtype,
                 'squeeze': tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(x[0], x[1])
                                                   if type(x) is list else tf.keras.backend.squeeze(x[0], -1)),
                 "cast": tf.keras.layers.Lambda(lambda x: tf.keras.backend.cast(x[0], x[1])),
                 "randn": tf.keras.layers.Lambda(lambda x: tf.keras.backend.random_normal(x[0], *x[1])
                                                 if "Tensor" in str(type(x[0]))
                                                 else tf.keras.backend.random_normal(x)),
                 "ones": tf.keras.layers.Lambda(lambda x: tf.keras.backend.ones(x[0], x[1])
                                                if "Tensor" in str(type(x[0]))
                                                else tf.keras.backend.ones(x)),
                 "zeros": tf.keras.layers.Lambda(lambda x: tf.keras.backend.zeros(x[0], x[1])
                                                 if "Tensor" in str(type(x[0]))
                                                 else tf.keras.backend.zeros(x)),
                 "softmax": tf.keras.layers.Lambda(lambda x: tf.keras.activations.softmax(x[0], *x[1])
                                                   if type(x[0]) is list else tf.keras.activations.softmax(x)),
                 "gather": tf.keras.layers.Lambda(lambda x: tf.gather_nd(x[0], x[1])),
                 "mask": tf.keras.layers.Masking,
                 "lambda": tf.keras.layers.Lambda
                 }
        self.funcs.update(funcs)

        dtypes = {"float16": tf.float16,
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
        self.dtypes.update(dtypes)


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
        if diff_eq[-1]:

            # get key of DE variable
            lhs_var = lhs.split('[')[0]
            lhs_var = lhs_var.replace(' ', '')

            for key, var in equation_args['vars'].items():
                if key == lhs_var:
                    equation_args['inputs'].update(parse_dict({f'{key}_old': var.copy()}, backend=backend))

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

        # make sure that value of variable is a number
        if var['value'] is None:
            var['value'] = 0.
        init_val = var['value'] if hasattr(var['value'], 'shape') else np.zeros(()) + var['value']
        dtype = getattr(tf, var['dtype']) if type(var['dtype']) is str else var['dtype']
        shape = var['shape'] if 'shape' in var.keys() else init_val.shape

        if var['vtype'] == 'raw':

            # just extract raw variable value
            tf_var = var['value']

        elif var['vtype'] == 'state_var':

            # create a tensorflow variable that can change its value over the course of a simulation
            tf_var = backend.add_variable(value=init_val,
                                          name=var_name,
                                          dtype=dtype,
                                          shape=shape,
                                          **kwargs)

        elif var['vtype'] == 'constant':

            # create dense, constant tensor
            tf_var = backend.add_constant(value=init_val,
                                          name=var_name,
                                          shape=shape,
                                          dtype=dtype,
                                          **kwargs
                                          )

        elif var['vtype'] == 'placeholder':

            tf_var = backend.add_placeholder(name=var_name,
                                             shape=shape,
                                             dtype=dtype,
                                             **kwargs
                                             )

        else:

            raise ValueError('Variable type must be `raw`, `state_variable`, `constant` or `placeholder`.')

        var_dict_tf[var_name] = tf_var

    return var_dict_tf
