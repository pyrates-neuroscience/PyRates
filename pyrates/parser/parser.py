"""This module provides parser classes and functions to parse string-based equations into operations.
"""

# external imports
from pyparsing import Literal, CaselessLiteral, Word, Combine, Optional, \
    ZeroOrMore, Forward, nums, alphas, ParserElement
from numbers import Number
import math
import tensorflow as tf
import typing as tp

# pyrates internal imports

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


# expression parsers (lhs/rhs of an equation)
#############################################


class ExpressionParser(ParserElement):
    """Base class for parsing mathematical expressions.

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

    Examples
    --------

    References
    ----------

    """

    def __init__(self, expr_str: str, args: dict, lhs: bool = False) -> None:
        """Instantiates expression parser.
        """

        # call super init
        #################

        super().__init__()

        # bind input args to instance
        #############################

        self.lhs = lhs
        self.args = args

        # if left-hand side of an equation, check whether it includes a differential operator
        if "d/dt" in expr_str:
            lhs_split = expr_str.split('*')
            expr_str = ""
            for lhs_part in lhs_split[1:]:
                expr_str += lhs_part
            self.solve = True
        else:
            self.solve = False

        self.expr_str = expr_str

        # additional attributes
        #######################

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

        # define operations and functions
        #################################

        # base math operations
        self.ops = {}
        self.sparse_ops = {}

        # additional functions
        self.funcs = {}
        self.sparse_funcs = {}

        # allowed data-types
        self.dtypes = {}

        # add functions from args dictionary, if passed
        for key, val in self.args.items():
            if callable(val):
                self.funcs[key] = val

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
        op = self.parse(self.expr_stack[:])

        # extract update operation
        update = self.args.pop('rhs')['var'] if 'rhs' in self.args.keys() else None

        return op, update, self.solve

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
            self._op_tmp = -self.parse(expr_stack)

        elif op in "+-*/^@<=>=!==":

            # collect elements to combine
            op2 = self.parse(expr_stack)
            op1 = self.parse(expr_stack)

            # enable automatic broadcasting
            if hasattr(op1, 'shape') and hasattr(op2, 'shape') and \
                    op1.shape != op2.shape and len(op1.shape) > 0 and len(op2.shape) > 0:
                if len(op1.shape) > len(op2.shape):
                    target_shape = op1.shape
                    if 1 in target_shape:
                        op2 = self.funcs['reshape'](op2, target_shape)
                    else:
                        idx = list(target_shape).index(op2.shape[0])
                        if idx == 0:
                            op2 = self.funcs['reshape'](op2, [1, op1.shape[0]])
                        else:
                            op2 = self.funcs['reshape'](op2, [op1.shape[1], 1])
                elif len(op2.shape) > len(op1.shape) and 1 in op2.shape:
                    target_shape = op2.shape
                    if 1 in target_shape:
                        op1 = self.funcs['reshape'](op1, target_shape)
                    else:
                        idx = list(target_shape).index(op1.shape[0])
                        if idx == 0:
                            op1 = self.funcs['reshape'](op1, [1, target_shape[0]])
                        else:
                            op1 = self.funcs['reshape'](op1, [target_shape[1], 1])
            elif hasattr(op1, 'dense_shape') and hasattr(op2, 'shape'):
                if len(op2.shape) == 1:
                    op2 = self.funcs['reshape'](op2, list(op2.shape) + [1])
            elif hasattr(op1, 'shape'):
                op2 = self.funcs['zeros'](op1.shape) + op2
            elif hasattr(op2, 'shape'):
                op1 = self.funcs['zeros'](op2.shape) + op1

            # combine elements via mathematical/boolean operator
            try:
                self._op_tmp = self.ops[op](op1, op2)
            except TypeError:
                self._op_tmp = self.sparse_ops[op](op1, op2)

        elif op in ".T.I":

            # transpose/invert expression
            try:
                self._op_tmp = self.ops[op](self.parse(expr_stack))
            except TypeError:
                self._op_tmp = self.sparse_ops[op](self.parse(expr_stack))

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
            idx = ""
            for index in indices[::-1]:
                for i, ind in enumerate(index):
                    if type(ind) == str:
                        idx += ind
                    elif isinstance(ind, Number):
                        idx += f"{ind}"
                    else:
                        try:
                            exec(f"var_{i} = ind.__copy__()")
                        except AttributeError:
                            exec(f"var_{i} = copy(ind)")
                        idx += f"var_{i}"
                idx += ","
            idx = idx[0:-1]

            # apply idx to op
            op_to_idx = self.parse(expr_stack)
            try:

                # standard indexing
                self._op_tmp = eval(f"op_to_idx[{idx}]")

            except (TypeError, ValueError):

                if self.lhs:

                    if self.solve:

                        # perform differential equation update for indexed variable
                        update = self.args['dt']['var'] * self.args.pop('rhs')['var']
                        self._op_tmp = eval(f"self.funcs['scatter_add'](op_to_idx, {idx}, update)")

                    else:

                        # perform variable update for indexed variable
                        update = self.args.pop('rhs')['var']
                        self._op_tmp = eval(f"self.funcs['scatter_update'](op_to_idx, self.funcs['squeeze']({idx}), "
                                            f"self.funcs['squeeze'](update))")

                else:

                    try:

                        # indexing by list of indices
                        self._op_tmp = eval(f"self.funcs['array_idx'](op_to_idx, {idx})")

                    except (TypeError, ValueError):

                        # indexing via boolean array
                        self._op_tmp = eval(f"self.funcs['boolean_mask'](op_to_idx, {idx})")

        elif op == "PI":

            # return float representation of pi
            self._op_tmp = math.pi

        elif op == "E":

            # return float representation of e
            self._op_tmp = math.e

        elif op in self.args.keys():

            # extract variable from args dict
            self._op_tmp = self.args[op]['var']

        elif any(["float" in op, "bool" in op, "int" in op, "complex" in op]):

            # extract data type
            try:
                dtype = self.dtypes[op[0:-1]]
            except AttributeError:
                raise ValueError(f"Datatype casting error in expression: {self.expr_str}. {op[0:-1]} is not a valid "
                                 f"data-type for this parser.")

            # cast new data type to argument
            self._op_tmp = self.funcs['cast'](self.parse(expr_stack), dtype)

        elif op[-1] == "(":

            # extract function
            try:
                f = self.funcs[op[0:-1]]
            except TypeError:
                try:
                    f = self.sparse_funcs[op[0:-1]]
                except KeyError:
                    raise KeyError(
                        f"Undefined function in expression: {self.expr_str}. {op[0:-1]} needs to be provided "
                        f"in arguments dictionary.")
            except KeyError:
                raise KeyError(f"Undefined function in expression: {self.expr_str}. {op[0:-1]} needs to be provided "
                               f"in arguments dictionary.")

            # parse arguments
            args = []
            while len(expr_stack) > 0:
                args.append(self.parse(expr_stack))
                if len(expr_stack) == 0 or expr_stack[-1] != ",":
                    break
                else:
                    expr_stack.pop()

            # apply function to arguments
            self._op_tmp = f(*tuple(args[::-1]))

        elif any([op == "True", op == "true", op == "False", op == "false"]):

            # return boolean
            self._op_tmp = True if op in "Truetrue" else False

        elif "." in op:

            # return float
            self._op_tmp = float(op)

        elif op.isnumeric():

            # return integer
            self._op_tmp = int(op)

        elif op[0].isalpha():

            if self.lhs:

                op_tmp = self.args.pop('rhs')
                self._op_tmp = op_tmp['var']
                self.args[op] = op_tmp

            else:

                raise ValueError(f"Undefined variable detected in expression: {self.expr_str}. {op} was not found "
                                 f"in the respective arguments dictionary.")

        else:

            raise ValueError(f"Undefined operation detected in expression: {self.expr_str}. {op} cannot be "
                             f"interpreted by this parser.")

        return self._op_tmp


class TFExpressionParser(ExpressionParser):
    """Expression parser that transforms expression into tensorflow operations on a tensorflow graph.

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

    def __init__(self, expr_str: str, args: dict, lhs: bool = False,
                 tf_graph: tp.Optional[tf.Graph] = None) -> None:
        """Instantiates tensorflow expression parser.
        """

        # call super init
        #################

        super().__init__(expr_str=expr_str, args=args, lhs=lhs)

        # define tensorflow graph on which to create the operations
        ###########################################################

        self.tf_graph = tf_graph if tf_graph else tf.get_default_graph()

        # define operations and functions
        #################################

        # base math operations
        ops = {"+": tf.add,
               "-": tf.subtract,
               "*": tf.multiply,
               "/": tf.truediv,
               "%": tf.mod,
               "^": tf.pow,
               "@": tf.matmul,
               ".T": tf.matrix_transpose,
               ".I": tf.matrix_inverse,
               ">": tf.greater,
               "<": tf.less,
               "==": tf.equal,
               "!=": tf.not_equal,
               ">=": tf.greater_equal,
               "<=": tf.less_equal
               }
        for key, val in ops.items():
            self.ops[key] = val

        # additional functions
        funcs = {"sin": tf.sin,
                 "cos": tf.cos,
                 "tan": tf.tan,
                 "abs": tf.abs,
                 "sqrt": tf.sqrt,
                 "max": tf.reduce_max,
                 "min": tf.reduce_min,
                 "argmax": tf.argmax,
                 "argmin": tf.argmin,
                 "round": tf.to_int32,
                 "roundto": round_to_prec,
                 "sum": tf.reduce_sum,
                 "tile": tf.tile,
                 "reshape": tf.reshape,
                 'squeeze': tf.squeeze,
                 "cast": tf.cast,
                 "randn": tf.random_normal,
                 "ones": tf.ones,
                 "zeros": tf.zeros,
                 "softmax": tf.nn.softmax,
                 "boolean_mask": tf.boolean_mask,
                 "scatter": tf.scatter_nd,
                 "scatter_add": tf.scatter_nd_add,
                 "scatter_update": tf.scatter_nd_update,
                 "array_idx": tf.gather_nd,
                 "new_var": tf.get_variable
                 }
        for key, val in funcs.items():
            self.funcs[key] = val

        # counterparts for sparse tensors
        self.sparse_ops = {"+": tf.sparse_add,
                           ".T": tf.sparse_transpose,
                           "@": tf.sparse_tensor_dense_matmul}
        self.sparse_funcs = {"max": tf.sparse_reduce_max,
                             "sum": tf.sparse_reduce_sum,
                             "reshape": tf.sparse_reshape,
                             "softmax": tf.sparse_softmax,
                             "boolean_mask": tf.boolean_mask,
                             "array_idx": tf.sparse_mask
                             }

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
        for key, val in dtypes.items():
            self.dtypes[key] = val

    def parse(self, expr_stack: list) -> tp.Union[tf.Operation, tf.Tensor, tf.Variable, float, int]:
        """Parses string-based expression.
        """

        with self.tf_graph.as_default():

            # set dependencies and check for sparse tensors
            ###############################################

            # set dependencies
            dependencies = []
            for arg in expr_stack:
                if arg in self.args.keys() and self.args[arg]['dependency']:
                    dependencies += self.args[arg]['op'] if type(self.args[arg]['op']) is list else \
                        [self.args[arg]['op']]

            # create tensorflow operation/variable
            with tf.control_dependencies(dependencies):
                self._op_tmp = super().parse(expr_stack=expr_stack)

            return self._op_tmp


def parse_equation(equation: str, args: dict, tf_graph: tp.Optional[tf.Graph] = None
                   ) -> tuple:
    """Parses lhs and rhs of an equation.

    Parameters
    ----------
    equation
        Mathematical equation in string format.
    args
        Dictionary containing all variables and functions needed to evaluate the expression.
    tf_graph
        Tensorflow graph on which all operations will be created.

    Returns
    -------
    tuple

    Examples
    --------

    References
    ----------

    """

    lhs, rhs = equation.split(' = ')

    tf_graph = tf_graph if tf_graph else tf.get_default_graph()

    with tf_graph.as_default():

        # parse rhs
        rhs_parser = TFExpressionParser(expr_str=rhs, args=args, tf_graph=tf_graph)
        rhs_op = rhs_parser.parse_expr()

        # handle rhs evaluation
        if rhs_op[1] is None:
            rhs_op = rhs_op[0]
        else:
            rhs_op = rhs_op[0].assign(rhs_op[1])
        args['rhs'] = {'var': rhs_op, 'dependency': False}

        # parse lhs
        lhs_parser = TFExpressionParser(expr_str=lhs, args=args, lhs=True, tf_graph=tf_graph)

    return lhs_parser.parse_expr(), args


def parse_dict(var_dict: dict,
               tf_graph: tp.Optional[tf.Graph] = None
               ) -> dict:
    """Parses a dictionary with variable information and creates tensorflow variables from that information.

    Parameters
    ----------
    var_dict
        Contains key-value pairs for each variable that should be translated into the tensorflow graph.
        Each value is a dictionary again containing the variable information (needs at least a field for `vtype`).
    tf_graph
        Instance of `tensorflow.Graph` in which all variables with be initialized.
    var_scope
        Optional variable scope under which the variables will be initialized.

    Returns
    -------
    Tuple
        Containing the variables and the variable names.

    """

    var_dict_tf = {}

    # get tensorflow graph
    tf_graph = tf_graph if tf_graph else tf.get_default_graph()

    with tf_graph.as_default():

        # go through dictionary items and instantiate variables
        #######################################################

        for var_name, var in var_dict.items():

            if var['vtype'] == 'raw':

                # just extract raw variable value
                tf_var = var['value']

            elif var['vtype'] == 'state_var':

                # create a tensorflow variable that can change its value over the course of a simulation
                tf_var = tf.get_variable(name=var_name,
                                         shape=var['shape'],
                                         dtype=getattr(tf, var['dtype']),
                                         initializer=tf.constant_initializer(var['value'])
                                         )

            elif var['vtype'] == 'constant':

                # create dense, constant tensor
                tf_var = tf.constant(value=var['value'],
                                     name=var_name,
                                     shape=var['shape'],
                                     dtype=getattr(tf, var['dtype'])
                                     )

            elif var['vtype'] == 'placeholder':

                tf_var = tf.placeholder(name=var_name,
                                        shape=var['shape'],
                                        dtype=getattr(tf, var['dtype'])
                                        )

            elif var['vtype'] in 'constant_sparse_state_var_sparse':

                # create a sparse tensor
                tf_var = tf.SparseTensor(dense_shape=var['shape'],
                                         values=var['value'],
                                         indices=var['indices']
                                         )

            else:

                raise ValueError('Variable type must be `raw`, `state_variable`, `constant` or `placeholder`.')

            var_dict_tf[var_name] = tf_var

    return var_dict_tf


def round_to_prec(x, prec=0):
    prec = 10**prec
    return tf.round(x * prec) / prec


# class LambdaExpressionParser(ExpressionParser):
#     """Expression parser that turns expressions into lambda functions.
#     """
#
#     def parse(self, expr_stack: list) -> tp.Union[np.ndarray, float, int, tp.Callable]:
#         """Parse elements in expression stack to lambda function.
#
#         Parameters
#         ----------
#         expr_stack
#             Ordered list with expression variables and operations.
#
#         Returns
#         -------
#         type.Any
#
#         """
#
#         # get next operation from stack
#         op = expr_stack.pop()
#
#         # check type of operation
#         #########################
#
#         if op == '-one':
#
#             # multiply expression by minus one
#             arg = self.parse(expr_stack)
#             if callable(arg):
#                 self._op_tmp = lambda: -arg()
#             else:
#                 self._op_tmp = lambda: -arg
#
#         elif op in "+-*/^@<=>=!==":
#
#             # combine elements via mathematical/boolean operator
#             op2 = self.parse(expr_stack)
#             op1 = self.parse(expr_stack)
#             if callable(op1):
#                 if callable(op2):
#                     self._op_tmp = lambda: self.ops[op](op1(), op2())
#                 else:
#                     self._op_tmp = lambda: self.ops[op](op1(), op2)
#             else:
#                 if callable(op2):
#                     self._op_tmp = lambda: self.ops[op](op1, op2())
#                 else:
#                     self._op_tmp = lambda: self.ops[op](op1, op2)
#
#         elif op in ".T.I":
#
#             # transpose/invert expression
#             arg = self.parse(expr_stack)
#             if callable(arg):
#                 self._op_tmp = lambda: self.ops[op](arg())
#             else:
#                 self._op_tmp = lambda: self.ops[op](arg)
#
#         elif op == "]":
#
#             # parse indices
#             indices = []
#             while len(expr_stack) > 0 and expr_stack[-1] != "[":
#                 index = []
#                 while len(expr_stack) > 0 and expr_stack[-1] not in ",[":
#                     if expr_stack[-1] == ":":
#                         index.append(expr_stack.pop())
#                     else:
#                         index.append(self.parse(expr_stack))
#                 indices.append(index[::-1])
#                 if expr_stack[-1] == ",":
#                     expr_stack.pop()
#             expr_stack.pop()
#
#             # build str representation of idx
#             idx = ""
#             global new_vars
#             new_vars = []
#             i = 0
#             for index in indices[::-1]:
#                 for ind in index:
#                     if type(ind) == str:
#                         idx += ind
#                     elif isinstance(ind, Number):
#                         idx += f"{ind}"
#                     else:
#                         try:
#                             new_vars.append(ind.__copy__())
#                         except AttributeError:
#                             new_vars.append(copy(ind))
#                         if callable(new_vars[i]):
#                             idx += f"new_vars[{i}]()"
#                         else:
#                             idx += f"new_vars[{i}]"
#                         i += 1
#                 idx += ","
#             idx = idx[0:-1]
#
#             # apply indices to op
#             global op_to_idx
#             op_to_idx = self.parse(expr_stack)
#             if callable(op_to_idx):
#                 exec(f"self._op_tmp = lambda: op_to_idx()[{idx}]")
#             else:
#                 exec(f"self._op_tmp = lambda: op_to_idx[{idx}]")
#
#         elif op == "PI":
#
#             # return float representation of pi
#             self._op_tmp = math.pi
#
#         elif op == "E":
#
#             # return float representation of e
#             self._op_tmp = math.e
#
#         elif any(["float" in op, "bool" in op, "int" in op, "complex" in op]):
#
#             # extract data type
#             try:
#                 dtype = self.dtypes[op[0:-1]]
#             except AttributeError:
#                 raise ValueError(f"Datatype casting error in expression: {self.expr_str}. {op[0:-1]} is not a valid "
#                                  f"data-type for this parser.")
#
#             # cast new data type to argument
#             arg = self.parse(expr_stack)
#             if callable(arg):
#                 self._op_tmp = self.funcs['cast'](arg(), dtype)
#             else:
#                 self._op_tmp = self.funcs['cast'](arg, dtype)
#
#         elif op[-1] == "(":
#
#             # extract function
#             try:
#                 f = self.funcs[op[0:-1]]
#             except KeyError:
#                 raise ValueError(f"Undefined function in expression: {self.expr_str}. {op[0:-1]} needs to be provided "
#                                  f"in arguments dictionary.")
#
#             # parse arguments
#             args = []
#             while len(expr_stack) > 0:
#                 args.append(self.parse(expr_stack))
#                 if len(expr_stack) == 0 or expr_stack[-1] != ",":
#                     break
#                 else:
#                     expr_stack.pop()
#
#             # apply function to arguments
#             self._op_tmp = lambda: f(*tuple([arg() if callable(arg) else arg for arg in args[::-1]]))
#
#         elif op in self.args.keys():
#
#             # extract variable from args dict
#             self._op_tmp = self.args[op]
#
#         elif any([op == "True", op == "true", op == "False", op == "false"]):
#
#             # return boolean
#             self._op_tmp = True if op in "Truetrue" else False
#
#         elif "." in op:
#
#             # return float
#             self._op_tmp = float(op)
#
#         elif op[0].isnumeric():
#
#             # return integer
#             self._op_tmp = int(op)
#
#         elif op[0].isalpha():
#
#             if self.lhs:
#
#                 op = self.args['rhs']
#                 self._op_tmp = op
#                 self.args[op] = op
#
#             else:
#
#                 raise ValueError(f"Undefined variable detected in expression: {self.expr_str}. {op} was not found "
#                                  f"in the respective arguments dictionary.")
#
#         else:
#
#             raise ValueError(f"Undefined operation detected in expression: {self.expr_str}. {op} cannot be "
#                              f"interpreted by this parser.")
#
#         return self._op_tmp


# class NPExpressionParser(LambdaExpressionParser):
#     """Expression parser that turns expressions into numpy operations.
#     """
#
#     def __init__(self, expr_str: str, args: dict, lhs: bool = False) -> None:
#         """Instantiate numpy expression parser.
#         """
#
#         # call super init
#         #################
#
#         super().__init__(expr_str=expr_str, args=args, lhs=lhs)
#
#         # define operations and functions
#         #################################
#
#         # base math operations
#         ops = {"+": np.add,
#                "-": np.subtract,
#                "*": np.multiply,
#                "/": np.true_divide,
#                "%": np.mod,
#                "^": np.float_power,
#                "@": np.matmul,
#                ".T": np.transpose,
#                ".I": np.invert,
#                ">": np.greater,
#                "<": np.less,
#                "==": np.equal,
#                "!=": np.not_equal,
#                ">=": np.greater_equal,
#                "<=": np.less_equal
#                }
#         for key, val in ops.items():
#             self.ops[key] = val
#
#         # additional functions
#         funcs = {"sin": np.sin,
#                  "cos": np.cos,
#                  "tan": np.tan,
#                  "abs": np.abs,
#                  "max": np.max,
#                  "min": np.min,
#                  "argmax": np.argmax,
#                  "argmin": np.argmin,
#                  "round": np.round,
#                  "sum": np.sum,
#                  "tile": np.tile,
#                  "reshape": np.reshape,
#                  "cast": np.array,
#                  "randn": np.random.randn,
#                  "ones": np.ones,
#                  "zeros": np.zeros,
#                  "new_var": np.array
#                  }
#         for key, val in funcs.items():
#             self.funcs[key] = val
#
#         dtypes = {"float16": np.float16,
#                   "float32": np.float32,
#                   "float64": np.float64,
#                   "int16": np.int16,
#                   "int32": np.int32,
#                   "int64": np.int64,
#                   "uint16": np.uint16,
#                   "uint32": np.uint32,
#                   "uint64": np.uint64,
#                   "complex64": np.complex64,
#                   "complex128": np.complex128,
#                   "bool": tf.bool
#                   }
#         for key, val in dtypes.items():
#             self.dtypes[key] = val


# functions using the expression parser
#######################################

# # solver classes (update lhs according to rhs)
# ##############################################
#
#
# class Solver(object):
#     """Base solver class (currently only implements basic forward euler).
#
#     Parameters
#     ----------
#     rhs
#         Tensorflow operation that represents right-hand side of a differential equation.
#     state_var
#         Tensorflow variable that should be integrated over time.
#     dt
#         Step-size of the integration over time [unit = s].
#
#     Attributes
#     ----------
#
#     Methods
#     -------
#
#     References
#     ----------
#
#     Examples
#     --------
#
#     """
#
#     def __init__(self,
#                  rhs: tp.Union[tf.Operation, tf.Tensor, tp.Callable, tuple],
#                  state_var: tp.Union[tf.Variable, tf.Tensor, np.ndarray, float, int],
#                  dt: tp.Optional[float] = None
#                  ) -> None:
#         """Instantiates solver.
#         """
#
#         # initialize instance attributes
#         ################################
#
#         self.rhs = rhs
#         self.state_var = state_var
#         self.dt = dt
#
#         # define integration expression
#         ###############################
#
#         # TODO: Implement Butcher tableau and its translation into various solver algorithms
#         if self.dt is None:
#             self.integration_expression = "rhs"
#         else:
#             self.integration_expression = "dt * rhs"
#
#     def solve(self) -> tp.Union[tf.Operation, tf.Tensor]:
#         """Creates tensorflow method for performing a single differentiation step.
#         """
#
#         raise NotImplementedError('This method needs to be implemented at the child class level '
#                                   '(any class inheriting the `Solver` class).')
#
#
# class TFSolver(Solver):
#
#     def __init__(self,
#                  rhs: tp.Union[tf.Operation, tf.Tensor],
#                  state_var: tp.Union[tf.Variable, tf.Tensor],
#                  dt: tp.Optional[float] = None,
#                  _tf_graph: tp.Optional[tf.Graph] = None
#                  ) -> None:
#         """Instantiates solver.
#         """
#
#         # call super init
#         #################
#
#         super().__init__(rhs=rhs, state_var=state_var, dt=dt)
#
#         # initialize additional attributes
#         ##################################
#
#         self._tf_graph = _tf_graph if _tf_graph else tf.get_default_graph()
#
#     def solve(self) -> tp.Union[tf.Operation, tf.Tensor]:
#         """Creates tensorflow method for performing a single differentiation step.
#         """
#
#         with self._tf_graph.as_default():
#
#             # go through integration expressions to solve DE
#             ################################################
#
#             # parse the integration expression
#             expr_args = {'dt': {'var': self.dt, 'dependency': False}, 'rhs': {'var': self.rhs, 'dependency': False}}
#             parser = TFExpressionParser(self.integration_expression, expr_args, _tf_graph=self._tf_graph)
#             op = parser.parse_expr()
#
#             # update the target state variable
#             if self.dt is not None:
#                 op = self.state_var + op
#
#         return op
#
#
# class NPSolver(Solver):
#
#     def solve(self) -> list:
#
#         steps = list()
#
#         # go through integration expressions to solve DE
#         ################################################
#
#         for expr in self.integration_expressions:
#
#             # parse the integration expression
#             expr_args = {'dt': self.dt, 'rhs': self.rhs}
#             parser = NPExpressionParser(expr, expr_args)
#             op = parser.parse_expr()
#
#             # update the target state variable
#             if self.dt is None:
#                 if callable(op):
#                     update = lambda: op()
#                 else:
#                     update = lambda: op
#             else:
#                 if callable(op):
#                     if callable(self.state_var):
#                         update = lambda: self.state_var() + op()
#                     else:
#                         update = lambda: self.state_var + op()
#                 else:
#                     if callable(self.state_var):
#                         update = lambda: self.state_var() + op
#                     else:
#                         update = lambda: self.state_var + op
#             steps.append(update)
#
#         return steps

