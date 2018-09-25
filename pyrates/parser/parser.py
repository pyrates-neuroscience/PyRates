"""This module provides parser classes and functions to parse string-based equations into operations.
"""

# external imports
from pyparsing import Literal, CaselessLiteral, Word, Combine, Optional, \
    ZeroOrMore, Forward, nums, alphas, ParserElement
from numbers import Number
from copy import copy
import math
import tensorflow as tf
import typing as tp
import numpy as np

# pyrates internal imports

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


# equation parsers
##################

class EquationParser(object):
    """Parses lhs and rhs of an equation.

    Parameters
    ----------
    expr_str
        Mathematical equation in string format.
    args
        Dictionary containing all variables and functions needed to evaluate the expression.
    engine
        String indicating which computational backbone/backend to use. Can be either `numpy` or `tensorflow`.
    tf_graph
        Tensorflow graph on which all operations will be created.

    Attributes
    ----------

    Methods
    -------

    Examples
    --------

    References
    ----------

    """

    def __init__(self, expr_str: str, args: dict, engine: str = 'tensorflow', tf_graph: tp.Optional[tf.Graph] = None
                 ) -> None:
        """Instantiates equation parser.
        """

        # bind inputs args to instance
        ##############################

        self.expr_str = expr_str
        self.args = args
        self.tf_graph = tf_graph if tf_graph else tf.get_default_graph()

        # parse lhs and rhs of equation
        ###############################

        # split into lhs and rhs
        lhs, rhs = self.expr_str.split(' = ')

        # parse rhs
        if engine == 'tensorflow':
            rhs_parser = TFExpressionParser(expr_str=rhs, args=self.args, tf_graph=self.tf_graph)
        elif engine == 'numpy':
            rhs_parser = NPExpressionParser(expr_str=rhs, args=self.args)
        else:
            raise ValueError('Engine needs to be set to either `tensorflow` or `numpy`.')
        rhs_op = rhs_parser.parse_expr()

        # add information about rhs to args
        self.args['rhs'] = {'var': rhs_op, 'dependency': False}

        # check update type of lhs
        if "d/dt" in lhs:
            lhs_split = lhs.split('*')
            lhs = ""
            for lhs_part in lhs_split[1:]:
                lhs += lhs_part
            solve = True
        else:
            solve = False

        # parse lhs
        if engine == 'tensorflow':
            lhs_parser = TFExpressionParser(expr_str=lhs, args=self.args, lhs=True,  tf_graph=self.tf_graph)
        elif engine == 'numpy':
            lhs_parser = NPExpressionParser(expr_str=rhs, args=self.args, lhs=True)
        else:
            raise ValueError('Engine needs to be set to either `tensorflow` or `numpy`.')
        self.target_var = lhs_parser.parse_expr()

        # solve for state variable
        ##########################

        if solve:

            # set integration step-size
            try:
                dt = self.args['dt']['var']
            except KeyError:
                raise ValueError('Integration step-size has to be passed for differential equations. Please '
                                 'add a field `dt` to `args` with the corresponding value.')

            # create solver instance
            if engine == 'tensorflow':
                solver = TFSolver(rhs=rhs_op,
                                  state_var=self.target_var,
                                  dt=dt,
                                  tf_graph=self.tf_graph)
            else:
                solver = NPSolver(rhs=rhs_op,
                                  state_var=self.target_var,
                                  dt=dt)

            # collect solver update operation
            self.update = solver.solve()

        elif 'rhs' in self.args.keys():
            self.update = self.args.pop('rhs')['var']


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

    Attributes
    ----------

    Methods
    -------

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

        self.expr_str = expr_str
        self.args = args

        # additional attributes
        #######################

        self.expr = None
        self.expr_stack = []
        self.expr_list = []
        self._op_tmp = None
        self.op = None
        self.lhs = lhs

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
            index_full = index_base + ZeroOrMore((index_comb + index_base))
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

        # additional functions
        self.funcs = {}

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
        self.op = self.parse(self.expr_stack[:])

        return self.op

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

            # combine elements via mathematical/boolean operator
            op2 = self.parse(expr_stack)
            op1 = self.parse(expr_stack)
            self._op_tmp = self.ops[op](op1, op2)

        elif op in ".T.I":

            # transpose/invert expression
            self._op_tmp = self.ops[op](self.parse(expr_stack))

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
                self._op_tmp = eval(f"op_to_idx[{idx}]")
            except (TypeError, ValueError):
                try:
                    self._op_tmp = eval(f"self.funcs['array_idx'](op_to_idx, {idx})")
                except (TypeError, ValueError):
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
            except KeyError:
                raise ValueError(f"Undefined function in expression: {self.expr_str}. {op[0:-1]} needs to be provided "
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


class LambdaExpressionParser(ExpressionParser):
    """Expression parser that turns expressions into lambda functions.
    """

    def parse(self, expr_stack: list) -> tp.Union[np.ndarray, float, int, tp.Callable]:
        """Parse elements in expression stack to lambda function.

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
            arg = self.parse(expr_stack)
            if callable(arg):
                self._op_tmp = lambda: -arg()
            else:
                self._op_tmp = lambda: -arg

        elif op in "+-*/^@<=>=!==":

            # combine elements via mathematical/boolean operator
            op2 = self.parse(expr_stack)
            op1 = self.parse(expr_stack)
            if callable(op1):
                if callable(op2):
                    self._op_tmp = lambda: self.ops[op](op1(), op2())
                else:
                    self._op_tmp = lambda: self.ops[op](op1(), op2)
            else:
                if callable(op2):
                    self._op_tmp = lambda: self.ops[op](op1, op2())
                else:
                    self._op_tmp = lambda: self.ops[op](op1, op2)

        elif op in ".T.I":

            # transpose/invert expression
            arg = self.parse(expr_stack)
            if callable(arg):
                self._op_tmp = lambda: self.ops[op](arg())
            else:
                self._op_tmp = lambda: self.ops[op](arg)

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

            # build str representation of idx
            idx = ""
            global new_vars
            new_vars = []
            i = 0
            for index in indices[::-1]:
                for ind in index:
                    if type(ind) == str:
                        idx += ind
                    elif isinstance(ind, Number):
                        idx += f"{ind}"
                    else:
                        try:
                            new_vars.append(ind.__copy__())
                        except AttributeError:
                            new_vars.append(copy(ind))
                        if callable(new_vars[i]):
                            idx += f"new_vars[{i}]()"
                        else:
                            idx += f"new_vars[{i}]"
                        i += 1
                idx += ","
            idx = idx[0:-1]

            # apply indices to op
            global op_to_idx
            op_to_idx = self.parse(expr_stack)
            if callable(op_to_idx):
                exec(f"self._op_tmp = lambda: op_to_idx()[{idx}]")
            else:
                exec(f"self._op_tmp = lambda: op_to_idx[{idx}]")

        elif op == "PI":

            # return float representation of pi
            self._op_tmp = math.pi

        elif op == "E":

            # return float representation of e
            self._op_tmp = math.e

        elif any(["float" in op, "bool" in op, "int" in op, "complex" in op]):

            # extract data type
            try:
                dtype = self.dtypes[op[0:-1]]
            except AttributeError:
                raise ValueError(f"Datatype casting error in expression: {self.expr_str}. {op[0:-1]} is not a valid "
                                 f"data-type for this parser.")

            # cast new data type to argument
            arg = self.parse(expr_stack)
            if callable(arg):
                self._op_tmp = self.funcs['cast'](arg(), dtype)
            else:
                self._op_tmp = self.funcs['cast'](arg, dtype)

        elif op[-1] == "(":

            # extract function
            try:
                f = self.funcs[op[0:-1]]
            except KeyError:
                raise ValueError(f"Undefined function in expression: {self.expr_str}. {op[0:-1]} needs to be provided "
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
            self._op_tmp = lambda: f(*tuple([arg() if callable(arg) else arg for arg in args[::-1]]))

        elif op in self.args.keys():

            # extract variable from args dict
            self._op_tmp = self.args[op]

        elif any([op == "True", op == "true", op == "False", op == "false"]):

            # return boolean
            self._op_tmp = True if op in "Truetrue" else False

        elif "." in op:

            # return float
            self._op_tmp = float(op)

        elif op[0].isnumeric():

            # return integer
            self._op_tmp = int(op)

        elif op[0].isalpha():

            if self.lhs:

                op = self.args['rhs']
                self._op_tmp = op
                self.args[op] = op

            else:

                raise ValueError(f"Undefined variable detected in expression: {self.expr_str}. {op} was not found "
                                 f"in the respective arguments dictionary.")

        else:

            raise ValueError(f"Undefined operation detected in expression: {self.expr_str}. {op} cannot be "
                             f"interpreted by this parser.")

        return self._op_tmp


class TFExpressionParser(ExpressionParser):
    """Expression parser that transforms expression into tensorflow operations on a tensorflow graph.
    """

    def __init__(self, expr_str: str, args: dict, lhs: bool = False, tf_graph: tp.Optional[tf.Graph] = None) -> None:
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
                 "array_idx": tf.gather_nd,
                 "new_var": tf.get_variable
                 }
        for key, val in funcs.items():
            self.funcs[key] = val

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

            # set dependencies
            dependencies = []
            for arg in expr_stack:
                if arg in self.args.keys() and self.args[arg]['dependency']:
                    dependencies.append(self.args[arg]['op'])

            # create tensorflow operation/variable
            with tf.control_dependencies(dependencies):
                return super().parse(expr_stack=expr_stack)


class NPExpressionParser(LambdaExpressionParser):
    """Expression parser that turns expressions into numpy operations.
    """

    def __init__(self, expr_str: str, args: dict, lhs: bool = False) -> None:
        """Instantiate numpy expression parser.
        """

        # call super init
        #################

        super().__init__(expr_str=expr_str, args=args, lhs=lhs)

        # define operations and functions
        #################################

        # base math operations
        ops = {"+": np.add,
               "-": np.subtract,
               "*": np.multiply,
               "/": np.true_divide,
               "%": np.mod,
               "^": np.float_power,
               "@": np.matmul,
               ".T": np.transpose,
               ".I": np.invert,
               ">": np.greater,
               "<": np.less,
               "==": np.equal,
               "!=": np.not_equal,
               ">=": np.greater_equal,
               "<=": np.less_equal
               }
        for key, val in ops.items():
            self.ops[key] = val

        # additional functions
        funcs = {"sin": np.sin,
                 "cos": np.cos,
                 "tan": np.tan,
                 "abs": np.abs,
                 "max": np.max,
                 "min": np.min,
                 "argmax": np.argmax,
                 "argmin": np.argmin,
                 "round": np.round,
                 "sum": np.sum,
                 "tile": np.tile,
                 "reshape": np.reshape,
                 "cast": np.array,
                 "randn": np.random.randn,
                 "ones": np.ones,
                 "zeros": np.zeros,
                 "new_var": np.array
                 }
        for key, val in funcs.items():
            self.funcs[key] = val

        dtypes = {"float16": np.float16,
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
                  "bool": tf.bool
                  }
        for key, val in dtypes.items():
            self.dtypes[key] = val


# solver classes (update lhs according to rhs)
##############################################


class Solver(object):
    """Base solver class (currently only implements basic forward euler).

    Parameters
    ----------
    rhs
        Tensorflow operation that represents right-hand side of a differential equation.
    state_var
        Tensorflow variable that should be integrated over time.
    dt
        Step-size of the integration over time [unit = s].

    Attributes
    ----------

    Methods
    -------

    References
    ----------

    Examples
    --------

    """

    def __init__(self,
                 rhs: tp.Union[tf.Operation, tf.Tensor, tp.Callable, tuple],
                 state_var: tp.Union[tf.Variable, tf.Tensor, np.ndarray, float, int],
                 dt: tp.Optional[float] = None
                 ) -> None:
        """Instantiates solver.
        """

        # initialize instance attributes
        ################################

        self.rhs = rhs
        self.state_var = state_var
        self.dt = dt

        # define integration expression
        ###############################

        # TODO: Implement Butcher tableau and its translation into various solver algorithms
        if self.dt is None:
            self.integration_expression = "rhs"
        else:
            self.integration_expression = "dt * rhs"

    def solve(self) -> tp.Union[tf.Operation, tf.Tensor]:
        """Creates tensorflow method for performing a single differentiation step.
        """

        raise NotImplementedError('This method needs to be implemented at the child class level '
                                  '(any class inheriting the `Solver` class).')


class TFSolver(Solver):

    def __init__(self,
                 rhs: tp.Union[tf.Operation, tf.Tensor],
                 state_var: tp.Union[tf.Variable, tf.Tensor],
                 dt: tp.Optional[float] = None,
                 tf_graph: tp.Optional[tf.Graph] = None
                 ) -> None:
        """Instantiates solver.
        """

        # call super init
        #################

        super().__init__(rhs=rhs, state_var=state_var, dt=dt)

        # initialize additional attributes
        ##################################

        self.tf_graph = tf_graph if tf_graph else tf.get_default_graph()

    def solve(self) -> tp.Union[tf.Operation, tf.Tensor]:
        """Creates tensorflow method for performing a single differentiation step.
        """

        with self.tf_graph.as_default():

            # go through integration expressions to solve DE
            ################################################

            # parse the integration expression
            expr_args = {'dt': {'var': self.dt, 'dependency': False}, 'rhs': {'var': self.rhs, 'dependency': False}}
            parser = TFExpressionParser(self.integration_expression, expr_args, tf_graph=self.tf_graph)
            op = parser.parse_expr()

            # update the target state variable
            if self.dt is not None:
                op = self.state_var + op

        return op


class NPSolver(Solver):

    def solve(self) -> list:

        steps = list()

        # go through integration expressions to solve DE
        ################################################

        for expr in self.integration_expressions:

            # parse the integration expression
            expr_args = {'dt': self.dt, 'rhs': self.rhs}
            parser = NPExpressionParser(expr, expr_args)
            op = parser.parse_expr()

            # update the target state variable
            if self.dt is None:
                if callable(op):
                    update = lambda: op()
                else:
                    update = lambda: op
            else:
                if callable(op):
                    if callable(self.state_var):
                        update = lambda: self.state_var() + op()
                    else:
                        update = lambda: self.state_var + op()
                else:
                    if callable(self.state_var):
                        update = lambda: self.state_var() + op
                    else:
                        update = lambda: self.state_var + op
            steps.append(update)

        return steps


def parse_dict(var_dict: dict,
               engine: str = "tensorflow",
               tf_graph: tp.Optional[tf.Graph] = None,
               var_scope: tp.Optional[str] = None
               ) -> tp.Tuple[list, list]:
    """Parses a dictionary with variable information and creates tensorflow variables from that information.

    Parameters
    ----------
    var_dict
    engine
    tf_graph
    var_scope

    Returns
    -------
    Tuple

    """

    var_col = []
    var_names = []

    if engine == "tensorflow":

        # get tensorflow graph
        tf_graph = tf_graph if tf_graph else tf.get_default_graph()

        with tf_graph.as_default():

            with tf.variable_scope(var_scope):

                # go through dictionary items and instantiate variables
                #######################################################

                for var_name, var in var_dict.items():

                    if var['vtype'] == 'raw':

                        tf_var = var['value']

                    elif var['vtype'] == 'state_var':

                        tf_var = tf.get_variable(name=var['name'],
                                                 shape=var['shape'],
                                                 dtype=getattr(tf, var['dtype']),
                                                 initializer=tf.constant_initializer(var['value'])
                                                 )

                    elif var['vtype'] == 'constant_sparse':

                        # Check the shape, zeros and non-zero elements in the input matrix
                        # Check if zeros are more than 30 percent of the whole dense matrix and while doing that,
                        # record the index of each non zero element.

                        if len(var['shape']) == 2:
                            tN = 1

                            for Num in var['shape']:
                                tN = Num * tN

                            zN = 0
                            i = 0

                            NonZer_idx = []
                            NonZer_val = []

                            for arr in var['value']:
                                j = 0
                                for elem in arr:

                                    if elem == 0.0:
                                        zN += 1

                                    else:

                                        NonZer = [i, j]
                                        NonZer_idx.append(NonZer)

                                        NonZer_val.append(elem)
                                    j += 1
                                i += 1
                            if zN > 0.3*tN:

                                tf_var = tf.SparseTensor(indices=NonZer_idx,
                                                         values=NonZer_val,
                                                         dense_shape=var['shape'])

                    elif var['vtype'] == 'constant':

                        tf_var = tf.constant(value=var['value'],
                                             name=var['name'],
                                             shape=var['shape'],
                                             dtype=getattr(tf, var['dtype'])
                                             )

                    elif var['vtype'] == 'placeholder':

                        tf_var = tf.placeholder(name=var['name'],
                                                shape=var['shape'],
                                                dtype=getattr(tf, var['dtype'])
                                                )

                    else:

                        raise ValueError('Variable type must be `raw`, `state_variable`, `constant` or `placeholder`.')

                    var_col.append(tf_var)
                    var_names.append(var_name)

    elif engine == "numpy":

        for var_name, var in var_dict.items():

            if var['vtype'] == 'raw':
                np_var = var['value']
            else:
                np_var = np.zeros(shape=var['shape'],
                                  dtype=getattr(np, var['dtype']))
                np_var += var['value']

            var_col.append(np_var)
            var_names.append(var_name)

    else:

        raise ValueError('Engine must be set to either `tensorflow` or `numpy`.')

    return var_col, var_names


def round_to_prec(x, prec=0):
    prec = 10**prec
    return tf.round(x * prec) / prec
