"""This module provides parser classes and functions to parse string-based equations into operations.
"""

# external imports
from pyparsing import Literal, CaselessLiteral, Word, Combine, Optional, \
    ZeroOrMore, Forward, nums, alphas, ParserElement
import math
import tensorflow as tf
import typing as tp
import numpy as np

# pyrates internal imports

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


class ExpressionParser(ParserElement):
    """Base class for parsing mathematical expressions.

    Parameters
    ----------
    expr_str
        Mathematical expression in string format.
    args
        Dictionary containing all variables and functions needed to evaluate the expression.
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
    
    def __init__(self, expr_str: str, args: dict, tf_graph: tp.Optional[tf.Graph] = None) -> None:
        """Instantiates expression parser.
        """
        
        # call super init
        #################
        
        super().__init__()
        
        # bind input args to instance
        #############################

        self.expr_str = expr_str
        self.args = args
        self.tf_graph = tf_graph if tf_graph else tf.get_default_graph()
        
        # additional attributes
        #######################

        self.expr = None
        self.expr_stack = []
        
        # define algebra
        ################

        if not self.expr:

            # general symbols
            point = Literal(".")
            comma = Literal(",")
            e = CaselessLiteral("E")
            pi = CaselessLiteral("PI")

            # numeric types
            num_float = Combine(Word("+-" + nums, nums) +
                                Optional(point + Optional(Word(nums))) +
                                Optional(e + Word("+-" + nums, nums)))
            num_int = Word("+-" + nums, nums)

            # variable and function names
            name = Word(alphas, alphas + nums + "_$")
            arg = name | num_float | num_int | Literal("True") | Literal("False")
            func_args = Combine(arg + ZeroOrMore(comma + arg))

            # indices
            idx_1d = Literal(":") | name | \
                     Combine(num_int + Optional(":" + Optional(num_int)) + Optional(":" + Optional(num_int)))
            idx = Combine(idx_1d + Optional(comma + idx_1d) + Optional(comma + idx_1d) + Optional(comma + idx_1d))

            # basic mathematical operations
            plus = Literal("+")
            minus = Literal("-")
            mult = Literal("*")
            div = Literal("/")
            dot = Literal("@")
            pow = Literal("^")
            transp = Combine(point + Literal("T"))
            inv = Combine(point + Literal("I"))

            # math operation groups
            op_add = plus | minus
            op_mult = mult | div | dot
            op_exp = pow | inv | transp

            # parentheses
            par_l = Literal("(").suppress()
            par_r = Literal(")").suppress()
            idx_l = Literal("[")
            idx_r = Literal("]")

            # base types
            self.expr = Forward()
            factor = Forward()
            atom = (Optional("-") + (pi | e | name + par_l + func_args.suppress() + par_r | name | num_float | num_int
                                     ).setParseAction(self.push_first)).setParseAction(self.push_negone) | \
                   (par_l + self.expr.suppress() + par_r).setParseAction(self.push_negone)

            # hierarchical relationships between operations
            func = atom + Optional(par_l + func_args.setParseAction(self.push_first) + par_r)
            indexed = func + ZeroOrMore((idx_l + idx + idx_r).setParseAction(self.push_all))
            factor << indexed + ZeroOrMore((op_exp + Optional(factor)).setParseAction(self.push_first))
            term = factor + ZeroOrMore((op_mult + factor).setParseAction(self.push_first))
            self.expr << term + ZeroOrMore((op_add + term).setParseAction(self.push_first))

        # define operations and functions
        #################################

        # base math operations
        self.ops = {"+": tf.add,
                    "-": tf.subtract,
                    "*": tf.multiply,
                    "/": tf.truediv,
                    "^": tf.pow,
                    "@": tf.matmul,
                    ".T": tf.matrix_transpose,
                    ".I": tf.matrix_inverse}

        # additional functions
        self.funcs = {"sin": tf.sin,
                      "cos": tf.cos,
                      "tan": tf.tan,
                      "abs": tf.abs,
                      "round": tf.to_int32,
                      "sum": tf.reduce_sum,
                      "tile": tf.tile
                      }

        # add functions from args dictionary, if passed
        for key, val in self.args.items():
            if callable(val):
                self.funcs[key] = val

        # parse expression
        ##################

        # extract symbols and operations from expression string
        self.expr_list = self.expr.parseString(self.expr_str)

        # turn expression into tensorflow operation
        with self.tf_graph.as_default():
            self._op_tmp = None
            self.op = self.parse(self.expr_stack[:])

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

    def push_last(self, strg, loc, toks):
        """Push tokens in last-to-first order to expression stack.
        """
        self.expr_stack.append(toks[-1])

    def parse(self, expr_stack: list) -> tp.Union[tf.Operation, tf.Tensor, tf.Variable, float, int]:
        """Parse elements in expression stack to tensorflow operation.

        Parameters
        ----------
        expr_stack
            Ordered list with expression variables and operations.

        Returns
        -------
        type.Union[tf.Operation, tf.Tensor, tf.Variable, float]
            Tensorflow operation

        """

        with self.tf_graph.as_default():

            # get next operation from stack
            op = expr_stack.pop()

            # check type of operation
            #########################

            if op == '-one':

                # multiply expression by minus one
                self._op_tmp = -self.parse(expr_stack)

            elif op in "+-*/^@":

                # combine elements via mathematical operator
                op2 = self.parse(expr_stack)
                op1 = self.parse(expr_stack)
                self._op_tmp = self.ops[op](op1, op2)

            elif op in ".T.I":

                # transpose/invert expression
                self._op_tmp = self.ops[op](self.parse(expr_stack))

            elif op == "]":

                # apply indexing to element
                idx = expr_stack.pop()
                expr_stack.pop()
                op_to_idx = self.parse(expr_stack)
                if idx in self.args.keys():
                    idx = self.args[idx]
                    if idx.dtype == 'bool':
                        self._op_tmp = tf.boolean_mask(op_to_idx, idx)
                    else:
                        self._op_tmp = tf.gather_nd(op_to_idx, idx)
                else:
                    self._op_tmp = eval(f"op_to_idx[{idx}]")

            elif op == "PI":

                # return float representation of pi
                self._op_tmp = math.pi

            elif op == "E":

                # return float representation of e
                self._op_tmp = math.e

            elif op in self.funcs:

                # apply function to arguments
                args = [self.parse([e]) for e in expr_stack.pop().split(',')]
                self._op_tmp = self.funcs[op](*tuple(args))

            elif op in self.args.keys():

                # extract variable from args dict
                self._op_tmp = self.args[op]

            elif "." in op:

                # return float
                self._op_tmp = float(op)

            elif op in "TruetrueFalsefalse":

                # return boolean
                self._op_tmp = True if op in "Truetrue" else False

            elif op[0].isalpha():

                # return float(1) for undefined constants
                self._op_tmp = 1.

            else:

                # return int
                self._op_tmp = int(op)

        return self._op_tmp


class EquationParser(object):
    """Parses lhs and rhs of an equation.

    Parameters
    ----------
    expr_str
        Mathematical equation in string format.
    args
        Dictionary containing all variables and functions needed to evaluate the expression.
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

    def __init__(self, expr_str: str, args: dict, tf_graph: tp.Optional[tf.Graph] = None) -> None:
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
        lhs, rhs = self.expr_str.split('=')

        # parse rhs
        rhs_parser = ExpressionParser(rhs, self.args, self.tf_graph)
        rhs_op = rhs_parser.op

        # parse lhs
        lhs_parser = ExpressionParser(lhs, self.args, self.tf_graph)
        lhs_list = lhs_parser.expr_list.copy()

        # find state variable in lhs
        ############################

        var_n = 0
        self.target_var = None
        idx = None

        # go through list of lhs operations
        while len(lhs_list) > 0:

            symb = lhs_list.pop(0)

            # identifies symb as target state variable
            if symb not in 'dt/*[]':

                if symb not in self.args.keys():
                    raise ValueError(f'Could not find state variable in arguments dictionary. Please add {symb} '
                                     'to `args`.')
                self.target_var = self.args[symb]
                if var_n > 0:
                    raise ValueError('Multiple potential state variables found in left-hand side of equation. Please '
                                     'reformulate equation to follow one of the following formulations:'
                                     'y = f(...); d/dt y = f(...); Y[idx] = f(...).')
                var_n += 1

            # identifies indexing of state variable
            elif symb == '[':
                if not self.target_var:
                    raise ValueError('Beginning of index found befor state variable could be identified. Please '
                                     'reformulate equation to follow one of the following formulations:'
                                     'y = f(...); d/dt y = f(...); Y[idx] = f(...).')
                idx = self.parse_idx(lhs_list.pop(0))

        if self.target_var is None:
            raise ValueError('Could not find state variable in left-hand side of equation. Please '
                             'reformulate equation to follow one of the following formulations:'
                             'y = f(...); d/dt y = f(...); Y[idx] = f(...).')

        # solve for state variable
        ##########################

        if 'd' in list(lhs_parser.expr_list) and 'dt' in list(lhs_parser.expr_list):

            if 'dt' not in self.args.keys():
                raise ValueError('Integration step-size has to be passed with differential equations. Please '
                                 'add a field `dt` to `args` with the corresponding value.')

            solver = Solver(rhs=rhs_op,
                            state_var=self.target_var,
                            state_var_idx=idx,
                            dt=self.args['dt'],
                            tf_graph=self.tf_graph)

        else:

            solver = Solver(rhs=rhs_op,
                            state_var=self.target_var,
                            state_var_idx=idx,
                            dt=None,
                            tf_graph=self.tf_graph)

        # collect solver update operation
        with self.tf_graph.as_default():
            self.lhs_update = solver.solve()

    def parse_idx(self, idx):
        """Creates list of indices from idx.
        """

        # retrieve index from args dictionary, if provided
        if idx in self.args.keys():
            idx = self.args[idx]

        if type(idx) == str:
            test_var = np.zeros(self.target_var.shape, dtype=int)
            exec(f"test_var[{idx}] = 1")
            idx = np.argwhere(test_var == 1)
        elif idx.dtype == 'bool':
            idx = np.argwhere(idx)
        else:
            if idx.shape[1] != len(self.target_var.shape):
                raise ValueError('Each entry in indices needs to math the dimensionality of the target variable.')

        return idx


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
    tf_graph
        Tensorflow graph on which rhs and state_var have been created.

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
                 rhs: tp.Union[tf.Operation, tf.Tensor],
                 state_var: tp.Union[tf.Variable, tf.Tensor],
                 state_var_idx: tp.Optional[np.ndarray] = None,
                 dt: tp.Optional[float] = None,
                 tf_graph: tp.Optional[tf.Graph] = None
                 ) -> None:
        """Instantiates solver.
        """

        # initialize instance attributes
        ################################

        self.rhs = rhs
        self.state_var = state_var
        self.state_var_idx = state_var_idx
        self.dt = dt
        self.tf_graph = tf_graph if tf_graph else tf.get_default_graph()

        # define integration expression
        ###############################

        # TODO: Implement Butcher tableau and its translation into various solver algorithms
        if self.dt is None:
            self.integration_expressions = ["rhs"]
        else:
            self.integration_expressions = ["dt * rhs"]

    def solve(self) -> tp.Union[tf.Operation, tf.Tensor]:
        """Creates tensorflow method for performing a single differentiation step.
        """

        with self.tf_graph.as_default():

            steps = list()
            steps.append(tf.no_op())

            # go through integration expressions to solve DE
            ################################################

            for expr in self.integration_expressions:

                # parse the integration expression
                expr_args = {'dt': self.dt, 'rhs': self.rhs}
                parser = ExpressionParser(expr, expr_args, self.tf_graph)

                # update the target state variable
                with tf.control_dependencies([steps[-1]]):

                    if self.state_var_idx is None:

                        if self.dt is None:
                            steps.append(tf.assign(self.state_var, parser.op))
                        else:
                            steps.append(tf.assign_add(self.state_var, parser.op))
                    else:

                        if self.dt is None:
                            steps.append(tf.scatter_nd_update(ref=self.state_var,
                                                              indices=self.state_var_idx,
                                                              updates=tf.squeeze(parser.op)))
                        else:
                            steps.append(tf.scatter_nd_add(ref=self.state_var,
                                                           indices=self.state_var_idx,
                                                           updates=tf.squeeze(parser.op)))

            steps.pop(0)

        return tf.group(steps)
