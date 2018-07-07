"""This module provides parser classes and functions to parse string-based equations into operations.
"""

# external imports
from pyparsing import Literal, CaselessLiteral, Word, Combine, Group, Optional, \
    ZeroOrMore, Forward, nums, alphas, ParserElement
import math
import tensorflow as tf
import typing as type

# pyrates internal imports
from pyrates.solver import Solver

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


class ExpressionParser(ParserElement):
    """Base class for parsing mathematical expressions.
    """
    
    def __init__(self, expr: str, args: dict, tf_graph: type.Optional[tf.Graph] = None) -> None:
        """Instantiates expression parser.
        """
        
        # call super init
        #################
        
        super().__init__()
        
        # bind input args to instance
        #############################

        self.expr_str = expr
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

            # indices
            idx_1d = Combine(num_int + Optional(":" + Optional(num_int)) + Optional(":" + Optional(num_int)))
            idx = Combine(idx_1d + Optional(comma + idx_1d) + Optional(comma + idx_1d) + Optional(comma + idx_1d))

            # variable and function names
            name = Word(alphas, alphas + nums + "_$")
            arg = name | num_float | num_int | Literal("True") | Literal("False")
            func_args = Combine(arg + ZeroOrMore(comma + arg))

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
                                     ).setParseAction(self.push_first)).setParseAction(self.push_unary) | \
                   (par_l + self.expr.suppress() + par_r).setParseAction(self.push_unary)

            # hierarchical relationships between operations
            func = atom + Optional(par_l + func_args.setParseAction(self.push_first) + par_r)
            indexed = func + ZeroOrMore((idx_l + idx + idx_r).setParseAction(self.push_all))
            factor << indexed + ZeroOrMore((op_exp + factor).setParseAction(self.push_first))
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
                      "sum": tf.reduce_sum}

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
        """"""
        self.expr_stack.append(toks[0])
        
    def push_unary(self, strg, loc, toks):
        """"""
        if toks and toks[0] == '-':
            self.expr_stack.append('unary -')

    def push_all(self, strg, loc, toks):
        for t in toks:
            self.expr_stack.append(t)

    def push_last(self, strg, loc, toks):
        self.expr_stack.append(toks[-1])

    def parse(self, expr_stack: list) -> type.Union[tf.Operation, tf.Tensor, tf.Variable, float]:
        """"""

        with self.tf_graph.as_default():

            op = expr_stack.pop()

            if op == 'unary -':
                self._op_tmp = -self.parse(expr_stack)
            elif op in "+-*/^@":
                op2 = self.parse(expr_stack)
                op1 = self.parse(expr_stack)
                self._op_tmp = self.ops[op](op1, op2)
            elif op == "]":
                idx = expr_stack.pop()
                expr_stack.pop()
                op_to_idx = self.parse([expr_stack.pop()])
                self._op_tmp = op_to_idx[eval(idx)]
            elif op == "PI":
                self._op_tmp = math.pi
            elif op == "E":
                self._op_tmp = math.e
            elif op in self.funcs:
                args = [self.parse([e]) for e in expr_stack.pop().split(',')]
                self._op_tmp = self.funcs[op](*tuple(args))
            elif op in self.args.keys():
                self._op_tmp = self.args[op]
            elif op[0].isalpha():
                self._op_tmp = 0
            elif "." in op:
                self._op_tmp = float(op)
            else:
                self._op_tmp = int(op)

        return self._op_tmp


class EquationParser(object):
    """Parses lhs and rhs of an equation.
    """

    def __init__(self, expr_str: str, args: dict, tf_graph: type.Optional[tf.Graph] = None) -> None:
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

        # find state variable in lhs and solve for it
        #############################################

        # search for state variable in lhs
        var_n = 0
        self.target_var = None
        while len(lhs_list) > 0:
            symb = lhs_list.pop(0)
            if symb not in 'dt/*[]':
                if symb not in self.args.keys():
                    raise ValueError(f'Could not find state variable in arguments dictionary. Please add {symb} '
                                     'to `args`.')
                self.target_var = self.args[symb]
                if var_n > 0:
                    raise ValueError('Multiple potential state variables found in left-hand side of equation. Please'
                                     'reformulate equation to follow one of the following formulations:'
                                     'y = f(...); d/dt y = f(...); Y[idx] = f(...).')
                var_n += 1
            elif symb == '[':
                if not self.target_var:
                    raise ValueError('Beginning of index found befor state variable could be identified. Please'
                                     'reformulate equation to follow one of the following formulations:'
                                     'y = f(...); d/dt y = f(...); Y[idx] = f(...).')
                idx = lhs_list.pop(0)
                self.target_var = self.target_var[eval(idx)]

        if self.target_var is None:
            raise ValueError('Could not find state variable in left-hand side of equation. Please'
                             'reformulate equation to follow one of the following formulations:'
                             'y = f(...); d/dt y = f(...); Y[idx] = f(...).')

        # solve for state variable
        if 'd' in list(lhs_parser.expr_list) and 'dt' in list(lhs_parser.expr_list):
            if 'dt' not in self.args.keys():
                raise ValueError('Integration step-size has to be passed with differential equations. Please '
                                 'add a field `dt` to `args` with the corresponding value.')
            solver = Solver(rhs_op, self.target_var, self.args['dt'], self.tf_graph)
            with self.tf_graph.as_default():
                self.lhs_update = solver.solve()
        else:
            with self.tf_graph.as_default():
                self.lhs_update = self.target_var.assign(rhs_op)

############
# test bed #
############

import numpy as np
string = "A = (-A + sum(B,1)) * sin(c)"

gr = tf.Graph()
with gr.as_default():
    A = tf.Variable(np.ones(5), dtype=tf.float32)
    B = tf.constant(np.ones((5, 5)), dtype=tf.float32)
    c = tf.constant(7.3, dtype=tf.float32)
parser = EquationParser(string, {'A': A, 'B': B, 'c': c, 'dt': 1e-1, 'd': 1.}, gr)
op = parser.lhs_update

with tf.Session(graph=gr) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        sess.run(op)
        print(A.eval())
