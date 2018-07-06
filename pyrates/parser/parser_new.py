"""This module provides parser classes and functions to parse string-based equations into operations.
"""

# external imports
from pyparsing import Literal, CaselessLiteral, Word, Combine, Group, Optional, \
    ZeroOrMore, Forward, nums, alphas, ParserElement
import math
import tensorflow as tf
import typing as type

# pyrates internal imports

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
            e = CaselessLiteral("E")
            pi = CaselessLiteral("PI")

            # numeric types
            num_float = Combine(Word("+-" + nums, nums) +
                                Optional(point + Optional(Word(nums))) +
                                Optional(e + Word("+-" + nums, nums)))
            num_int = Word("+-" + nums, nums)

            # indices
            idx_1d = Combine(num_int + Optional(":" + Optional(num_int)) + Optional(":" + Optional(num_int)))
            idx = Combine(idx_1d + Optional("," + idx_1d) + Optional("," + idx_1d) + Optional("," + idx_1d))

            # variable and function names
            name = Word(alphas, alphas + nums + "_$")

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
            atom = (Optional("-") + (pi | e | num_float | name + par_l + self.expr + par_r | name
                                     ).setParseAction(self.push_first) | (par_l + self.expr.suppress() + par_r)
                    ).setParseAction(self.push_unary)

            # hierarchical relationships between operations
            indexed = Forward()
            indexed << atom + ZeroOrMore((idx_l + idx + idx_r).setParseAction(self.push_all))
            factor = Forward()
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
                self._op_tmp = self.funcs[op](self.parse(expr_stack))
            elif op in self.args.keys():
                self._op_tmp = self.args[op]
            elif op[0].isalpha():
                self._op_tmp = 0
            elif "." in op:
                self._op_tmp = float(op)
            else:
                self._op_tmp = int(op)

        return self._op_tmp


# exprStack = []
#
#
# def pushFirst(strg, loc, toks):
#     exprStack.append(toks[0])
#
#
# def pushUMinus(strg, loc, toks):
#     if toks and toks[0] == '-':
#         exprStack.append('unary -')
#         # ~ exprStack.append( '-1' )
#         # ~ exprStack.append( '*' )
#
#
# bnf = None
#
#
# def BNF():
#     """
#     expop   :: '^'
#     multop  :: '*' | '/'
#     addop   :: '+' | '-'
#     integer :: ['+' | '-'] '0'..'9'+
#     atom    :: PI | E | real | fn '(' expr ')' | '(' expr ')'
#     factor  :: atom [ expop factor ]*
#     term    :: factor [ multop factor ]*
#     expr    :: term [ addop term ]*
#     """
#     global bnf
#     if not bnf:
#         point = Literal(".")
#         e = CaselessLiteral("E")
#         fnumber = Combine(Word("+-" + nums, nums) +
#                           Optional(point + Optional(Word(nums))) +
#                           Optional(e + Word("+-" + nums, nums)))
#         inumber = Word("+-" + nums, nums)
#         idx_1D = Combine(inumber + Optional(":" + Optional(inumber)) + Optional(":" + Optional(inumber)))
#         idx = Combine(idx_1D + Optional("," + idx_1D) + Optional("," + idx_1D) + Optional("," + idx_1D))
#         ident = Word(alphas, alphas + nums + "_$")
#
#         plus = Literal("+")
#         minus = Literal("-")
#         mult = Literal("*")
#         div = Literal("/")
#         lpar = Literal("(").suppress()
#         rpar = Literal(")").suppress()
#         lidx = Literal("[")
#         ridx = Literal("]")
#         addop = plus | minus
#         multop = mult | div
#         expop = Literal("^")
#         pi = CaselessLiteral("PI")
#
#         expr = Forward()
#         atom = (Optional("-") + (pi | e | fnumber | ident + lpar + expr + rpar | ident + lidx + idx + ridx | ident | idx
#                                  ).setParseAction(pushFirst) | (lpar + expr.suppress() + rpar)
#                 ).setParseAction(pushUMinus)
#
#         # by defining exponentiation as "atom [ ^ factor ]..." instead of "atom [ ^ atom ]...", we get right-to-left exponents, instead of left-to-righ
#         # that is, 2^3^2 = 2^(3^2), not (2^3)^2.
#         factor = Forward()
#         factor << atom + ZeroOrMore((expop + factor).setParseAction(pushFirst))
#
#         term = factor + ZeroOrMore((multop + factor).setParseAction(pushFirst))
#         expr << term + ZeroOrMore((addop + term).setParseAction(pushFirst))
#         bnf = expr
#     return bnf
#
#
# # map operator symbols to corresponding arithmetic operations
# epsilon = 1e-12
# opn = {"+": tf.add,
#        "-": tf.subtract,
#        "*": tf.multiply,
#        "/": tf.truediv,
#        "^": tf.pow}
# fn = {"sin": tf.sin,
#       "cos": tf.cos,
#       "tan": tf.tan,
#       "abs": tf.abs,
#       "round": tf.to_int32,
#       "sgn": lambda a: abs(a) > epsilon and ((a > 0) - (a < 0)) or 0}
#
#
# def evaluate_stack(s):
#     op = s.pop()
#     if op == 'unary -':
#         return -evaluateStack(s)
#     if op in "+-*/^":
#         op2 = evaluateStack(s)
#         op1 = evaluateStack(s)
#         return opn[op](op1, op2)
#     elif op == "PI":
#         return math.pi  # 3.1415926535
#     elif op == "E":
#         return math.e  # 2.718281828
#     elif op in fn:
#         return fn[op](evaluateStack(s))
#     elif op[0].isalpha():
#         return 0
#     else:
#         return float(op)


############
# test bed #
############

import numpy as np
string = "A @ (B + 4)."

gr = tf.Graph()
with gr.as_default():
    A = tf.constant(np.ones((5, 5)), dtype=tf.float32)
    B = tf.constant(np.ones((5, 5)), dtype=tf.float32)
    c = tf.constant(7.3, dtype=tf.float32)
parser = ExpressionParser(string, {'A': A, 'B': B, 'c': c}, gr)
op = parser.op

with tf.Session(graph=gr) as sess:
    sess.run(op)
    print(op.eval())
print('hi')
