from pyparsing import Literal, CaselessLiteral, Word, Combine, Group, Optional, \
    ZeroOrMore, Forward, nums, alphas
import math
import operator
import tensorflow as tf

exprStack = []


def pushFirst(strg, loc, toks):
    exprStack.append(toks[0])


def pushUMinus(strg, loc, toks):
    if toks and toks[0] == '-':
        exprStack.append('unary -')
        # ~ exprStack.append( '-1' )
        # ~ exprStack.append( '*' )


bnf = None


def BNF():
    """
    expop   :: '^'
    multop  :: '*' | '/'
    addop   :: '+' | '-'
    integer :: ['+' | '-'] '0'..'9'+
    atom    :: PI | E | real | fn '(' expr ')' | '(' expr ')'
    factor  :: atom [ expop factor ]*
    term    :: factor [ multop factor ]*
    expr    :: term [ addop term ]*
    """
    global bnf
    if not bnf:
        point = Literal(".")
        e = CaselessLiteral("E")
        fnumber = Combine(Word("+-" + nums, nums) +
                          Optional(point + Optional(Word(nums))) +
                          Optional(e + Word("+-" + nums, nums)))
        inumber = Word("+-" + nums, nums)
        idx_1D = Combine(inumber + Optional(":" + Optional(inumber)) + Optional(":" + Optional(inumber)))
        idx = Combine(idx_1D + Optional("," + idx_1D) + Optional("," + idx_1D) + Optional("," + idx_1D))
        ident = Word(alphas, alphas + nums + "_$")

        plus = Literal("+")
        minus = Literal("-")
        mult = Literal("*")
        div = Literal("/")
        lpar = Literal("(").suppress()
        rpar = Literal(")").suppress()
        lidx = Literal("[")
        ridx = Literal("]")
        addop = plus | minus
        multop = mult | div
        expop = Literal("^")
        pi = CaselessLiteral("PI")

        expr = Forward()
        atom = (Optional("-") + (pi | e | fnumber | ident + lpar + expr + rpar | ident + lidx + idx + ridx | ident | idx
                                 ).setParseAction(pushFirst) | (lpar + expr.suppress() + rpar)
                ).setParseAction(pushUMinus)

        # by defining exponentiation as "atom [ ^ factor ]..." instead of "atom [ ^ atom ]...", we get right-to-left exponents, instead of left-to-righ
        # that is, 2^3^2 = 2^(3^2), not (2^3)^2.
        factor = Forward()
        factor << atom + ZeroOrMore((expop + factor).setParseAction(pushFirst))

        term = factor + ZeroOrMore((multop + factor).setParseAction(pushFirst))
        expr << term + ZeroOrMore((addop + term).setParseAction(pushFirst))
        bnf = expr
    return bnf


# map operator symbols to corresponding arithmetic operations
epsilon = 1e-12
opn = {"+": tf.add,
       "-": tf.subtract,
       "*": tf.multiply,
       "/": tf.truediv,
       "^": tf.pow}
fn = {"sin": tf.sin,
      "cos": tf.cos,
      "tan": tf.tan,
      "abs": tf.abs,
      "round": tf.to_int32,
      "sgn": lambda a: abs(a) > epsilon and ((a > 0) - (a < 0)) or 0}


def evaluateStack(s):
    op = s.pop()
    if op == 'unary -':
        return -evaluateStack(s)
    if op in "+-*/^":
        op2 = evaluateStack(s)
        op1 = evaluateStack(s)
        return opn[op](op1, op2)
    elif op == "PI":
        return math.pi  # 3.1415926535
    elif op == "E":
        return math.e  # 2.718281828
    elif op in fn:
        return fn[op](evaluateStack(s))
    elif op[0].isalpha():
        return 0
    else:
        return float(op)


############
# test bed #
############

string = "A[4:1:5,0:-1] + round(3)"

results = BNF().parseString(string)
val = evaluateStack(exprStack[:])

with tf.Session() as sess:
    sess.run(val)
    print(val.eval())
print('hi')
