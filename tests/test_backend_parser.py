"""Test suite for basic parser module functionality.
"""

# external imports
import numpy as np
import tensorflow as tf
import pytest

# pyrates internal imports
from pyrates.backend.parser import parse_equation_list, parse_dict
from pyrates.backend.parser import ExpressionParser
from pyrates.backend.backend_wrapper import TensorflowBackend

# meta infos
__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


###########
# Utility #
###########


def setup_module():
    print("\n")
    print("===============================")
    print("| Test Suite 1 : Parser Module |")
    print("===============================")


#########
# Tests #
#########

def test_1_1_expression_parser_init():
    """Testing initializations of different expression parsers:

    See Also
    --------
    :class:`ExpressionParser`: Detailed documentation of expression parser attributes and methods.
    :class:`NPExpressionParser`: Documentation of the numpy-based expression parser
    """

    # list parsers to be tested and their backends
    parsers = [ExpressionParser]
    backends = [TensorflowBackend]

    # test minimal minimal call example
    ###################################

    for Parser, backend in zip(parsers, backends):
        parser = Parser("a + b", {'a': np.ones((3, 3)), 'b': 5.}, backend=backend)
        assert isinstance(parser, Parser)


def test_1_2_expression_parser_parsing_exceptions():
    """Testing error handling of different erroneous parser instantiations:

    See Also
    --------
    :class:`ExpressionParser`: Detailed documentation of expression parser attributes and methods.
    :class:`LambdaExpressionParser`: Documentation of a non-symbolic expression parser.
    """

    # list parsers to be tested
    parsers = [ExpressionParser]
    backends = [TensorflowBackend]

    # test expected exceptions
    ##########################

    for Parser, backend in zip(parsers, backends):

        b = backend()

        # undefined variables
        with pytest.raises(ValueError):
            Parser("a + b", {}, backend=b).parse_expr()

        # wrong dimensionality of dictionary arguments
        args = parse_dict({'a': {'vtype': 'constant', 'value': np.ones((3, 3)), 'dtype': 'float32', 'shape': (3, 3)},
                           'b': {'vtype': 'constant', 'value': np.ones((4, 4)), 'dtype': 'float32', 'shape': (4, 4)}},
                          backend=b)
        with pytest.raises(ValueError):
            Parser("a + b", {'vars': args}, backend=b).parse_expr()

        # wrong data type of dictionary arguments
        args = parse_dict({'a': {'vtype': 'constant', 'value': np.ones((3, 3)), 'dtype': 'float32', 'shape': (3, 3)},
                           'b': {'vtype': 'constant', 'value': np.ones((3, 3)), 'dtype': 'float32', 'shape': (3, 3)}},
                          backend=b)
        with pytest.raises(TypeError):
            Parser("bool(a) + float32(b)", {'vars': args}, backend=b).parse_expr()()

        # undefined mathematical operator
        args = parse_dict({'a': {'vtype': 'constant', 'value': np.ones((3, 3)), 'dtype': 'float32', 'shape': (3, 3)},
                           'b': {'vtype': 'constant', 'value': 5., 'dtype': 'float32', 'shape': ()}},
                          backend=b)
        with pytest.raises(ValueError):
            Parser("a $ b", {'vars': args}, backend=b).parse_expr()()

        # undefined function
        args = parse_dict({'a': {'vtype': 'constant', 'value': np.ones((3, 3)), 'dtype': 'float32', 'shape': (3, 3)},
                           'b': {'vtype': 'constant', 'value': 5., 'dtype': 'float32', 'shape': ()}},
                          backend=b)
        with pytest.raises(KeyError):
            Parser("a / b(5.)", {'vars': args}, backend=b).parse_expr()()


def test_1_3_expression_parser_math_ops():
    """Testing handling of mathematical operations by expression parsers:

    See Also
    --------
    :class:`ExpressionParser`: Detailed documentation of expression parser attributes and methods.
    :class:`LambdaExpressionParser`: Documentation of a non-symbolic expression parser.
    """

    # define test cases
    math_expressions = [("4 + 5", 9.),                # simple addition
                        ("4 - 5", -1.),               # simple subtraction
                        ("4. * 5.", 20.),             # simple multiplication
                        ("4. / 5.", 0.8),             # simple division
                        ("4.^2", 16.),                # simple exponentiation
                        ("4 + -5", -1.),              # negation of variable
                        ("4. * -2", -8.),             # negation of variable in higher-order operation
                        ("4. + 5. * 2.", 14.),        # multiplication before addition
                        ("(4. + 5.) * 2.", 18.),      # parentheses before everything
                        ("4. * 5.^2", 100.)           # exponentiation before multiplication
                        ]

    # define backend
    b = TensorflowBackend()

    # test expression parsers on expression results
    ###############################################

    for expr, target in math_expressions:

        # tensorflow-based parser
        p = ExpressionParser(expr_str=expr, args={}, backend=b)
        p.parse_expr()
        with tf.Session(graph=b) as sess:
            sess.run(p.expr_op)
            result = p.expr_op.eval()
        assert result == pytest.approx(target, rel=1e-6)


def test_1_4_expression_parser_logic_ops():
    """Testing handling of logical operations by expression parsers:

    See Also
    --------
    :class:`ExpressionParser`: Detailed documentation of expression parser attributes and methods.
    :class:`LambdaExpressionParser`: Documentation of a non-symbolic expression parser.
    """

    # define test cases
    logic_expressions = ["4 == 4",                # simple equals
                         "4 != 5",                # simple not-equals
                         "4 < 5",                 # simple less
                         "5 > 4",                 # simple larger
                         "4 <= 4",                # less equals I
                         "4 <= 5",                # less equals II
                         "5 >= 5",                # larger equals I
                         "5 >= 4",                # larger equals II
                         ]

    # define backend
    b = TensorflowBackend()

    # test expression parsers on expression results
    ###############################################

    for expr in logic_expressions:

        # tensorflow-based parser
        p = ExpressionParser(expr_str=expr, args={}, backend=b)
        p.parse_expr()
        with tf.Session(graph=b) as sess:
            sess.run(p.expr_op)
            assert p.expr_op.eval()

    # test false logical expression for either parser
    #################################################

    expr = "5 >= 6"

    # tensorflow-based parser
    p = ExpressionParser(expr_str=expr, args={}, backend=b)
    p.parse_expr()
    with tf.Session(graph=b) as sess:
        sess.run(p.expr_op)
        assert not p.expr_op.eval()


def test_1_5_expression_parser_indexing():
    """Testing handling of indexing operations by expression parsers:

    See Also
    --------
    :class:`ExpressionParser`: Detailed documentation of expression parser attributes and methods.
    :class:`LambdaExpressionParser`: Documentation of a non-symbolic expression parser.
    """

    # define backend
    b = TensorflowBackend()

    # define variables
    A = np.array(np.random.randn(10, 10), dtype=np.float32)
    B = np.eye(10, dtype=np.float32) == 1
    args = parse_dict({'A': {'vtype': 'constant', 'value': A, 'shape': A.shape, 'dtype': A.dtype},
                       'B': {'vtype': 'constant', 'value': B, 'shape': B.shape, 'dtype': B.dtype}},
                      backend=b)

    # define valid test cases
    indexed_expressions = [("A[:]", A[:]),               # single-dim indexing I
                           ("A[0]", A[0]),               # single-dim indexing II
                           ("A[-2]", A[-2]),             # single-dim indexing III
                           ("A[0:5]", A[0:5]),           # single-dim slicing I
                           ("A[-1:0:-1]", A[-1:0:-1]),   # single-dim slicing II
                           ("A[4,5]", A[4, 5]),          # two-dim indexing I
                           ("A[5,0:-2]", A[5, 0:-2]),    # two-dim indexing II
                           ("A[A > 0.]", A * (A > 0.)),  # boolean indexing
                           ("A[B]",                      # indexing with other array
                            np.eye(A.shape[0]) * A[np.where(B)]),
                           ("A[int32(2. * 2.):8 - 1]",   # using expressions as indices
                            A[(2 * 2):8 - 1]),
                           ]

    # test expression parsers on expression results
    ###############################################

    for expr, target in indexed_expressions:

        # tensorflow-based parser
        p = ExpressionParser(expr_str=expr, args={'vars': args}, backend=b)
        p.parse_expr()

        with tf.Session(graph=b) as sess:
            sess.run(p.expr_op)
            result = p.expr_op.eval()
        assert result == pytest.approx(target, rel=1e-6)

    # define invalid test cases
    ###########################

    indexed_expressions_wrong = ["A[1.2]",       # indexing with float variables
                                 "A[all]",       # indexing with undefined key words
                                 "A[-11]",       # index out of bounds
                                 "A[0:5:2:1]",   # too many arguments for slicing
                                 "A[-1::0:-1]",  # wrong slicing syntax II
                                 ]

    # test expression parsers on expression results
    ###############################################

    for expr in indexed_expressions_wrong:

        # tensorflow-based parser
        with pytest.raises((IndexError, ValueError, SyntaxError, TypeError, BaseException)):
            p = ExpressionParser(expr_str=expr, args={'vars': args}, backend=b)
            p.parse_expr()
            with tf.Session(graph=b) as sess:
                sess.run(p.expr_op)


def test_1_6_expression_parser_funcs():
    """Testing handling of function calls by expression parsers:

    See Also
    --------
    :class:`ExpressionParser`: Detailed documentation of expression parser attributes and methods.
    :class:`LambdaExpressionParser`: Documentation of a non-symbolic expression parser.
    """

    # define backend
    b = TensorflowBackend()

    # define variables
    A = np.array(np.random.randn(10, 10), dtype=np.float32)
    args = parse_dict({'A': {'vtype': 'constant', 'value': A, 'shape': A.shape, 'dtype': A.dtype}}, backend=b)

    # define valid test cases
    #########################

    expressions = [("abs(5.)", 5.),              # simple function call
                   ("abs(-5.)", 5.),             # function call of negative arg
                   ("abs(4. * -2. + 1)", 7.),    # function call on mathematical expression
                   ("int64(4 > 5)", 0),          # function call on boolean expression
                   ("abs(A[A > 0])",
                    np.abs(A * (A > 0))),        # function call on indexed variable
                   ("abs(sin(1.5))",
                    np.abs(np.sin(1.5))),        # nested function calls
                   ]

    # test expression parsers on expression results
    ###############################################

    for expr, target in expressions:

        # tensorflow-based parser
        p = ExpressionParser(expr_str=expr, args={'vars': args}, backend=b)
        p.parse_expr()

        with tf.Session(graph=b) as sess:
            sess.run(p.expr_op)
            result = p.expr_op.eval()
        assert result == pytest.approx(target, rel=1e-6)

    # define invalid test cases
    ###########################

    expressions_wrong = ["abs((4.)",       # wrong parentheses I
                         "abs[4.]",        # wrong parentheses II
                         "abs(0. True)",   # no comma separation on arguments
                         "abs(0.,1,5,3)",  # wrong argument number
                         ]

    # test expression parsers on expression results
    ###############################################

    for expr in expressions_wrong:

        # tensorflow-based parser
        with pytest.raises((IndexError, ValueError, SyntaxError, TypeError, BaseException)):
            p = ExpressionParser(expr_str=expr, args={'vars': args}, backend=b)
            p.parse_expr()


def test_1_7_equation_parsing():
    """Tests equation parsing functionalities.

    See Also
    --------
    :func:`parse_equation`: Detailed documentation of parse_equation arguments.
    """

    # define backend
    b = TensorflowBackend()

    # define test equations
    equations = ["a = 5. + 2.",              # simple update of variable
                 "d/dt * a = 5. + 2.",       # simple differential equation
                 ]

    # define equation variables
    a = np.zeros(shape=(), dtype=np.float32)
    args = {'a': {'vtype': 'state_var', 'value': a, 'shape': a.shape, 'dtype': a.dtype},
            'dt': {'vtype': 'constant', 'value': 0.1, 'shape': (), 'dtype': 'float32'}}
    arguments = [parse_dict(args, backend=b), parse_dict(args, backend=b)]

    # define equation results
    results = [7., 0.7]

    # test equation parser on different test equations
    ##################################################

    for eq, args, target in zip(equations, arguments, results):

        # tensorflow-based parsing
        args_new = parse_equation_list(equations=[eq], equation_args={'vars': args}, backend=b)
        with tf.Session(graph=b) as sess:
            sess.run(tf.global_variables_initializer())
            result = sess.run(args_new['updates']['a'])
        assert result == pytest.approx(target, rel=1e-6)
