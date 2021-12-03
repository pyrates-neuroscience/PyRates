"""Test suite for basic parser module functionality.
"""

# external _imports
import numpy as np
from copy import deepcopy
import pytest

# pyrates internal _imports
from pyrates.backend.parser import parse_equations
from pyrates.backend.parser import ExpressionParser
from pyrates.backend.computegraph import ComputeGraph

# meta infos
__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


###########
# Utility #
###########


def setup_module():
    print("\n")
    print("==============================")
    print("| Test Suite : Parser Module |")
    print("==============================")


# list parsers to be tested and their backends
backends = ['default']
parsers = [ExpressionParser]

# test accuracy
accuracy = 1e-4

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

    # test minimal minimal call example
    ###################################

    for Parser, backend in zip(parsers, backends):
        cg = ComputeGraph(backend=backend)
        parser = Parser("a + b", {'a': np.ones((3, 3)), 'b': 5.}, cg=cg)
        assert isinstance(parser, Parser)


def test_1_2_expression_parser_parsing_exceptions():
    """Testing error handling of different erroneous parser instantiations:

    See Also
    --------
    :class:`ExpressionParser`: Detailed documentation of expression parser attributes and methods.
    :class:`LambdaExpressionParser`: Documentation of a non-symbolic expression parser.
    """

    # test expected exceptions
    ##########################

    for Parser, backend in zip(parsers, backends):

        # undefined variables
        with pytest.raises(KeyError):
            cg = ComputeGraph(backend=backend)
            Parser("a + b", args={}, cg=cg).parse_expr()

        # wrong dimensionality of dictionary arguments
        args = {'a': {'vtype': 'constant', 'value': np.ones((3, 3)), 'dtype': 'float32', 'shape': (3, 3)},
                'b': {'vtype': 'constant', 'value': np.ones((4, 4)), 'dtype': 'float32', 'shape': (4, 4)}}
        with pytest.raises(ValueError):
            cg = ComputeGraph(backend=backend)
            Parser("a + b", args=args, cg=cg).parse_expr()
            cg.eval_graph()

        # wrong data type of dictionary arguments
        args = {'a': {'vtype': 'constant', 'value': np.ones((3, 3)), 'dtype': 'float32', 'shape': (3, 3)},
                'b': {'vtype': 'constant', 'value': np.ones((3, 3)), 'dtype': 'float32', 'shape': (3, 3)}}
        with pytest.raises(ValueError):
            cg = ComputeGraph(backend=backend)
            Parser("bool(a) + float32(b)", args=args, cg=cg).parse_expr()
            cg.eval_graph()

        # undefined mathematical operator
        args = {'a': {'vtype': 'constant', 'value': np.ones((3, 3)), 'dtype': 'float32', 'shape': (3, 3)},
                'b': {'vtype': 'constant', 'value': 5., 'dtype': 'float32', 'shape': ()}}
        with pytest.raises(ValueError):
            cg = ComputeGraph(backend=backend)
            Parser("a $ b", args=args, cg=cg).parse_expr()
            cg.eval_graph()


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
                        ("4.^2.", 16.),               # simple exponentiation
                        ("4. + -5.", -1.),            # negation of variable
                        ("4. * -2.", -8.),            # negation of variable in higher-order operation
                        ("4. + 5. * 2.", 14.),        # multiplication before addition
                        ("(4. + 5.) * 2.", 18.),      # parentheses before everything
                        ("4. * 5.^2.", 100.)          # exponentiation before multiplication
                        ]

    # test expression parser on expression results
    ##############################################

    # define backends
    for backend, parser in zip(backends, parsers):

        # evaluate expressions
        for expr, target in math_expressions:
            cg = ComputeGraph(backend=backend)
            p = parser(expr_str=expr, args={}, cg=cg)
            p.parse_expr()
            result = cg.eval_node(cg.var_updates['non-DEs']['x'])
            assert result == pytest.approx(target, rel=accuracy, abs=accuracy)


def test_1_4_expression_parser_funcs():
    """Testing handling of function calls by expression parsers:

    See Also
    --------
    :class:`ExpressionParser`: Detailed documentation of expression parser attributes and methods.
    :class:`LambdaExpressionParser`: Documentation of a non-symbolic expression parser.
    """

    # define variables
    A = np.array(np.random.randn(10, 10), dtype=np.float32)
    args = {'A': {'vtype': 'constant', 'value': A, 'shape': A.shape, 'dtype': A.dtype}}

    # define valid test cases
    expressions = [("abs(5.)", 5.),            # simple function call
                   ("abs(-5.)", 5.),           # function call of negative arg
                   ("abs(4. * -2. + 1)", 7.),  # function call on mathematical expression
                   ("int64(4 > 5)", 0),        # function call on boolean expression
                   ("abs(index(A, 2))",
                    np.abs(A[2])),             # function call on indexed variable
                   ("abs(sin(1.5))",
                    np.abs(np.sin(1.5))),      # nested function calls
                   ]

    # define invalid test cases
    expressions_wrong = ["abs((4.)",       # wrong parentheses I
                         "abs[4.]",        # wrong parentheses II
                         "abs(0. True)",   # no comma separation on arguments
                         "abs(0.,1,5,3)",  # wrong argument number
                         ]

    # test function calling on different backends
    #############################################

    for backend, parser in zip(backends, parsers):

        # start testing: valid cases
        for expr, target in expressions:
            cg = ComputeGraph(backend=backend)
            parser(expr_str=expr, args=deepcopy(args), cg=cg).parse_expr()
            result = cg.eval_node(cg.var_updates['non-DEs']['x'])
            assert result == pytest.approx(target, rel=accuracy, abs=accuracy)

        # invalid cases
        for expr in expressions_wrong[:-1]:
            with pytest.raises((IndexError, ValueError, SyntaxError, TypeError, BaseException)):
                cg = ComputeGraph(backend=backend)
                parser(expr_str=expr, args=deepcopy(args), cg=cg).parse_expr()
                cg.eval_node(cg.var_updates['non-DEs']['x'])


def test_1_5_expression_parser_indexing():
    """Testing handling of indexing operations by expression parsers:

    See Also
    --------
    :class:`ExpressionParser`: Detailed documentation of expression parser attributes and methods.
    :class:`LambdaExpressionParser`: Documentation of a non-symbolic expression parser.
    """

    # test indexing ability of tensorflow-based parser
    ##################################################

    A = np.array(np.random.randn(10, 10), dtype=np.float32)
    B = np.asarray([0, 2, 4], dtype=np.int32)
    C = np.asarray([1, 3, 5], dtype=np.int32)
    arg_dict = {'A': {'vtype': 'constant', 'value': A, 'shape': A.shape, 'dtype': A.dtype},
                'B': {'vtype': 'constant', 'value': B, 'shape': B.shape, 'dtype': B.dtype},
                'C': {'vtype': 'constant', 'value': C, 'shape': C.shape, 'dtype': C.dtype},
                'd': {'vtype': 'constant', 'value': 4, 'shape': (), 'dtype': 'int32'}}

    # define valid test cases
    indexed_expressions = [
        ("index(A, 0)", A[0]),                    # single-dim indexing I
        ("index(A, 9)", A[9]),                    # single-dim indexing II
        ("index(A, B)", A[B]),                    # single-dim indexing III
        ("index_range(A, 0, 5)", A[0:5]),         # slicing I
        ("index_range(A, d, 8-1)", A[4:8 - 1]),   # slicing II
        ("index_axis(A)", A[:]),                  # slicing III
        ("index_2d(A, 4, 5)", A[4, 5]),           # two-dim indexing I
        ("index_2d(A, B, 1)", A[B, 1]),           # two-dim indexing II
        ("index_2d(A, B, C)", A[B, C]),           # two-dim indexing III
        ("index_axis(A, B, 1)", A[:, B]),         # two-dim indexing IV
    ]

    # define invalid test cases
    indexed_expressions_wrong = ["index(A, 1.2)",           # wrong data type of index
                                 "index(A, all)",           # non-existing indexing variable
                                 "index(A, 11)",            # index out of bounds
                                 "index_2d(A, 0, 5, 1)",    # too many arguments for indexing
                                 "A[::-1]",                 # wrong parser syntax
                                 ]

    # test indexing ability of numpy-based parser
    #############################################

    for backend, parser in zip(backends, parsers):

        # test expression parsers on expression results
        for expr, target in indexed_expressions:
            cg = ComputeGraph(backend=backend)
            parser(expr_str=expr, args=deepcopy(arg_dict), cg=cg).parse_expr()
            result = cg.eval_node(cg.var_updates['non-DEs']['x'])
            assert result == pytest.approx(target, rel=accuracy, abs=accuracy)

        # test expression parsers on expression results
        for expr in indexed_expressions_wrong:
            with pytest.raises((IndexError, ValueError, SyntaxError, TypeError, BaseException)):
                cg = ComputeGraph(backend=backend)
                parser(expr_str=expr, args=deepcopy(arg_dict), cg=cg).parse_expr()
                cg.eval_node(cg.var_updates['non-DEs']['x'])


def test_1_7_equation_parsing():
    """Tests equation parsing functionalities.

    See Also
    --------
    :func:`parse_equation`: Detailed documentation of parse_equation arguments.
    """

    # define test equations
    equations = ["a = 5. + 2.",              # simple update of variable
                 "d/dt * a = 5.**2",       # simple differential equation
                 ]

    # define equation variables
    a = np.zeros(shape=(1,), dtype=np.float32)
    args = {'node/op/a': {'vtype': 'state_var', 'value': a, 'shape': a.shape, 'dtype': a.dtype},
            }

    # define equation results
    results = [7., 25.0]
    result_vars = [('non-DEs', 'a'), ('DEs', 'a')]

    # test equation solving of parser
    #################################

    for backend, parser in zip(backends, parsers):

        # test equation parser on different test equations
        for eq, target, tvar in zip(equations, results, result_vars):

            # numpy-based parsing
            cg = ComputeGraph(backend=backend)
            parse_equations(equations=[(eq, 'node/op')], equation_args=deepcopy(args), cg=cg, def_shape=())
            result = cg.eval_node(cg.var_updates[tvar[0]][tvar[1]])
            assert result == pytest.approx(target, rel=accuracy, abs=accuracy)
