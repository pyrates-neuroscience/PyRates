"""Test suite for basic parser module functionality.
"""

# external imports
import numpy as np
from copy import deepcopy
import pytest

# pyrates internal imports
from pyrates.backend.parser import parse_equations
from pyrates.backend.parser import SympyParser
from pyrates.backend.base_backend import BaseBackend
from pyrates.backend.fortran_backend import FortranBackend

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
    parsers = [SympyParser, SympyParser]
    backends = [BaseBackend, FortranBackend]

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
    parsers = [SympyParser, SympyParser]
    backends = [BaseBackend, FortranBackend]

    # test expected exceptions
    ##########################

    for Parser, backend in zip(parsers, backends):

        b = backend()

        # undefined variables
        with pytest.raises(KeyError):
            Parser("a + b", args={}, backend=b).parse_expr()

        # wrong dimensionality of dictionary arguments
        args = {'a': {'vtype': 'constant', 'value': np.ones((3, 3)), 'dtype': 'float32', 'shape': (3, 3)},
                'b': {'vtype': 'constant', 'value': np.ones((4, 4)), 'dtype': 'float32', 'shape': (4, 4)}}
        with pytest.raises(ValueError):
            Parser("a + b", args=args, backend=b).parse_expr()
            b.graph.eval_graph()

        # wrong data type of dictionary arguments
        args = {'a': {'vtype': 'constant', 'value': np.ones((3, 3)), 'dtype': 'float32', 'shape': (3, 3)},
                'b': {'vtype': 'constant', 'value': np.ones((3, 3)), 'dtype': 'float32', 'shape': (3, 3)}}
        with pytest.raises(ValueError):
            Parser("bool(a) + float32(b)", args=args, backend=b).parse_expr()
            b.graph.eval_graph()

        # undefined mathematical operator
        args = {'a': {'vtype': 'constant', 'value': np.ones((3, 3)), 'dtype': 'float32', 'shape': (3, 3)},
                'b': {'vtype': 'constant', 'value': 5., 'dtype': 'float32', 'shape': ()}}
        with pytest.raises(ValueError):
            Parser("a $ b", args=args, backend=b).parse_expr()
            b.graph.eval_graph()

        # undefined function
        args = {'a': {'vtype': 'constant', 'value': np.ones((3, 3)), 'dtype': 'float32', 'shape': (3, 3)},
                'b': {'vtype': 'constant', 'value': 5., 'dtype': 'float32', 'shape': ()}}
        with pytest.raises((KeyError, IndexError)):
            Parser("a / b(5.)", args=args, backend=b).parse_expr()
            b.graph.eval_graph()


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
    for backend in [BaseBackend, FortranBackend]:

        b = backend()

        # evaluate expressions
        for expr, target in math_expressions:

            p = SympyParser(expr_str=expr, args={}, backend=b)
            p.parse_expr()
            cg = p.backend.graph
            result = cg.eval_node(cg.var_updates['non-DEs']['x'])
            b.clear()
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

    # test expression parsers on expression results
    ###############################################

    # define backends
    for backend in [BaseBackend, FortranBackend]:

        b = backend()

        # evaluate expressions
        for expr in logic_expressions:

            p = SympyParser(expr_str=expr, args={}, backend=b)
            p.parse_expr()
            cg = p.backend.graph
            result = cg.eval_node(cg.var_updates['non-DEs']['x'])
            b.clear()
            assert result

        # false logical expression
        expr = "5 >= 6"

        # numpy-based parser
        p = SympyParser(expr_str=expr, args={}, backend=b)
        p.parse_expr()
        cg = b.graph
        result = cg.eval_node(cg.var_updates['non-DEs']['x'])
        b.clear()
        assert not result


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
    B = np.eye(10, dtype=np.float32) == 1
    arg_dict = {'A': {'vtype': 'constant', 'value': A, 'shape': A.shape, 'dtype': A.dtype},
                'B': {'vtype': 'constant', 'value': B, 'shape': B.shape, 'dtype': B.dtype},
                'd': {'vtype': 'constant', 'value': 4, 'Shape': (), 'dtype': 'int32'}}

    # define valid test cases
    indexed_expressions = [("index_axis(A)", A[:]),                  # single-dim indexing I
                           ("index(A, 0)", A[0]),                    # single-dim indexing II
                           ("index(A, 9)", A[9]),                    # single-dim indexing III
                           ("index_range(A, 0, 5)", A[0:5]),         # single-dim slicing I
                           ("index_2d(A, 4, 5)", A[4, 5]),           # two-dim indexing I
                           ("index(A, A > 0.)", A[A > 0.]),          # boolean indexing
                           ("index(A, B)", A[B]),                    # indexing with other array
                           ("index_range(A, d, 8-1)",  A[4:8 - 1]),  # using variables as indices
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

    for backend in [BaseBackend, FortranBackend]:

        b = backend()

        # test expression parsers on expression results
        for expr, target in indexed_expressions:
            p = SympyParser(expr_str=expr, args=deepcopy(arg_dict), backend=b)
            p.parse_expr()
            cg = b.graph
            result = cg.eval_node(cg.var_updates['non-DEs']['x'])
            b.clear()
            assert result == pytest.approx(target, rel=1e-6)

        # test expression parsers on expression results
        for expr in indexed_expressions_wrong:
            with pytest.raises((IndexError, ValueError, SyntaxError, TypeError, BaseException)):
                p = SympyParser(expr_str=expr, args=deepcopy(arg_dict), backend=b)
                p.parse_expr()
                cg = b.graph
                cg.eval_node(cg.var_updates['non-DEs']['x'])
                b.clear()


def test_1_6_expression_parser_funcs():
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

    for backend in [BaseBackend, FortranBackend]:

        b = backend()

        # start testing: valid cases
        for expr, target in expressions:
            SympyParser(expr_str=expr, args=deepcopy(args), backend=b).parse_expr()
            cg = b.graph
            result = cg.eval_node(cg.var_updates['non-DEs']['x'])
            b.clear()
            assert result == pytest.approx(target, rel=1e-6)

        # invalid cases
        for expr in expressions_wrong[:-1]:
            with pytest.raises((IndexError, ValueError, SyntaxError, TypeError, BaseException)):
                SympyParser(expr_str=expr, args=deepcopy(args), backend=b).parse_expr()
                cg = b.graph
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

    for backend in [BaseBackend, FortranBackend]:

        b = backend()

        # test equation parser on different test equations
        for eq, target, tvar in zip(equations, results, result_vars):

            # numpy-based parsing
            parse_equations(equations=[(eq, 'node/op')], equation_args=deepcopy(args), backend=b)
            cg = b.graph
            result = cg.eval_node(cg.var_updates[tvar[0]][tvar[1]])
            b.clear()
            assert result == pytest.approx(target, rel=1e-4)
