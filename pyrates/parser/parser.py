"""

"""

# external imports
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.function import UndefinedFunction
from sympy import Expr, lambdify, symbols, MatrixSymbol
import tensorflow as tf
from typing import Tuple, Union

# meta infos
__author__ = "Richard Gast"
__status__ = "Development"


class RHSParser(object):
    """Parses right-hand side of an equation.

    Parameters
    ----------
    expression
        Right-hand side of an equation in string format.
    args
        Dictionary containing all variables and functions needed to evaluate the expression.
    tf_graph
        Tensorflow graph on which all operations will be created.

    """

    def __init__(self, expression: str, args: dict, tf_graph: tf.Graph):
        """Instantiates RHSParser.
        """

        self.expression = parse_expr(expression, evaluate=False)
        self.args = args
        self.tf_graph = tf_graph
        self.tf_op = tf.no_op()
        self.tf_op_args = []
        self.lambdify_args = []

    def parse(self) -> Union[tf.Operation, tf.Tensor]:
        """Turns the expression into a runnable tensorflow operation.

        Parameters
        ----------

        Returns
        -------
        Union[tf.Operator, tf.Tensor]
            Tensorflow operation that represents rhs.

        """
        with self.tf_graph.as_default():

            tf_ops, custom_funcs = self.custom_funcs_to_tf_ops(self.expression)

            for i, (tf_op, func) in enumerate(zip(tf_ops, custom_funcs)):
                new_exp = symbols('var_' + str(i))
                self.expression = self.expression.subs(func, new_exp)
                self.args[str(new_exp)] = tf_op

            for i, symb in enumerate(self.expression.free_symbols):
                symb_name = str(symb)
                if symb_name not in self.args.keys():
                    raise ValueError(symb_name + ' must be defined in args!')
                if isinstance(self.args[symb_name], tf.Operation) or \
                        isinstance(self.args[symb_name], tf.Tensor) or \
                        isinstance(self.args[symb_name], tf.Variable):
                    arg = self.args[symb_name]
                else:
                    arg = tf.constant(self.args[symb_name], name=symb_name)
                if len(arg.shape) > 1:
                    new_symb = MatrixSymbol(symb_name, arg.shape[0], 1)
                elif arg.shape > 1:
                    new_symb = MatrixSymbol(symb_name, arg.shape[0], arg.shape[1])
                else:
                    new_symb = symbols(symb_name)
                self.lambdify_args.append(new_symb)
                self.expression.subs(symb, new_symb)
                self.tf_op_args.append(arg)

        func = lambdify(args=tuple(self.lambdify_args), expr=self.expression, modules='tensorflow')

        return func(*tuple(self.tf_op_args))

    def custom_funcs_to_tf_ops(self, expression: Expr) -> Tuple[list, list]:
        """Turns all custom functions in the expression into tensorflow operations.

        Parameters
        ----------
        expression
            Right-hand side of an equation.

        Returns
        -------
        Tuple
            Tuple of two lists. First list contains the tensorflow operations, the second contains the expressions
            that were turned into tensorflow operations.

        """
        func_name = expression.func.class_key()[-1]
        funcs = []
        tf_ops = []

        if isinstance(expression.func, UndefinedFunction):
            if func_name not in self.args.keys():
                raise ValueError(func_name + ' must be defined in args!')
            parser = RHSParser(self.args[func_name], self.args, self.tf_graph)
            tf_op, tf_op_args = parser.parse()
            tf_ops.append(tf_op)
            funcs.append(expression)
            for arg in tf_op_args:
                self.args[arg.name] = arg

        if len(expression.args) > 0:
            for expr in expression.args:
                tf_ops_tmp, funcs_tmp = self.custom_funcs_to_tf_ops(expr)
                tf_ops += tf_ops_tmp
                funcs += funcs_tmp

        return tf_ops, funcs


class LHSParser(object):
    """Parses left-hand side of an equation.

    Parameters
    ----------
    expression
        Right-hand side of an equation in string format.
    args
        Dictionary containing all variables and functions needed to evaluate the expression.
    tf_graph
        Tensorflow graph on which all operations will be created.

    """

    def __init__(self, expression: str, args: dict, rhs: Union[tf.Operation, tf.Tensor], tf_graph: tf.Graph):
        """Instantiates RHSParser.
        """

        self.expression = parse_expr(expression, evaluate=False)
        self.args = args
        self.tf_graph = tf_graph
        self.rhs = rhs

    def parse(self) -> Tuple[Union[tf.Operation, tf.Tensor], Union[tf.Variable, tf.Tensor]]:
        """Turns the expression into a runnable tensorflow operation.

        Parameters
        ----------

        Returns
        -------
        tensorflow_object
            Either a tensorflow operation (if its a differential equation) or a tensorflow variable
            (the state variable to be updated by a right-hand side of an equation).

        """

        if self.expression.find('d') and self.expression.find('dt'):

            from pyrates.solver import Solver

            if len(self.expression.free_symbols) > 3:
                raise AttributeError('The left-hand side of an expression needs to be of the form `out_var` or '
                                     '`d/dt out_var`!')
            if 'dt' not in self.args.keys():
                raise AttributeError('The step-size `dt` has to be passed with `args` for differential equations.')

            for out_var in self.expression.free_symbols:

                out_var_name = str(out_var)

                if (out_var_name != 'd') and (out_var_name != 'dt'):

                    if out_var_name not in self.args.keys():
                        raise AttributeError('Output variables must be included in expression_args dictionary.')

                    solver = Solver(self.rhs, self.args[out_var_name], self.args['dt'], self.tf_graph)
                    tf_op = solver.solve()

                    break

        else:

            if len(self.expression.free_symbols) > 1:
                raise AttributeError('The left-hand side of an expression needs to be of the form `out_var` or '
                                     '`d/dt out_var`!')

            out_var = self.expression.free_symbols.pop()
            out_var_name = str(out_var)

            if out_var_name not in self.args.keys():
                raise AttributeError('Output variables must be included in expression_args dictionary.')

            with self.tf_graph.as_default():
                tf_op = self.args[out_var_name].assign(self.rhs)

        return tf_op, self.args[out_var_name]


class EquationParser(object):

    def __init__(self, eq: str):

        lhs, rhs = eq.split('=')

        eq_parts = [lhs, rhs]
        self.operations = []

        for eq_part in eq_parts:

            expr = parse_expr(eq_part, evaluate=False)
            self.get_operations(expr)

    def get_operations(self, expr: Expr):

        self.operations.append(expr.func)

        for arg in expr.args:
            self.get_operations(arg)
