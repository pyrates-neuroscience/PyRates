"""

"""

# external imports
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.function import UndefinedFunction
from sympy import Expr, lambdify, symbols, MatrixSymbol
from sympy.matrices.expressions.matexpr import MatrixElement
import tensorflow as tf
from typing import Tuple, Union, Optional

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

        # make sure that sliced vectors/matrices are replaced with the appropriate sympy expressions in expression
        ##########################################################################################################

        if expression.find('['):
            expression_unsliced = expression
            for sub_expr in expression_unsliced.split(' '):
                idx_start = sub_expr.find('[')
                if idx_start:
                    sub_expr_new = sub_expr[0:idx_start]
                    expression_unsliced.replace(sub_expr, sub_expr_new)
            self.expression_unsliced = expression_unsliced
            self.sliced_expr = True
        else:
            self.sliced_expr = False

        for key, arg in args.items():
            if len(arg.shape) == 1:
                expression.replace(key, f"""MatrixSymbol('{key}', {arg.shape[0]}, 1)""")
            elif len(arg.shape) == 2:
                expression.replace(key, f"""MatrixSymbol('{key}', {arg.shape[0]}, {arg.shape[1]})""")

        # parse expression and initialize important variables
        #####################################################

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

            # turn custom functions into tensorflow operations
            ##################################################

            tf_ops, custom_funcs = self.custom_funcs_to_tf_ops(self.expression)

            # replace the custom functions in the expression with the tensorflow operation results
            ######################################################################################

            for i, (tf_op, func) in enumerate(zip(tf_ops, custom_funcs)):
                new_exp = symbols('var_' + str(i))
                self.expression = self.expression.subs(func, new_exp)
                self.args[str(new_exp)] = tf_op

            # collect all arguments needed to transform the expression into a tensorflow operation
            ######################################################################################

            # go through all variables of the expression
            for i, symb in enumerate(self.expression.free_symbols):

                symb_name = str(symb)

                # check whether the variable was passed with the args dictionary
                if symb_name not in self.args.keys():
                    raise ValueError(symb_name + ' must be defined in args!')

                arg = self.args[symb_name]

                # check whether the variable is a scalar or some kind of tensor and inform sympy about it
                if len(arg.shape) > 1:
                    new_symb = MatrixSymbol(symb_name, arg.shape[0], arg.shape[1])
                elif len(arg.shape) == 1 and arg.shape[0] > 1:
                    new_symb = MatrixSymbol(symb_name, arg.shape[0], 1)
                else:
                    new_symb = symbols(symb_name)
                self.expression.subs(symb, new_symb)

                # collect the arguments
                self.lambdify_args.append(new_symb)
                self.tf_op_args.append(arg)

        # turn expression into tensorflow function
        ##########################################

        if self.sliced_expr:

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

        if len(expression.args) > 0 and expression.func != MatrixElement:
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

        # make sure that vectors/matrices are replaced with the appropriate sympy expressions in expression
        ###################################################################################################

        for key, arg in args.items():
            if len(arg.shape) == 1:
                expression = expression.replace(key, f"""MatrixSymbol('{key}', {arg.shape[0]}, 1)""")
            elif len(arg.shape) == 2:
                expression = expression.replace(key, f"""MatrixSymbol('{key}', {arg.shape[0]}, {arg.shape[1]})""")

        # parse expression and initialize important variables
        #####################################################

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

        if 'dt' in self.expression.free_symbols:

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

            #if len(self.expression.free_symbols) > 1:
            #    raise AttributeError('The left-hand side of an expression needs to be of the form `out_var` or '
            #                         '`d/dt out_var`!')

            with self.tf_graph.as_default():

                if self.expression.func == MatrixElement:

                    indices = []
                    index_names = []
                    for symb in self.expression.free_symbols:

                        if str(symb) != 'd' and str(symb) != 'dt':

                            out_var = symb
                            out_var_name = str(out_var)
                            if out_var_name not in self.args.keys():
                                raise AttributeError('Output variables must be included in expression_args dictionary.')

                        else:

                            indices.append(symb)
                            index_names.append(str(symb))

                    if len(indices) == 1:
                        tf_op = self.args[out_var_name][self.args[index_names[0]]].assign(self.rhs)
                    elif len(indices) == 2:
                        tf_op = self.args[out_var_name][self.args[index_names[0]],
                                                        self.args[index_names[1]]].assign(self.rhs)
                    else:
                        raise ValueError('Currently, only 2d state variables are supported.')

                else:

                    out_var = self.expression.free_symbols.pop()
                    out_var_name = str(out_var)
                    if out_var_name not in self.args.keys():
                        raise AttributeError('Output variables must be included in expression_args dictionary.')

                    tf_op = self.args[out_var_name].assign(self.rhs)

        return tf_op, self.args[out_var_name]


class EquationParser(object):

    def __init__(self, eq: str):

        lhs, rhs = eq.split('=')

        eq_parts = [lhs, rhs]
        self.operations = []
        self.variables = []

        for eq_part in eq_parts:

            idx = 0
            while idx < len(eq_part):
                start = eq_part[idx:].find('[')
                if start == -1:
                    break
                end = eq_part[idx:].find(']') + 1
                eq_part = eq_part[0:start] + eq_part[end:]
                idx += end

            expr = parse_expr(eq_part, evaluate=False)
            self.get_operations(expr)
            self.get_variables(expr)

    def get_operations(self, expr: Expr):

        self.operations.append(expr.func)

        for arg in expr.args:
            self.get_operations(arg)

    def get_variables(self, expr: Expr):

        self.variables.append([str(symb) for symb in expr.free_symbols])


def parse_dict(var_dict: dict, var_scope: str, tf_graph: Optional[tf.Graph] = None):

    tf_graph = tf_graph if tf_graph else tf.get_default_graph()

    with tf_graph.as_default():

        with tf.variable_scope(var_scope):

            data_types = {'float16': tf.float16,
                          'float32': tf.float32,
                          'float64': tf.float64,
                          'int16': tf.int16,
                          'int32': tf.int32,
                          'int64': tf.int64,
                          'double': tf.double,
                          'complex64': tf.complex64,
                          'complex128': tf.complex128,
                          'string': tf.string,
                          'bool': tf.bool}

            tf_vars = []
            var_names = []

            for var_name, var in var_dict.items():

                if isinstance(var, tf.Variable) or isinstance(var, tf.Tensor) or isinstance(var, tf.Operation) \
                        or isinstance(var, tf.IndexedSlices):

                    tf_var = var

                elif var['variable_type'] == 'state_variable':

                    tf_var = tf.get_variable(name=var['name'],
                                             shape=var['shape'],
                                             dtype=data_types[var['data_type']],
                                             initializer=tf.constant_initializer(var['initial_value'])
                                             )

                elif var['variable_type'] == 'constant':

                    tf_var = tf.constant(value=var['initial_value'],
                                         name=var['name'],
                                         shape=var['shape'],
                                         dtype=data_types[var['data_type']]
                                         )

                elif var['variable_type'] == 'placeholder':

                    tf_var = tf.placeholder(name=var['name'],
                                            shape=var['shape'],
                                            dtype=data_types[var['data_type']]
                                            )

                else:

                    raise AttributeError('Variable type of each variable needs to be either `state_variable`,'
                                         ' `constant` or `placeholder`.')

                tf_vars.append(tf_var)
                var_names.append(var_name)

    return tf_vars, var_names

