"""

"""

# external imports
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.function import UndefinedFunction
from sympy import Expr, lambdify, symbols, MatrixSlice, MatrixSymbol, Symbol
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

    def __init__(self, expression: str, args: dict, tf_graph: tf.Graph, matrix_expr: bool = True):
        """Instantiates RHSParser.
        """

        # initialization of instance fields
        ###################################

        self.expression = expression
        self.args = args
        self.tf_graph = tf_graph
        self.tf_op = tf.no_op()
        self.tf_op_args = []
        self.lambdify_args = []

        # parse expression
        ##################

        if matrix_expr:
            simplified_expr = self.to_simple_expr()
            self.expression = self.to_matrix_expr(simplified_expr)
        else:
            self.expression = parse_expr(self.expression)

    def to_simple_expr(self) -> str:
        """Parses expression and replaces all matrix-based operations with scalar operations.
        """

        expr = self.expression

        # remove/replace matrix-based operations from expression
        ########################################################

        matrix_ops = {'@': '*',
                      '.T': '',
                      '.I': '**(-1)'
                      }

        for op, subs in matrix_ops.items():
            expr = expr.replace(op, subs)

        # remove indices from expression
        ################################

        idx = 0
        while idx < len(expr):
            start = expr[idx:].find('[')
            if start != -1:
                end = expr[idx:].find(']') + 1
                expr = expr[idx:start] + expr[end:]
                idx = end
            else:
                break

        return expr

    def to_matrix_expr(self, simple_expr: str):
        """Parses expression and makes it compatible with sympy matrix operations.
        """

        expr = parse_expr(simple_expr)

        element_wise_ops = {'is_Mul': {'func': 'custom_mult()',
                                       'arg': {'variable_type': 'tensorflow',
                                               'func': tf.multiply}
                                       },
                            'is_Pow': {'func': 'custom_power()',
                                       'arg': {'variable_type': 'tensorflow',
                                               'func': tf.pow}
                                       },
                            'is_Add': {'func': 'custom_add()',
                                       'arg': {'variable_type': 'tensorflow',
                                               'func': tf.add}
                                       },
                            'idx': {'func': self.apply_idx},
                            'transp': {'arg': {'variable_type': 'tensorflow',
                                               'func': tf.transpose}
                                       },
                            'inv': {'arg': {'variable_type': 'tensorflow',
                                            'func': tf.matrix_inverse}
                                    },
                            'dot': {'arg': {'variable_type': 'tensorflow',
                                            'func': tf.matmul}}
                            }

        return self.replace(expr, element_wise_ops)

    def apply_idx(self, args: list) -> str:
        """Applies custom idx function to arguments
        """

        idx_str = f"{self.symbol_to_str(args[0])}["
        for i in range(1, len(args)):
            idx_str += f"{self.symbol_to_str(args[i])}, "

        return f"{idx_str[:-2]}]"

    def replace(self, expr: Expr, replacement_dict: dict, to_matrix: bool = True) -> Expr:
        """Recursively goes through expr and replaces the keys in dict with the values.

        Parameters
        ----------
        expr
        replacement_dict

        Returns
        -------

        """

        func = expr.func
        args = expr.args

        if len(args) > 0:

            # go recursively through expression
            ###################################

            new_args = []
            for arg in args:
                new_args.append(self.replace(arg, replacement_dict))

            # replace func if necessary
            ###########################

            for i, (key, new_func) in enumerate(replacement_dict.items()):

                if 'is_' in key and getattr(func, key):

                    base_str = new_func['func'][0:-1]

                    if new_func['func'].find('(') != -1:

                        func_str = ''
                        if len(new_args) > 2:

                            for j, arg in enumerate(new_args):
                                if j < len(new_args) - 1:
                                    func_str += f"{base_str} {self.symbol_to_str(arg)}, "
                                else:
                                    func_str += f"{self.symbol_to_str(arg)})"
                            func_str = func_str[0:-2]
                            for _ in range(j + 1):
                                func_str += ')'

                        else:

                            func_str = f"{base_str} {self.symbol_to_str(new_args[0])}, " \
                                       f"{self.symbol_to_str(new_args[1])})"

                        self.args[new_func['func'][0:-2]] = new_func['arg']

                    else:

                        func_str = f"{self.symbol_to_str(new_args[0])} " \
                                   f"{new_func['func']} " \
                                   f"{self.symbol_to_str(new_args[1])}"

                    new_expr = parse_expr(func_str)

                    break

                elif key == str(func):

                    if 'func' in new_func.keys():
                        func_str = new_func['func'](new_args)
                        new_expr = parse_expr(func_str)
                    else:
                        new_expr = expr

                    if 'arg' in new_func.keys():
                        self.args[key] = new_func['arg']

                elif i < len(replacement_dict) - 1:

                    new_expr = expr

                else:

                    new_expr = func(*tuple(new_args))

        else:

            # replace simple symbols with matrix symbols where necessary
            ############################################################

            if str(expr) in self.args.keys():

                arg = self.args[str(expr)]
                if not hasattr(arg, 'shape'):
                    new_expr = expr
                elif len(arg.shape) > 1:
                    new_expr = MatrixSymbol(str(expr), arg.shape[0], arg.shape[1])
                elif len(arg.shape) == 1 and arg.shape[0] > 1:
                    new_expr = MatrixSymbol(str(expr), arg.shape[0], 1)
                else:
                    new_expr = expr

            else:

                new_expr = expr

        return new_expr

    def symbol_to_str(self, symb: Union[MatrixSymbol, Symbol]) -> str:
        """Turns sympy symbol into string representation.

        Parameters
        ----------
        symb

        Returns
        -------

        """

        if isinstance(symb, MatrixSymbol):
            symb_str = f"MatrixSymbol('{symb}', {symb.shape[0]}, {symb.shape[1]})"
        else:
            symb_str = str(symb)

        return symb_str

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

                # collect the arguments
                self.lambdify_args.append(symb)
                self.tf_op_args.append(self.args[symb_name])

        # turn expression into tensorflow function
        ##########################################

        func = lambdify(args=tuple(self.lambdify_args), expr=self.expression,
                        modules=[{'ImmutableDenseMatrix': tf.Variable}, 'tensorflow'])

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

        # extract the name of the operation
        func_name = expression.func.class_key()[-1]

        funcs = []
        tf_ops = []

        # parsing of custom functions
        #############################

        if isinstance(expression.func, UndefinedFunction):

            if func_name not in self.args.keys():
                raise ValueError(func_name + ' must be defined in args!')

            # extract custom function definition
            new_expr = self.args[func_name]

            # parsing of tensorflow functions
            #################################

            if new_expr['variable_type'] == 'tensorflow':

                func_args = []

                for arg in expression.args:

                    if str(arg) in self.args:

                        # extract argument from args
                        new_arg = self.args[str(arg)]
                        new_arg = new_arg['func'] if type(new_arg) is dict else new_arg
                        func_args.append(new_arg)

                    elif arg.is_Number:

                        # turn constant into tensorflow constant
                        func_args.append(tf.constant(float(arg), dtype=tf.float32))

                    else:

                        # create new instance of RHSParser and create tensorflow operation from arg
                        parser = RHSParser(str(arg), self.args, self.tf_graph)
                        tf_op_tmp = parser.parse()
                        func_args.append(tf_op_tmp)

                # create tensorflow operation from function
                tf_op = new_expr['func'](*tuple(func_args))

                tf_ops.append(tf_op)
                funcs.append(expression)

            # parsing of string-based functions
            ###################################

            elif new_expr['variable_type'] == 'str':

                # create new instance of RHSParser and create tensorflow operation from string
                parser = RHSParser(self.args[func_name]['func'], self.args, self.tf_graph)
                tf_op = parser.parse()

                tf_ops.append(tf_op)
                funcs.append(expression)

            else:

                raise ValueError('Custom functions can only be of type `tensorflow` or `str`.')

        # if multiple arguments exist that are connected by func, call this function recursively on each argument
        #########################################################################################################

        if len(expression.args) > 0 and expression.func != MatrixElement and expression.func != MatrixSlice \
                and not isinstance(expression, MatrixSymbol):

            for expr in expression.args:

                # recursive call
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

        # initialize variables
        ######################

        self.expression = expression
        self.args = args
        self.tf_graph = tf_graph
        self.rhs = rhs
        self.indices = []

        # parse expression
        ##################

        simplified_expr = self.to_simple_expr()
        self.expression = parse_expr(simplified_expr)

    def to_simple_expr(self) -> str:
        """Parses expression and removes indices.
        """

        expr = self.expression
        idx = 0
        while idx < len(expr):
            start = expr[idx:].find('[')
            if start != -1:
                end = expr[idx:].find(']') + 1
                self.add_to_indices(expr[start+1:end-1])
                expr = expr[idx:start] + expr[end:]
                idx = end
            else:
                break

        return expr

    def add_to_indices(self, index_str: str):
        """Adds each variable in index_str (comma-separated) to self.indices.

        Parameters
        ----------
        index_str

        Returns
        -------

        """

        for idx in index_str.split(','):
            if idx == ':':
                self.indices.append(':')
            else:
                self.indices.append(parse_expr(idx))

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

        if symbols('dt') in self.expression.free_symbols:

            from pyrates.solver import Solver

            if len(self.expression.free_symbols) > 3:
                raise AttributeError('The left-hand side of an expression needs to be of the form `out_var` or '
                                     '`d/dt out_var`!')
            if 'dt' not in self.args.keys():
                raise AttributeError('The step-size `dt` has to be passed with `args` for differential equations.')

            for var in self.expression.free_symbols:

                var_name = str(var)

                if var_name != 'd' and var_name != 'dt':

                    if var_name not in self.args.keys():
                        raise AttributeError('Output variables must be included in expression_args dictionary.')

                    out_var = self.apply_indices(self.args[var_name])
                    solver = Solver(self.rhs, out_var, self.args['dt'], self.tf_graph)
                    tf_op = solver.solve()

                    break

        else:

            with self.tf_graph.as_default():

                for var in self.expression.free_symbols:

                    var_name = str(var)

                    if var_name != 'd' and var_name != 'dt':

                        if var_name not in self.args.keys():
                            raise AttributeError('Output variables must be included in expression_args dictionary.')

                        break

                out_var = self.apply_indices(self.args[var_name])
                tf_op = out_var.assign(self.rhs)

        return tf_op, out_var

    def apply_indices(self, out_var) -> Union[tf.Variable, tf.Tensor, tf.Operation]:
        """Applies stored indices to tensor.

        Parameters
        ----------
        out_var

        Returns
        -------

        """

        if type(out_var) is dict:
            out_var = out_var['func']

        # go trough indices and turn them into integers or tensors
        ##########################################################

        new_indices = []
        for idx in self.indices:
            if idx == ':':
                new_indices.append(':')
            elif str(idx) in self.args.keys():
                new_indices.append(self.args[str(idx)])
            else:
                new_indices.append(int(idx))

        # apply new indices to out_var
        ##############################

        if len(self.indices) > 0:

            if len(self.indices) == 1:
                if new_indices[0] == ':':
                    indexed_var = out_var[:]
                else:
                    indexed_var = out_var[new_indices[0]]
            else:
                if new_indices[0] == ':':
                    indexed_var = out_var[:, new_indices[1]]
                elif new_indices[1] == ':':
                    indexed_var = out_var[new_indices[0], :]
                else:
                    indexed_var = out_var[new_indices[0], new_indices[1]]

        else:

            indexed_var = out_var

        return indexed_var


class EquationParser(object):

    def __init__(self, eq: str):

        lhs, rhs = eq.split('=')

        eq_parts = [lhs, rhs]
        self.operations = []
        self.variables = []

        for eq_part in eq_parts:

            parser = RHSParser(eq_part)
            self.get_operations(parser.expression)
            self.get_variables(parser.expression)

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

                if isinstance(var, bool):

                    tf_var = var

                elif isinstance(var, float) or isinstance(var, int):

                    tf_var = tf.constant(var)

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

                    tf_var = var

                tf_vars.append(tf_var)
                var_names.append(var_name)

    return tf_vars, var_names

