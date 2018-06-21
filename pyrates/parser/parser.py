"""This module contains different parser classes used to parse strings/dictionaries passed by the user and turn them
into formats that can be handled by tensorflow.
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

    Attributes
    ----------

    Methods
    -------

    Examples
    --------

    References
    ----------

    """

    def __init__(self, expression: str, args: dict, tf_graph: tf.Graph):
        """Instantiates RHSParser.
        """

        # initialization of instance fields
        ###################################

        self.args = args
        self.tf_graph = tf_graph
        self.tf_op = tf.no_op()
        self.tf_op_args = []
        self.lambdify_args = []

        # parse expression
        ##################

        parsed_expr = self.parse_expr(expression)
        self.expression = self.check_for_matrix_ops(parsed_expr)

    def parse_expr(self, expr: str) -> Expr:
        """Parses expression and checks whether certain matrix operations have been used that cannot be parsed by sympy.

        Parameters
        ----------
        expr
            Expression string (should represent right-hand side of an equation).

        Returns
        -------
        Expr
            Sympy-parsed expression

        """

        # check for matrix-based expressions
        #####################################

        # breaking operator definition
        breaking_ops = {'@': 'dot(A, B)',
                        '.T': 'transp(A)',
                        '.I': 'inv(A)',
                        '[': 'idx(A, row, col)'
                        }

        # replacable operator definition
        replace_ops = {':': 'all'}

        # check for breaking operations in expression string
        for op, subs in breaking_ops.items():
            if op in expr:
                raise ValueError(f"Operator sign {op} cannot be parsed. Please use {subs} instead.")

        # replace operations in expression string
        for op, subs in replace_ops.items():
            expr = expr.replace(op, subs)

        return parse_expr(expr)

    def check_for_matrix_ops(self, expr: Expr) -> Expr:
        """Goes through parsed expression and makes it compatible with sympy matrix operations.

        Parameters
        ----------
        expr
            Sympy-parsed expression.

        Returns
        -------
        Expr
            New, matrix operation compatible expression

        """

        replace_ops = {'is_Mul': {'func': 'custom_mult()',
                                  'arg': tf.multiply
                                  },
                       'is_Pow': {'func': 'custom_power()',
                                  'arg': tf.pow
                                  },
                       'is_Add': {'func': 'custom_add()',
                                  'arg': tf.add
                                  },
                       'idx': {'func': self.apply_idx},
                       'transp': {'arg': tf.transpose
                                  },
                       'inv': {'arg': tf.matrix_inverse
                               },
                       'dot': {'arg': tf.matmul}
                       }

        return self.replace(expr, replace_ops)

    def apply_idx(self, args: list) -> str:
        """Applies idx() function to arguments.

        Parameters
        ----------
        args
            List of arguments passed to the idx() function.

        Returns
        -------
        str
            Brackets indexing notation ( A[row, col]).

        """

        # turn `all` slice indicator into `:`
        for i in range(len(args)):
            if str(args[i]) == 'all':
                args[i] = ':'

        idx_str = f"{self.symbol_to_str(args[0])}["
        for i in range(1, len(args)):
            idx_str += f"{self.symbol_to_str(args[i])}, "

        return f"{idx_str[:-2]}]"

    def replace(self, expr: Expr, replacement_dict: dict) -> Expr:
        """Recursively goes through expr and replaces matches to the keys in dict with the respective values.

        Parameters
        ----------
        expr
            Sympy-parsed expression.
        replacement_dict
            Key, value pairs. Keys will be checked for in expr, values will be used to replace key matches in expr.

        Returns
        -------
        Expr
            New sympy expression with potential replacements.

        """

        # extract top level function and arguments to that function from expression
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

                    # replace target sympy operations with custom functions
                    #######################################################

                    if new_func['func'].find('(') != -1:

                        # string replacements are expected to be of form: `custom_func()`
                        base_str = new_func['func'][0:-1]
                        func_str = ''

                        if len(new_args) > 2:

                            # iterate through arguments to func and create string of form
                            # `custom_func(arg1, custom_func(arg2, arg3))`
                            for j, arg in enumerate(new_args):
                                if j < len(new_args) - 1:
                                    func_str += f"{base_str} {self.symbol_to_str(arg)}, "
                                else:
                                    func_str += f"{self.symbol_to_str(arg)})"
                            func_str = func_str[0:-2]
                            for _ in range(j + 1):
                                func_str += ')'

                        else:

                            # create string of form `custom_func(arg1, arg2)`
                            func_str = f"{base_str} {self.symbol_to_str(new_args[0])}, " \
                                       f"{self.symbol_to_str(new_args[1])})"

                        # save arguments to custom function on args dictionary with the function name as key
                        self.args[new_func['func'][0:-2]] = new_func['arg']

                    else:

                        # create function string of form `arg1 <new_operator> arg2`
                        func_str = f"{self.symbol_to_str(new_args[0])} " \
                                   f"{new_func['func']} " \
                                   f"{self.symbol_to_str(new_args[1])}"

                    # parse function string with sympy
                    new_expr = self.parse_expr(func_str)

                    break

                elif key == str(func):

                    # replace target functions (identified by their name) with other functions or just add arguments
                    ################################################################################################

                    # if function was passed, use it to get a new function string and then parse that string
                    if 'func' in new_func.keys():
                        func_str = new_func['func'](new_args)
                        new_expr = parse_expr(func_str)
                    else:
                        new_expr = expr

                    # if function arguments where passed, add them to the args dictionary
                    if 'arg' in new_func.keys():
                        self.args[key] = new_func['arg']

                    break

                elif i < len(replacement_dict) - 1:

                    new_expr = expr

                else:

                    # if in the last iteration and no new expression has been created, apply new_args (collected through
                    # recursion) to the top-level function
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
            Symbol or MatrixSymbol

        Returns
        -------
        str
            String-based representation of symb.

        """

        if isinstance(symb, MatrixSymbol):
            symb_str = f"MatrixSymbol('{symb}', {symb.shape[0]}, {symb.shape[1]})"
        else:
            symb_str = str(symb)

        return symb_str

    def transform(self) -> Union[tf.Operation, tf.Tensor]:
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

                # get a new sympy symbol for the result
                new_exp = symbols('var_' + str(i))

                # substitute the function call with the symbol representing its result
                self.expression = self.expression.subs(func, new_exp)

                # add the tensorflow operation calculating the result to the args dictionary
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
        """Recursive function that turns all custom functions in the expression into tensorflow operations.

        Parameters
        ----------
        expression
            Sympy-parsed expression.

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

        if isinstance(expression.func, UndefinedFunction):

            # parsing of custom functions
            #############################

            if func_name not in self.args.keys():
                raise ValueError(func_name + ' must be defined in args!')

            # extract custom function definition
            new_expr = self.args[func_name]

            if callable(new_expr):

                # parsing of tensorflow functions
                #################################

                func_args = []

                for arg in expression.args:

                    if str(arg) in self.args:

                        # extract argument from args
                        new_arg = self.args[str(arg)]
                        func_args.append(new_arg)

                    elif arg.is_Number:

                        # turn constant into tensorflow constant
                        # TODO: avoid this. Datatype should be infered somehow
                        func_args.append(tf.constant(float(arg), dtype=tf.float32))

                    else:

                        # create new instance of RHSParser and create tensorflow operation from arg
                        parser = RHSParser(str(arg), self.args, self.tf_graph)
                        tf_op_tmp = parser.transform()
                        func_args.append(tf_op_tmp)

                # create tensorflow operation from function
                tf_op = new_expr(*tuple(func_args))

                # collect tensorflow operation and original expression
                tf_ops.append(tf_op)
                funcs.append(expression)

            elif type(new_expr) is 'str':

                # parsing of string-based functions
                ###################################

                # create new instance of RHSParser and create tensorflow operation from string
                parser = RHSParser(self.args[func_name], self.args, self.tf_graph)
                tf_op = parser.transform()

                # collect tensorflow operation and original expression
                tf_ops.append(tf_op)
                funcs.append(expression)

            else:

                raise ValueError('Custom functions can only be of type `tensorflow function` or `str`.')

        # if multiple arguments exist that are connected by func, call this function recursively on each argument
        #########################################################################################################

        if len(expression.args) > 0 and expression.func != MatrixElement and expression.func != MatrixSlice \
                and not isinstance(expression, MatrixSymbol):

            # go through all arguments of top-level function
            for expr in expression.args:

                # recursive call
                tf_ops_tmp, funcs_tmp = self.custom_funcs_to_tf_ops(expr)

                # add results to collector variables
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

    Attributes
    ----------

    Methods
    -------

    Examples
    --------

    References
    ----------

    """

    def __init__(self, expression: str, args: dict, rhs: Union[tf.Operation, tf.Tensor], tf_graph: tf.Graph):
        """Instantiates LHSParser.
        """

        # initialize variables
        ######################

        self.args = args
        self.tf_graph = tf_graph
        self.rhs = rhs
        self.indices = []

        # parse expression
        ##################

        parser = RHSParser(expression, args, self.tf_graph)
        self.expression = parser.expression

    def transform(self) -> Tuple[Union[tf.Operation, tf.Tensor], Union[tf.Variable, tf.Tensor]]:
        """Turns the expression into a runnable tensorflow operation.

        Parameters
        ----------

        Returns
        -------
        Tuple[Union[tf.Operation, tf.Tensor], Union[tf.Variable, tf.Tensor]]
            Either a tensorflow operation (if its a differential equation) or a tensorflow variable
            (the state variable to be updated by a right-hand side of an equation).

        """

        # get target variable from expression
        target_var = self.get_target_var()

        if symbols('dt') in self.expression.free_symbols:

            # solve differential equation
            #############################

            from pyrates.solver import Solver

            if 'dt' not in self.args.keys():
                raise AttributeError('The step-size `dt` has to be passed with `args` for differential equations.')

            solver = Solver(self.rhs, target_var, self.args['dt'], self.tf_graph)
            tf_op = solver.solve()

        else:

            # calculate value of target variable
            ####################################

            with self.tf_graph.as_default():
                tf_op = target_var.assign(self.rhs)

        return tf_op, target_var

    def get_target_var(self) -> Union[tf.Variable, tf.Tensor, tf.Operation]:
        """Finds target variable in expression and extracts it from args.

        Parameters
        ----------

        Returns
        -------
        Union[tf.Variable, tf.Tensor, tf.Operation]
            Target tensorflow variable from args.

        """

        expr = self.expression

        if len(expr.free_symbols) == 1:

            target_var = self.apply_slicing(expr)

        else:

            target_var = None

            # go through the left-hand side arguments
            for arg in expr.args:

                # find target variable in arguments
                if not (symbols('d') in arg.free_symbols or symbols('dt') in arg.free_symbols):

                    target_var = self.apply_slicing(arg)

                    break

        if target_var is None:
            raise ValueError('Target variable has to be included in left-hand side of expression!')

        return target_var

    def apply_slicing(self, arg: Expr) -> Union[tf.Variable, tf.Tensor, tf.Operation]:
        """Apply slicing to arg if necessary.

        Parameters
        ----------
        arg
            Sympy expression.

        Returns
        -------
        Union[tf.Variable, tf.Tensor, tf.Operation]

        """

        if arg.func == MatrixElement:

            # extract element from target variable matrix
            #############################################

            # get and check variable name
            var_name = str(arg.args[0])
            if var_name not in self.args.keys():
                raise AttributeError("Output variables or their indices must be included in "
                                     "expression_args.")

            # extract target variable from args
            if len(arg.args) == 3:
                target_var = self.args[var_name][int(arg.args[1]), int(arg.args[2])]
            else:
                target_var = self.args[var_name][int(arg.args[1])]

        elif arg.func == MatrixSlice:

            # extract slice from target variable matrix
            ###########################################

            # get and check variable name
            var_name = str(arg.args[0])
            if var_name not in self.args.keys():
                raise AttributeError("Output variables or their indices must be included in "
                                     "expression_args.")

            # extract target variable from args
            if len(arg.args) == 3:
                target_var = self.args[var_name][int(arg.args[1][0]):int(arg.args[1][1]),
                             int(arg.args[2][0]):int(arg.args[2][1])]
            else:
                target_var = self.args[var_name][int(arg.args[1][0]):int(arg.args[1][1])]

        else:

            # get and check variable name
            var_name = str(arg)
            if var_name not in self.args.keys():
                raise AttributeError("Output variables or their indices must be included in "
                                     "expression_args.")

            # extract target variable from args
            target_var = self.args[var_name]

        return target_var


def parse_dict(var_dict: dict, var_scope: str, tf_graph: Optional[tf.Graph] = None) -> Tuple[list, list]:
    """Parses a dictionary with variable information and creates tensorflow variables from that information.

    Parameters
    ----------
    var_dict
    var_scope
    tf_graph

    Returns
    -------
    Tuple

    """

    # get tensorflow graph
    tf_graph = tf_graph if tf_graph else tf.get_default_graph()

    with tf_graph.as_default():

        with tf.variable_scope(var_scope):

            # data-type definition
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

            # go through dictionary items and instantiate variables
            #######################################################

            for var_name, var in var_dict.items():

                if var['variable_type'] == 'raw':

                    tf_var = var['variable']

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

                    raise ValueError('Variable type must be `raw`, `state_variable`, `constant` or `placeholder`.')

                tf_vars.append(tf_var)
                var_names.append(var_name)

    return tf_vars, var_names

