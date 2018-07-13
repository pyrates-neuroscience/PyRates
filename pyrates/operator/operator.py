"""This module contains the operator class used to create executable operations from expression strings.
"""

# external imports
import re
from typing import List, Optional, Union
import tensorflow as tf

# pyrates internal imports
from pyrates.parser import EquationParser

from pyrates.abc import AbstractBaseTemplate
from pyrates.utility.yaml_parser import TemplateLoader

# meta infos
__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


class Operator(object):
    """Basic operator class that turns a list of expression strings into tensorflow operations using the variables
    provided in the expression args dictionary.

    Parameters
    ----------

    Attributes
    ----------

    Methods
    -------

    References
    ----------

    Examples
    --------

    """
    def __init__(self, expressions: List[str], expression_args: dict, key: str,
                 variable_scope: str, dependencies: Optional[list] = None, tf_graph: Optional[tf.Graph] = None):
        """Instantiates operator.
        """

        self.DEs = []
        self.updates = []
        self.key = key
        self.tf_graph = tf_graph if tf_graph else tf.get_default_graph()
        self.args = expression_args

        if dependencies is None:
            dependencies = []

        # parse expressions
        ###################

        with self.tf_graph.as_default():

            with tf.variable_scope(variable_scope):

                for i, expr in enumerate(expressions):

                    with tf.control_dependencies(dependencies):

                        # parse equation
                        parser = EquationParser(expr, self.args, engine='tensorflow', tf_graph=self.tf_graph)
                        self.args = parser.args

                        # collect tensorflow variables and update operations
                        if hasattr(parser, 'update'):
                            self.DEs.append((parser.target_var, parser.update))
                        else:
                            self.updates.append(parser.target_var)

    def create(self):
        """Create a single tensorflow operation for the set of parsed expressions.
        """

        with self.tf_graph.as_default():

            # group the tensorflow operations across expressions
            with tf.control_dependencies(self.updates):
                if len(self.DEs) > 0:
                    updates = tf.group([var.assign(upd) for var, upd in self.DEs], name=self.key)
                else:
                    updates = tf.group(self.updates, name=self.key)

        return updates


class OperatorTemplate(AbstractBaseTemplate):
    """Generic template for an operator with a name, equation(s), variables and possible
    initialization conditions. The template can be used to create variations of a specific
    equation or variables."""

    cache = {}  # tracks all unique instances of applied operator templates

    def __init__(self, name: str, path: str, equation: str, variables: dict, description: str,
                 options: dict = None):
        """For now: only allow single equation in operator template."""

        super().__init__(name, path, description)

        self.equation = equation
        self.variables = variables
        self.options = options
        # if options:
        #     raise NotImplementedError

    def apply(self, options: dict = None, return_key=False):
        """Returns the non-editable but unique, cashed definition of the operator."""

        if not options:
            key = (self.path, None)
        else:
            key = (self.path, frozenset(options.items()))

        try:
            instance = self.cache[key]
            _, values = self._separate_variables()
        except KeyError:
            if options:
                raise NotImplementedError("Applying options to a template is not implemented yet.")
            variables, values = self._separate_variables()
            instance = dict(equation=self.equation, variables=variables)
            self.cache[key] = instance

        if return_key:
            return instance, values, key
        else:
            return instance, values
        # TODO: return operator instance

    def _separate_variables(self):
        """Return variable definitions and the respective values."""
        # this part can be improved a lot with a proper expression parser

        variables = {}
        values = {}
        for variable, properties in self.variables.items():
            var_dict = {}
            for prop, expr in properties.items():
                if prop == "default":
                    var_dict["variable_type"], var_dict["data_type"], value = self._parse_vprops(expr)
                    if value:
                        values[variable] = value
                    # else: don't pass information for that variable
                else:
                    var_dict[prop] = expr
            variables[variable] = var_dict

        return variables, values

    @staticmethod
    def _parse_vprops(expr: Union[str, int, float]):
        """Naive version of a parser for the default key of variables in a template. Returns data type,
        variable type and default value of the variable."""

        value = None
        if isinstance(expr, int):
            vtype = "constant"
            value = expr
            dtype = "int32"
        elif isinstance(expr, float):
            vtype = "constant"
            value = expr
            dtype = "float32"
        else:
            if expr.startswith(("input", "output", "variable")):
                vtype = "state_variable"
            elif expr.startswith("constant"):
                vtype = "constant"
            elif expr.startswith("placeholder"):
                vtype = "placeholder"
            else:
                try:
                    if "." in expr:
                        value = float(expr)
                    else:
                        value = int(expr)
                    vtype = "constant"
                except ValueError:
                    raise ValueError(f"Unable to interpret variable type in default definition {expr}.")

            if expr.endswith("(float)"):
                dtype = "float32" # why float32 and not float64?
            elif expr.endswith("(int)"):
                dtype = "int32"
            elif "." in expr:
                dtype = "float32"
                value = re.search("[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)", expr).group()
                # see https://stackoverflow.com/questions/12643009/regular-expression-for-floating-point-numbers
            elif re.search("[0-9]+", expr):
                dtype = "int32"
                value = re.search("[0-9]+", expr).group()
            else:
                dtype = "float32" # base assumption

        return vtype, dtype, value


class OperatorTemplateLoader(TemplateLoader):
    """Template loader specific to an OperatorTemplate. """

    def __new__(cls, path):

        return super().__new__(cls, path, OperatorTemplate)

    @classmethod
    def update_template(cls, base, name: str, path: str, equation: Union[str, dict] = None,
                        variables: dict = None, description: str = None, options: dict = None):
        """Update all entries of the Operator template in their respective ways."""

        if equation:
            # if it is a string, just replace
            if isinstance(equation, str):
                pass  # pass equation string to constructor
            # else, update according to predefined rules, assuming dict structure
            else:
                equation = cls.update_equation(base.equation, **equation)
        else:
            # copy equation from parent template
            equation = base.equation

        if variables:
            variables = cls.update_variables(base.variables, variables)
        else:
            variables = base.variables

        rogue_variables = []
        for var in variables:
            # remove variables that are not present in the equation anymore
            if var not in equation:
                # save entries in list, since dictionary must not change size during iteration
                rogue_variables.append(var)

        for var in rogue_variables:
            variables.pop(var)

        if options:
            options = cls.update_options(base.options, options)
        else:
            # copy old options dict
            options = base.options

        if not description:
            description = base.__doc__  # or do we want to enforce documenting a template?

        return OperatorTemplate(name=name, path=path, equation=equation, variables=variables,
                                description=description, options=options)

    @staticmethod
    def update_equation(equation: str,  # original equation
                        replace: dict = False,  # replace parts of the string
                        remove: Union[list, tuple] = False,  # remove parts of the string
                        append: str = False,  # append to the end of the string
                        prepend: str = False,  # add to beginning of string
                        ):

        # replace existing terms by new ones
        if replace:
            for old, new in replace.items():
                equation = equation.replace(old, new)
                # this might fail, if multiple replacements refer or contain the same variables
                # is it possible to call str.replace with tuples?

        # remove terms
        if remove:
            for old in remove:
                equation = equation.replace(old, "")

        # append terms at the end of the equation string
        if append:
            # only allowing single append per update
            equation = f"{equation} {append}"

            # prepend terms at the beginning of the equation string
        if prepend:
            # only allowing single prepend per update
            equation = f"{prepend} {equation}"

        return equation

    @staticmethod
    def update_variables(variables: dict, updates: dict):

        updated = variables.copy()

        for var, var_dict in updates.items():
            if var in updated:
                # update dictionary defining single variable
                updated[var].update(var_dict)
            else:
                # copy new variable into variables dictionary
                updated.update({var: var_dict})

        return updated