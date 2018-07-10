"""This module contains the operator class used to create executable operations from expression strings.
"""

# external imports
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

    instance_cache = {}

    def __init__(self, name: str, path: str, equation: str, variables: dict, description: str,
                 options: dict = None):
        """For now: only allow single equation in operator template."""

        super().__init__(name, path, description)

        self.equation = equation
        self.variables = variables

        self.options = options
        # if options:
        #     raise NotImplementedError

    def apply(self, options: dict =None):

        if options:
            hashable = (self.path, tuple((key, value) for key, value in options.items()))
        else:
            hashable = (self.path, ())

        if hashable in self.instance_cache:
            return self.instance_cache[hashable]
        else:
            return OperatorInstance(self, options)
        # TODO: return operator instance


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