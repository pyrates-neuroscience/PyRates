# external imports
from copy import deepcopy
from typing import Union

# pyrates internal imports
from pyrates.frontend.abc import AbstractBaseTemplate
from pyrates.frontend.parser.yaml import TemplateLoader

# meta infos
from pyrates.ir.operator import OperatorIR

__author__ = " Daniel Rose"
__status__ = "Development"


class OperatorTemplate(AbstractBaseTemplate):
    """Generic template for an operator with a name, equations, variables and possible
    initialization conditions. The template can be used to create variations of a specific
    equations or variables."""

    cache = {}  # tracks all unique instances of applied operator templates
    target_ir = OperatorIR

    # key_map = {}  # tracks renaming of of operator keys to have shorter unique keys

    def __init__(self, name: str, path: str, equations: Union[list, str], variables: dict,
                 description: str = "An operator template."):
        """For now: only allow single equations in operator template."""

        super().__init__(name, path, description)

        if isinstance(equations, str):
            self.equations = [equations]
        else:
            self.equations = equations
        self.variables = variables

    def apply(self, return_key=False, values: dict = None):
        """Returns the non-editable but unique, cashed definition of the operator."""

        return super().apply(return_key=return_key, values=values)


class OperatorTemplateLoader(TemplateLoader):
    """Template loader specific to an OperatorTemplate. """

    def __new__(cls, path):

        return super().__new__(cls, path, OperatorTemplate)

    @classmethod
    def update_template(cls, base, name: str, path: str, equations: Union[str, list, dict] = None,
                        variables: dict = None, description: str = None):
        """Update all entries of the Operator template in their respective ways."""

        if equations:
            # if it is a string, just replace
            if isinstance(equations, str):
                equations = [equations]
            elif isinstance(equations, list):
                pass  # pass equations string to constructor
            # else, update according to predefined rules, assuming dict structure
            elif isinstance(equations, dict):
                equations = [cls.update_equation(eq, **equations) for eq in base.equations]

            else:
                raise TypeError("Unknown data type for attribute 'equations'.")
        else:
            # copy equations from parent template
            equations = base.equations

        if variables:
            variables = cls.update_variables(base.variables, variables)
        else:
            variables = base.variables

        rogue_variables = set()
        for var in variables:
            # remove variables that are not present in the equations
            found = False
            for eq in equations:
                if var in eq:
                    found = True

            if not found:
                # save entries in list, since dictionary must not change size during iteration
                rogue_variables.add(var)

        for var in rogue_variables:
            variables.pop(var)

        if not description:
            description = base.__doc__  # or do we want to enforce documenting a template?

        return OperatorTemplate(name=name, path=path, equations=equations, variables=variables,
                                description=description)

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
            if isinstance(remove, str):
                equation = equation.replace(remove, "")
            else:
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

        updated = deepcopy(variables)

        for var, var_dict in updates.items():
            if var in updated:
                # update dictionary defining single variable
                updated[var].update(var_dict)
            else:
                # copy new variable into variables dictionary
                updated.update({var: var_dict})

        return updated
