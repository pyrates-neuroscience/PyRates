# external imports
import re
from collections import Sequence
from copy import deepcopy
from typing import Union, List

# pyrates internal imports
from pyrates import PyRatesException
from pyrates.frontend.abc import AbstractBaseTemplate, AbstractBaseIR
from pyrates.frontend.yaml_parser import TemplateLoader, deep_freeze

# meta infos
__author__ = " Daniel Rose"
__status__ = "Development"


class OperatorIR(AbstractBaseIR):
    cache = {}
    key_map = {}
    key_counter = {}

    def __init__(self, equations: List[str], inputs: list, output: str):

        self.output = output
        self.inputs = inputs
        self.equations = equations

    @classmethod
    def from_template(cls, template, return_key=False, values: dict = None):

        # figure out key name of given combination of template and assigned values
        name = template.name
        if values:
            # if values are given, figure out, whether combination is known
            frozen_values = deep_freeze(values)
            try:
                key = cls.key_map[(name, frozen_values)]
            except KeyError:
                # not known, need to assign new key and register key in key_map
                if name in cls.key_counter:
                    key = f"{name}.{cls.key_counter[name]+1}"
                    cls.key_counter[name] += 1
                else:
                    key = f"{name}.1"
                    cls.key_counter[name] = 1
                cls.key_map[(name, frozen_values)] = key

        else:  # without additional values specified, default name is op_name.0
            key = f"{name}.0"

        try:
            instance, variables = cls.cache[key]
            # instance = defrost(instance)
            # variables = defrost(variables)
            # instance, variables = dict(instance), dict(variables)
        except KeyError:
            # get variable definitions and specified default values
            # ToDo: remove variable separation: instead pass variables detached from equations?
            variables, inputs, output = cls._separate_variables(template)

            # reduce order of ODE if necessary
            # this step is currently skipped to streamline equation interface
            # instead equations need to be given as either non-differential equations or first-order
            # linear differential equations
            # *equations, variables = cls._reduce_ode_order(template.equations, variables)
            equations = template.equations

            # operator instance is invoked as a dictionary of equations and variable definition
            # this may be subject to change

            if values:
                for vname, update in values.items():
                    # if isinstance(update, dict):
                    #     if "value" in update:
                    #         variables[vname["value"]] = update["value"]
                    #     if "vtype" in
                    # only values are updated, assuming all other specs remain the same
                    variables[vname]["value"] = values[vname]
                    # should fail, if variable is unknown

            instance = cls(equations=equations, inputs=inputs, output=output)
            cls.cache[key] = (instance, variables)

        if return_key:
            return instance.copy(), deepcopy(variables), key
        else:
            return instance.copy(), deepcopy(variables)

    @staticmethod
    def _reduce_ode_order(equations: str, variables):
        """Checks if a 2nd-order ODE is present and reduces it to two coupled first-order ODEs.
        Currently limited to special case of the form '(d/dt + a)^2 * x = b'.

        Parameters
        ----------
        equations
            string of form 'a = b'
        """

        # matches pattern of form `(d/dt + a)^2 * y` and extracts `a` and `y`
        match = re.match(r"\(\s*d\s*/\s*dt\s*[+-]\s*([\d]*[.]?[\d]*/?[a-zA-Z]\w*)\s*\)\s*\^2\s*\*\s*([a-zA-Z]\w*)",
                         equations)

        if match:
            # assume the entire lhs was matched, fails if there is something remaining on the lhs
            lhs, rhs = equations.split("=")
            a, var = match.groups()  # returns coefficient `a` and variable `y`
            eq1 = f"d/dt * {var} = {var}_t"
            eq2 = f"d/dt * {var}_t = {rhs} - ({a})^2 * {var} - 2. * {a} * {var}_t"

            variables[f"{var}_t"] = {"dtype": "float32",
                                     "description": "integration variable",
                                     "vtype": "state_var",
                                     "value": variables[var]['value'] if variables[var]['value'] else 0.}

            return eq1, eq2, variables
        else:
            return equations, variables

    @classmethod
    def _separate_variables(cls, template):
        """
        Return variable definitions and the respective values.

        Returns
        -------
        variables
        inputs
        output
        """
        # this part can be improved a lot with a proper expression parser

        variables = {}
        inputs = {}
        output = None
        for variable, properties in template.variables.items():
            var_dict = deepcopy(properties)
            # default shape is scalar
            if "shape" not in var_dict:
                var_dict["shape"] = "(1,)"

            # identify variable type and data type
            # note: this assume that a "default" must be given for every variable
            try:
                var_dict.update(cls._parse_vprops(var_dict.pop("default")))
            except KeyError:
                raise PyRatesException("Variables need to have a 'default' (variable type, data type and/or value) "
                                       "specified.")

            # separate in/out specification from variable type specification
            if var_dict["vtype"] == "input":
                inputs[variable] = dict(sources=[], reduce_dim=True)  # default to True for now
                var_dict["vtype"] = "state_var"
            elif var_dict["vtype"] == "output":
                if output is None:
                    output = variable  # for now assume maximum one output is present
                else:
                    raise PyRatesException("More than one output specification found in operator. "
                                           "Only one output per operator is supported.")
                var_dict["vtype"] = "state_var"

            variables[variable] = var_dict

        return variables, inputs, output

    @staticmethod
    def _parse_vprops(expr: Union[str, int, float]):
        """Naive version of a parser for the default key of variables in a template. Returns data type,
        variable type and default value of the variable."""

        value = 0.  # Setting initial conditions for undefined variables to 0. Is that reasonable?
        if isinstance(expr, int):
            vtype = "constant"
            value = expr
            # dtype = "int32"
            dtype = "float32"  # default to float for maximum compatibility
        elif isinstance(expr, float):
            vtype = "constant"
            value = expr
            dtype = "float32"
            # restriction to 32bit float for consistency. May not be reasonable at all times.
        else:
            if expr.startswith("input"):
                vtype = "input"
            elif expr.startswith("output"):
                vtype = "output"
            elif expr.startswith("variable"):
                vtype = "state_var"
            elif expr.startswith("constant"):
                vtype = "constant"
            elif expr.startswith("placeholder"):
                vtype = "placeholder"
            else:
                try:
                    # if "." in expr:
                    value = float(expr)  # default to float
                    # else:
                    #     value = int(expr)
                    vtype = "constant"
                except ValueError:
                    raise ValueError(f"Unable to interpret variable type in default definition {expr}.")

            if expr.endswith("(float)"):
                dtype = "float32"  # why float32 and not float64?
            elif expr.endswith("(int)"):
                # dtype = "int32"
                dtype = "float32"  # default to float for maximum compatibility
            elif "." in expr:
                dtype = "float32"
                value = float(re.search("[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)", expr).group())
                # see https://stackoverflow.com/questions/12643009/regular-expression-for-floating-point-numbers
            elif re.search("[0-9]+", expr):
                # dtype = "int32"
                dtype = "float32"  # default to float for maximum compatibility
                value = float(re.search("[0-9]+", expr).group())
            else:
                dtype = "float32"  # base assumption

        return dict(vtype=vtype, dtype=dtype, value=value)

    def copy(self):

        return self.__class__(deepcopy(self.equations), deepcopy(self.inputs), deepcopy(self.output))

    def _getter(self, key: str):
        """
        Checks if a variable named by key exists in an equations.
        Parameters
        ----------
        key

        Returns
        -------
        key
        """

        for equation in self.equations:
            if key in equation:
                return key
        else:
            raise PyRatesException(f"Variable `{key}` not found in equations {self.equations}")


class OperatorTemplate(AbstractBaseTemplate):
    """Generic template for an operator with a name, equations, variables and possible
    initialization conditions. The template can be used to create variations of a specific
    equations or variables."""

    cache = {}  # tracks all unique instances of applied operator templates

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
