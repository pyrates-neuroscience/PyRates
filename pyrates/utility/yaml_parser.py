""" Some utility functions for parsing YAML-based definitions of circuits and components.
"""
from typing import Union, List

__author__ = "Daniel Rose"
__status__ = "Development"

import importlib
from pyrates.operator import OperatorTemplate
from pyrates.node import NodeTemplate


class TemplateLoader:
    """Class that loads templates from YAML and returns an OperatorTemplate class instance"""

    cache = {}  # dictionary that keeps track of already loaded templates

    def __new__(cls, path: str):
        """Load template recursively and return OperatorTemplate class.

        Parameters
        ----------

        path
            string containing path of YAML template of the form path.to.template
        """

        if path in cls.cache:
            template = cls.cache[path]
        else:
            template_dict = cls.load_template_from_yaml(path)
            try:
                base_path = template_dict.pop("base")
            except KeyError:
                raise KeyError(f"No 'base' defined for template {path}. Please define a "
                               f"base to derive the template from.")
            if base_path == "OperatorTemplate":
                template = OperatorTemplate(**template_dict)
            elif base_path == "NodeTemplate":
                template = NodeTemplate(**template_dict)

            else:
                # load base if needed
                if "." in base_path:
                    # reference to template in different file
                    template = TemplateLoader(base_path)
                else:
                    # reference to template in same file
                    base_path = ".".join((*path.split(".")[:-1], base_path))
                    template = TemplateLoader(base_path)
                template = cls.update_template(template, **template_dict)
                # may fail if "base" is present but empty

            cls.cache[path] = template

        return template

    @classmethod
    def load_template_from_yaml(cls, path: str):
        """As name says: Load a template from YAML and return the resulting dictionary.

        Parameters
        ----------

        path
            string containing path of YAML template of the form path.to.template
        """
        name, filename, directory = cls.parse_path(path)
        from ruamel.yaml import YAML
        import os

        yaml = YAML(typ="safe", pure=True)

        if not filename.endswith(".yaml"):
            filename = f"{filename}.yaml"

        filepath = os.path.join(directory, filename)

        with open(filepath, "r") as file:
            file_dict = yaml.load(file)

        if name in file_dict:
            template_dict = file_dict[name]
            template_dict["path"] = path
            template_dict["name"] = name
        else:
            raise AttributeError(f"Could not find {name} in {filepath}.")

        return template_dict

    @classmethod
    def update_template(cls, *args, **kwargs):
        """Updates the template with a given list of arguments."""
        raise NotImplementedError

    @staticmethod
    def parse_path(path: str):
        """Parse a path of form path.to.template, returning a tuple of (name, file, abspath)."""

        if "." in path:
            parts = path.split(".")
            name = parts[-1]

            if parts[0] == "pyrates":
                # look for pyrates library and return absolute path
                file = parts[-2]
                parentdir = ".".join(parts[:-2])
                # let Python figure out where to look for the module
                module = importlib.import_module(parentdir)

                abspath = module.__path__[0]  # __path__ returns a list[str]

                return name, file, abspath

            else:
                # import custom defined model with relative or absolute path
                import os
                file = os.path.join(*parts[:-1])
                return name, file, ""  # empty filepath

        else:
            raise NotImplementedError
            # this should only happen, if "base" is specified, but empty

    @staticmethod
    def update_options(options: Union[dict, None], updates: dict):

        if options:
            updated = options.copy()
        else:
            updated = {}

        for opt, opt_dict in updates.items():
            if opt in updated:
                # update dictionary defining single condition
                updated[opt].update(opt_dict)
            else:
                # copy new condition into options dictionary
                updated.update({opt: opt_dict})

        return updated


class OperatorTemplateLoader(TemplateLoader):
    """Template loader specific to an OperatorTemplate. """

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
            # copy old options dict
            options = cls.update_options(base.options, options)
        else:
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


class NodeTemplateLoader(TemplateLoader):
    """Template loader specific to an OperatorTemplate. """

    @classmethod
    def update_template(cls, base, name: str, path: str, label: str,
                        operators: Union[str, List[str], dict] = None,
                        description: str = None,
                        options: dict = None):
        """Update all entries of a base node template to a more specific template."""

        if operators:
            cls.update_operators(base.operators, operators)
        else:
            operators = base.operators

        if options:
            # copy old options dict
            options = cls.update_options(base.options, options)
        else:
            options = base.options

        if not description:
            description = base.__doc__  # or do we want to enforce documenting a template?

        return NodeTemplate(name=name, path=path, label=label, operators=operators,
                            description=description, options=options)

    @staticmethod
    def update_operators(base_operators: dict, updates: Union[str, List[str], dict]):
        """Update operators of a given template. Note that currently, only the new information is
        propagated into the operators dictionary. Comparing or replacing operators does not work currently.

        Parameters:
        -----------

        base_operators:
            Reference to one or more operators in the base class.
        updates:
            Reference to one ore more operators in the child class
            - string refers to path or name of single operator
            - list refers to multiple operators of the same class
            - dict contains operator path or name as key and options/defaults as sub-dictionaries
        """
        # updated = base_operators.copy()
        updated = {}
        if isinstance(updates, str):
            updated[updates] = {}  # single operator path with no variations
        elif isinstance(updates, list):
            for path in updates:
                updated[path] = {}  # multiple operator paths with no variations
        elif isinstance(updates, dict):
            for path, variations in updates.items():
                updated[path] = variations
            # dictionary with operator path as key and variations as sub-dictionary
        else:
            raise TypeError("Unable to interpret type of operator updates. Must be a single string,"
                            "list of strings or dictionary.")
        # # Check somewhere, if child operators have same input/output as base operators?
        #
        return updated





