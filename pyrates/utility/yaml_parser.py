""" Some utility functions for parsing YAML-based definitions of circuits and components.
"""
from typing import Union

__author__ = "Daniel Rose"
__status__ = "Development"

import importlib
from pyrates.operator import OperatorTemplate


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
            if "base" in template_dict:
                base_path = template_dict.pop("base")
                if base_path == "OperatorTemplate":
                    template = OperatorTemplate(**template_dict)
                    # also basic assumption, if none is given
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

            else:
                # create template (no base needed)
                template = OperatorTemplate(**template_dict)

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

    @classmethod
    def update_template(cls, base, name: str = None, path: str = None, equations: Union[str, dict] = None,
                        variables: dict = None, description: str = None, conditions: dict = None):
        """Update all entries of the Operator template in their respective ways."""

        if equations:
            # if it is a string, just replace
            if isinstance(equations, str):
                pass  # pass equations string to constructor
            # else, update according to predefined rules, assuming dict structure
            else:
                equations = cls.update_equation(base.equations, **equations)
        else:
            # copy equations from parent template
            equations = base.equations

        if variables:
            variables = cls.update_variables(base.variables, variables)
        else:
            variables = base.variables

        rogue_variables = []
        for var in variables:
            # remove variables that are not present in the equation anymore
            if not var in equations:
                # save entries in list, since dictionary must not change size during iteration
                rogue_variables.append(var)

        for var in rogue_variables:
            variables.pop(var)

        if conditions:
            # copy old conditions dict
            conditions = cls.update_conditions(base.conditions, conditions)
        else:
            conditions = base.conditions

        if not description:
            description = base.__doc__  # or do we want to enforce documenting a template?

        return OperatorTemplate(name=name, path=path, equations=equations, variables=variables,
                                description=description, conditions=conditions)

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

    @staticmethod
    def update_conditions(conditions: Union[dict, None], updates: dict):

        if conditions:
            updated = conditions.copy()
        else:
            updated = {}

        for cond, cond_dict in updates.items():
            if cond in updated:
                # update dictionary defining single condition
                updated[cond].update(cond_dict)
            else:
                # copy new condition into conditions dictionary
                updated.update({cond: cond_dict})

        return updated
