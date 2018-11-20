""" Some utility functions for parsing YAML-based definitions of circuits and components.
"""

from pyrates import PyRatesException

__author__ = "Daniel Rose"
__status__ = "Development"

import importlib


class TemplateLoader:
    """Class that loads templates from YAML and returns an OperatorTemplate class instance"""

    cache = {}  # dictionary that keeps track of already loaded templates

    def __new__(cls, path: str, template_cls: type):
        """Load template recursively and return OperatorTemplate class.

        Parameters
        ----------

        path
            string containing path of YAML template of the form path.to.template
        template_cls
            class that the loaded template will be instantiated with
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
            if base_path == template_cls.__name__:
                # if base refers to the python representation, instantiate here
                template = template_cls(**template_dict)
            else:
                # load base if needed
                if "." in base_path:
                    # reference to template in different file
                    # noinspection PyCallingNonCallable
                    template = cls(base_path)
                else:
                    # reference to template in same file
                    base_path = ".".join((*path.split(".")[:-1], base_path))
                    # noinspection PyCallingNonCallable
                    template = cls(base_path)
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
            string containing path of YAML template of the form path.to.template or path/to/template.file.TemplateName.
            The dot notation refers to a path that can be found using python's import functionality. The slash notation
            refers to a file in an absolute or relative path from the current working directory.
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
        # ToDo: Add parent path as required argument, to better locate errors in YAML templates.

        if "/" in path:
            *dirs, file = path.split("/")
            dirs = "/".join(dirs)
            *file, name = file.split(".")
            file = ".".join(file)
            import os
            abspath = os.path.abspath(dirs)
        else:
            if "." in path:
                parts = path.split(".")
                name = parts[-1]

                if parts[0] == "pyrates":
                    # look for pyrates library and return absolute path
                    file = parts[-2]
                    parentdir = ".".join(parts[:-2])
                    # let Python figure out where to look for the module
                    try:
                        module = importlib.import_module(parentdir)
                    except ModuleNotFoundError:
                        raise PyRatesException(f"Could not find Python (module) directory associated to path "
                                               f"`{parentdir}` of Template `{path}`.")
                    try:
                        abspath = module.__path__[0]  # __path__ returns a list[str]
                    except TypeError:
                        raise PyRatesException(f"Something is wrong with the given YAML template path `{path}`.")

                else:
                    # import custom defined model with relative or absolute path
                    import os
                    file = os.path.join(*parts[:-1])
                    abspath = ""  # empty filepath

            else:
                raise NotImplementedError(f"Was base specified in template '{path}', but left empty?")
                # this should only happen, if "base" is specified, but empty

        return name, file, abspath


