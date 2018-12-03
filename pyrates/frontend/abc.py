""" Abstract base classes
"""
from importlib import import_module

# from pyrates.ir.abc import AbstractBaseIR

__author__ = "Daniel Rose"
__status__ = "Development"


class AbstractBaseTemplate:
    """Abstract base class for templates"""

    target_ir = None  # placeholder for template-specific intermediate representation (IR) target class

    def __init__(self, name: str, path: str, description: str = "A template."):
        self.name = name
        self.path = path
        self.__doc__ = description  # overwrite class-specific doc with user-defined description

    def __repr__(self):
        return f"<{self.__class__.__name__} '{self.path}'>"

    def _format_path(self, path):
        """Check if path contains a folder structure and prepend own path, if it doesn't"""
        # ToDo: rename to something more meaningful like _prepend_parent_path or _check_path_prepend_parent

        if "." not in path:
            path = ".".join((*self.path.split('.')[:-1], path))
        return path

    @classmethod
    def from_yaml(cls, path):
        """Convenience method that looks for a loader class for the template type and applies it, assuming
        the class naming convention '<template class>Loader'.

        Parameters:
        -----------
        path
            Path to template in YAML file of form 'directories.file.template'
        """
        # ToDo: add AbstractBaseTemplate._format_path functionality here
        module = import_module(cls.__module__)
        loader = getattr(module, f"{cls.__name__}Loader")
        return loader(path)

    def apply(self, *args, **kwargs):

        raise NotImplementedError


