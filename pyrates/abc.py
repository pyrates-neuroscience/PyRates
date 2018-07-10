""" Abstract base classes
"""

__author__ = "Daniel Rose"
__status__ = "Development"


class AbstractBaseTemplate:
    """Abstract base class for templates"""

    def __init__(self, name: str, path: str, description: str):
        self.name = name
        self.path = path
        self.__doc__ = description

    def __repr__(self):
        return f"<{self.__class__.__name__} '{self.path}'>"

    def _format_path(self, path):
        """Check if path contains a folder structure and prepend own path, if it doesn't"""

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

        from importlib import import_module
        module = import_module(cls.__module__)
        loader = getattr(module, f"{cls.__name__}Loader")
        return loader(path)
