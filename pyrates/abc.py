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
        return f"{self.__class__.__name__} <{self.path}>"

    def _format_path(self, path):
        """Check if path contains a folder structure and prepend own path, if it doesn't"""

        if "." not in path:
            path = ".".join((*self.path.split('.')[:-1], path))
        return path
