""" Abstract base classes
"""
from importlib import import_module

__author__ = "Daniel Rose"
__status__ = "Development"


class AbstractBaseTemplate:
    """Abstract base class for templates"""

    def __init__(self, name: str, path: str, description: str = "Template without description"):
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
        # ToDo: add AbstractBaseTemplate._format_path functionality here
        module = import_module(cls.__module__)
        loader = getattr(module, f"{cls.__name__}Loader")
        return loader(path)

    def apply(self, *args, **kwargs):

        module = import_module(self.__class__.__module__)
        cls_name = self.__class__.__name__
        if cls_name.endswith("Template"):
            cls_name = f"{cls_name[:-8]}IR"
        else:
            raise ValueError("Class name should end with `Template`.")
        cls = getattr(module, cls_name)
        intermediate_representation = cls.from_template(self, *args, **kwargs)
        return intermediate_representation


class AbstractBaseIR:
    """Abstract base class for intermediate representation classes"""

    def __getitem__(self, key: str):
        """
        Custom implementation of __getitem__ that dissolves strings of form "key1/key2/key3" into
        lookups of form self[key1][key2][key3].

        Parameters
        ----------
        key

        Returns
        -------
        item
        """
        # check type:
        if not isinstance(key, str):
            raise TypeError("Keys must be strings of format `key1/key2/...`.")

        if "/" in key:
            top, *remainder = key.split("/")
            remainder = "/".join(remainder)
            return self._getter(top)[remainder]
        else:
            return self._getter(key)

    def _getter(self, key):
        """Invoked by __getitem__ or [] slicing. Needs to be implemented in subclass."""
        raise NotImplementedError

    @classmethod
    def from_template(cls, template):
        """Invoke IR instance from a given template"""
        raise NotImplementedError

    def __contains__(self, key):

        try:
            self[key]
        except KeyError:
            return False
        else:
            return True
