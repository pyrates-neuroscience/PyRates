"""
"""
from typing import Tuple, Iterator, Union

__author__ = "Daniel Rose"
__status__ = "Development"


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
        # TODO:
        #  idea: split key and then format into an iterator/generator --> hand iterator to getter function,
        #  so it can decide locally, if it wants to process more than one item in the iterator.
        # check type:
        if not isinstance(key, str):
            raise TypeError("Keys must be strings of format `key1/key2/...`.")

        # split key by slash (/) into namespaces and put them into an iterator
        key_iter = iter(key.split("/"))
        try:
            key = next(key_iter)
            item = self.getitem_from_iterator(key, key_iter)
            for key in key_iter:
                item = item.getitem_from_iterator(key, key_iter)
        except KeyError as e:
            if hasattr(self, key):
                item = getattr(self, key)
            else:
                raise e

        return item

    def getitem_from_iterator(self, key: str, key_iter: Iterator[str]):
        """Invoked by __getitem__ or [] slicing. Needs to be implemented in subclass."""
        raise NotImplementedError

    @classmethod
    def from_template(cls, template, **kwargs):
        """Invoke IR instance from a given template"""
        raise NotImplementedError

    def __contains__(self, key):

        try:
            self[key]
        except KeyError:
            return False
        else:
            return True

    def to_dict(self) -> dict:
        """Return a dictionary representation of the intermediate representation"""
        raise NotImplementedError

    # @classmethod
    # def from_dict(cls, **kwargs):
    #     """Initialize an IR class from a dictionary with all relevant values instead of something else."""
    #
    #     module = import_module(cls.__module__)
    #     template_cls = getattr(module, f"{cls.__name__[-2]}Template")
    #     return template_cls(name="", path="", **kwargs).apply

    def to_yaml(self, path: str, name: str):

        dict_repr = {name: self.to_dict()}

        from ruamel.yaml import YAML
        yaml = YAML()

        from pyrates.utility.filestorage import create_directory
        create_directory(path)
        from pathlib import Path
        path = Path(path)
        yaml.dump(dict_repr, path)


