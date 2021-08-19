
# -*- coding: utf-8 -*-
#
#
# PyRates software framework for flexible implementation of neural 
# network models and simulations. See also: 
# https://github.com/pyrates-neuroscience/PyRates
# 
# Copyright (C) 2017-2018 the original authors (Richard Gast and 
# Daniel Rose), the Max-Planck-Institute for Human Cognitive Brain 
# Sciences ("MPI CBS") and contributors
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
# 
# CITATION:
# 
# Richard Gast and Daniel Rose et. al. in preparation
"""
"""
from typing import Tuple, Iterator, Union, Optional

__author__ = "Daniel Rose"
__status__ = "Development"


class AbstractBaseIR:
    """Abstract base class for intermediate representation classes"""

    __slots__ = ["label", "_template", '_h']

    def __init__(self, label: str, template: str = None):
        self.label = label
        self._template = template

    @property
    def template(self):
        return self._template

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

    def __contains__(self, key):

        try:
            self[key]
        except KeyError:
            return False
        else:
            return True

    def __hash__(self):
        return self._h

    # @classmethod
    # def from_dict(cls, **kwargs):
    #     """Initialize an IR class from a dictionary with all relevant values instead of something else."""
    #
    #     module = import_module(cls.__module__)
    #     template_cls = getattr(module, f"{cls.__name__[-2]}Template")
    #     return template_cls(name="", path="", **kwargs).apply

    def to_file(self, filename: str, filetype: str = "pickle", template_name: Optional[str] = None) -> None:
        """Save an IR object to file. The filetype 'pickle' save the object as is to a file, whereas the 'yaml' option
        creates a template from the IR object. In the latter case you need to define a `template_name` that is used in
        the YAML file.

        Parameters
        ----------
        filename
            Path to file (absolute or relative).
        filetype
            Chooses which loader to use to load the file. Allowed types: pickle, yaml
        template_name
            This is required for the 'yaml' format, in order to define the name of the newly created template.


        Returns
        -------
        None
        """

        if filetype == "pickle":
            from pyrates.frontend.fileio import pickle
            pickle.dump(self, filename)

        else:
            from pyrates.frontend.fileio import FILEIOMODES
            ValueError(f"Unknown file format to save to. Allowed modes: {FILEIOMODES}")

    @classmethod
    def from_file(cls, filename: str, filetype: str = "pickle", error_on_instance_check: bool = True):
        """Load an IR instance from file. The function verifies that the loaded object is indeed an instance of the
        class this function was called from.

        Parameters
        ----------
        filename
            Path to file (relative or absolute)
        filetype
            Indicate which file type to expect
        error_on_instance_check
            Toggle whether or not to raise an error if the instance check fails.


        Returns
        -------
        Any
        """

        if filetype == "pickle":
            from pyrates.frontend.fileio import pickle

            instance = pickle.load(filename)

        else:
            from pyrates.frontend.fileio import FILEIOMODES
            ValueError(f"Unknown file format to save to. Allowed modes: {FILEIOMODES}")

        if isinstance(instance, cls):

            return instance

        else:
            message = f"The object loaded from '{filename}' is not an instance of the class `{cls}` requesting it."
            if error_on_instance_check:
                raise TypeError(message)
            else:
                import warnings
                warnings.warn(message)
