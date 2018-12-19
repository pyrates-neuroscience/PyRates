
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
from typing import Tuple, Iterator, Union

__author__ = "Daniel Rose"
__status__ = "Development"


class AbstractBaseIR:
    """Abstract base class for intermediate representation classes"""

    def __init__(self, template: str = None):
        self.template = template

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

    # @classmethod
    # def from_dict(cls, **kwargs):
    #     """Initialize an IR class from a dictionary with all relevant values instead of something else."""
    #
    #     module = import_module(cls.__module__)
    #     template_cls = getattr(module, f"{cls.__name__[-2]}Template")
    #     return template_cls(name="", path="", **kwargs).apply


