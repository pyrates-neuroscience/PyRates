""" Utility functions to store Circuit configurations and data in JSON files
and read/construct circuit from JSON.
"""

from typing import Generator, Tuple, Any
from collections import OrderedDict

__author__ = "Daniel Rose"
__status__ = "Development"

# from typing import Union, List
from inspect import getsource
import numpy as np
import json
from networkx import MultiDiGraph, node_link_data


# def get_attrs(obj: object) -> dict:
#     """Transform meta-data of a given object into a dictionary, ignoring default fields form class <object>."""
#     config_dict = dict()
#     for key, item in obj.__dict__.items():
#         config_dict[key] = item
#     config_dict["class"] = {"__module__": obj.__class__.__module__, "__name__": obj.__class__.__name__}
#     return config_dict


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # elif isinstance(obj, Population):
        #     return get_attrs(obj)
        # elif isinstance(obj, Synapse):
        #     attr_dict = get_attrs(obj)
        #     # remove synaptic_kernel from dict, if kernel_function is specified
        #     if "kernel_function" in attr_dict:
        #       JansenRitCircuit  attr_dict.pop("synaptic_kernel", None)
        #     return attr_dict
        # elif isinstance(obj, Axon):
        #     return get_attrs(obj)
        # elif isinstance(obj, MultiDiGraph):
        #     net_dict = node_link_data(obj)
        #     for population in net_dict["nodes"]:
        #         population.pop("data", None)
        #     return net_dict
        elif callable(obj):
            return getsource(obj)
        else:
            return super().default(obj)

# class Data:
#     pass


# class Data:
#     # @staticmethod
#     # def repr(obj):
#     #     items = []
#     #     for key, value in obj.__dict__.items():
#     #         try:
#     #             item = "%s = %r" % (key, value)
#     #             assert len(item) < 20
#     #         except:
#     #             item = "%s: <%s>" % (key, value.__class__.__name__)
#     #         items.append(item)
#     #
#     #     return f"{obj.__class__.__name__}({', '.join(items)})"
#
#     def dict(self):
#         passparams
#
#     def __init__(self, cls):
#         # cls.__repr__ = Data.repr
#         self.cls = cls
#
#     def __call__(self, *args, **kwargs):
#         inst = self.cls(*args, **kwargs)
#
#         def _repr(obj):
#             module = obj.__class__.__module__
#             name = obj.__class__.__name__
#             # return f"{module}.{name}({', '.join(args)}, {kwargs})"
#             return f"{module}.{name}()"
#         inst.__repr__ = _repr
#
#         return inst


# def repr_decorator(cls):
#     def cls_wrapper(*args, **kwargs):
#         def representer(obj):
#             module = obj.__class__.__module__
#             name = obj.__class__.__name__
#             return f"{module}.{name}({', '.join(args)},{kwargs})"
#         inst = cls(*args, **kwargs)
#         inst.__repr__ = representer(inst)
#         return cls(*args, **kwargs)
#     return cls_wrapper
#
#
# def repr_parser(obj: object, *args, **kwargs) -> str:
#     module = obj.__class__.__module__
#     name = obj.__class__.__name__
#     rep_str = f"{module}.{name}({args},{kwargs})"
#     return rep_str


class RepresentationBase:
    """Class that implements a __repr__ that yields the __init__ function signature with provided arguments in the
    form 'module.Class(arg1, arg2, arg3)'"""

    def __repr__(self) -> str:
        """Magic method that returns to repr(object). It retrieves the signature of __init__ and maps it to class
        attributes to return a representation in the form 'module.Class(arg1=x, arg2=y, arg3=3)'. Raises AttributeError,
        if the a parameter in the signature cannot be found (is not saved). The current implementation """

        # repr_str = f"{self.__module__!r}.{self.__class__!r}("
        # params = self._get_params()

        param_str = ", ".join((f"{name}={value}" for name, value in self._get_params()))
        _module = self.__class__.__module__
        _class = self.__class__.__name__

        return f"{_module}.{_class}({param_str})"

    def _get_params(self, include_defaults=False) -> Generator[Tuple[str, Any], None, None]:
        """Compile parameters given to __init__ into a list."""

        import inspect

        # retrieve signature of __init__ method
        sig = inspect.signature(self.__init__)

        for name, param in sig.parameters.items():
            try:
                value = getattr(self, name)
            except AttributeError:
                raise AttributeError(f"Could not find attribute {name} that was defined in the signature of __init__. "
                                     f"Maybe it was not saved?")

            if not include_defaults:
                if np.all(param.default != inspect.Parameter.empty):
                    if np.all(value == param.default):
                        continue

            yield (name, value)

    def to_dict(self, include_defaults=False, recursive=False) -> OrderedDict:
        """Parse representation string to a dictionary to later convert it to json."""

        _dict = OrderedDict()
        _dict["class"] = {"__module__": self.__class__.__module__, "__name__": self.__class__.__name__}
        # _dict["module"] = self.__class__.__module__
        # _dict["class"] = self.__class__.__name__

        for name, value in self._get_params(include_defaults=include_defaults):
            _dict[name] = value

        # Apply dict transformation recursively to all relevant objects
        if recursive:
            for key, item in _dict.items():
                if isinstance(item, RepresentationBase):
                    _dict[key] = item.to_dict(include_defaults=include_defaults, recursive=True)

        return _dict

    def to_json(self, details="min", path="", filename=""):
        """Parse a dictionary into """

        import os

        if details == "min":
            include_defaults = False
        elif details == "defaults":
            include_defaults = True
        elif details == "max":
            raise NotImplementedError("No explicit implementation for maximum details yet.")
        else:
            raise ValueError("Unrecognized argument to parameter 'details'. Expects either 'min' or 'max'.")

        _dict = self.to_dict(include_defaults=include_defaults, recursive=True)

        if filename:
            filepath = os.path.join(path, filename)
            with open(filepath, "w") as outfile:
                json.dump(_dict, outfile, cls=CustomEncoder, indent=4)

        # return json as string
        return json.dumps(_dict, cls=CustomEncoder)


# if __name__ == "__main__":
#
#     @Data
#     class Dummy():
#         def __init__(self, a, b=1):
#             self.a = a
#
#     c = Dummy(a=2)
#     print(c)
