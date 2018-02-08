""" Utility functions to store Circuit configurations and data in JSON files
and read/construct circuit from JSON.
"""
from collections import OrderedDict
from typing import Generator, Tuple, Any

from networkx import node_link_data

__author__ = "Daniel Rose"
__status__ = "Development"

# from typing import Union, List
from inspect import getsource
import numpy as np
import json


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):

        # from core.population import Population
        # from core.synapse import Synapse

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, RepresentationBase):
            return obj.to_dict()
        # elif isinstance(obj, Synapse):
        #     attr_dict = obj.to_dict()
        #     # remove synaptic_kernel from dict, if kernel_function is specified
        #     if "kernel_function" in attr_dict:
        #       JansenRitCircuit  attr_dict.pop("synaptic_kernel", None)
        #     return attr_dict
        # elif isinstance(obj, Axon):
        #     return get_attrs(obj)
        elif callable(obj):
            return getsource(obj)
        else:
            return super().default(obj)


class RepresentationBase(object):
    """Class that implements a __repr__ that yields the __init__ function signature with provided arguments in the
    form 'module.Class(arg1, arg2, arg3)'"""

    def __repr__(self) -> str:
        """Magic method that returns to repr(object). It retrieves the signature of __init__ and maps it to class
        attributes to return a representation in the form 'module.Class(arg1=x, arg2=y, arg3=3)'. Raises AttributeError,
        if the a parameter in the signature cannot be found (is not saved). The current implementation """

        # repr_str = f"{self.__module__!r}.{self.__class__!r}("
        # params = self._get_params()

        param_str = ", ".join((f"{name}={value!r}" for name, value in self._get_params()))
        _module = self.__class__.__module__
        _class = self.__class__.__name__

        return f"{_module}.{_class}({param_str})"

    def _get_params(self, include_defaults=False) -> Generator[Tuple[str, Any], None, None]:
        """Compile parameters given to __init__ into a list."""

        import inspect

        # retrieve signature of __init__ method
        sig = inspect.signature(self.__init__)

        for name, param in sig.parameters.items():
            if hasattr(self, "transfer_function_args") and name in self.transfer_function_args:
                    value = self.transfer_function_args[name]
            elif hasattr(self, "kernel_function_args") and name in self.kernel_function_args:
                    value = self.kernel_function_args[name]
            else:
                try:
                    value = getattr(self, name)
                except AttributeError:
                    raise AttributeError(f"Could not find attribute '{name}' that was defined in the signature of "
                                         f"__init__.")

            if not include_defaults:
                if np.all(param.default != inspect.Parameter.empty):
                    if np.all(value == param.default):
                        continue

            yield (name, value)

    def to_dict(self, include_defaults=False, include_graph=False, recursive=False) -> OrderedDict:
        """Parse representation string to a dictionary to later convert it to json."""

        _dict = OrderedDict()
        _dict["class"] = {"__module__": self.__class__.__module__, "__name__": self.__class__.__name__}
        # _dict["module"] = self.__class__.__module__
        # _dict["class"] = self.__class__.__name__

        for name, value in self._get_params(include_defaults=include_defaults):
            _dict[name] = value

        # Include graph if desired
        from core.circuit import Circuit
        if include_graph and isinstance(self, Circuit):
            net_dict = node_link_data(self.network_graph)
            if recursive:
                for node in net_dict["nodes"]:
                    node["data"].to_dict(include_defaults=include_defaults, recursive=True)
            _dict["network_graph"] = net_dict

        # Apply dict transformation recursively to all relevant objects
        if recursive:
            for key, item in _dict.items():
                if isinstance(item, RepresentationBase):
                    _dict[key] = item.to_dict(include_defaults=include_defaults, recursive=True)

        return _dict

    def to_json(self, include_defaults=False, include_graph=False, path="", filename=""):
        """Parse a dictionary into """

        # from core.utility.json_filestorage import CustomEncoder

        _dict = self.to_dict(include_defaults=include_defaults, include_graph=include_graph, recursive=True)

        if filename:
            import os
            import errno

            filepath = os.path.join(path, filename)

            # create directory if necessary
            if not os.path.exists(os.path.dirname(filepath)):
                try:
                    os.makedirs(os.path.dirname(filepath))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            with open(filepath, "w") as outfile:
                json.dump(_dict, outfile, cls=CustomEncoder, indent=4)

        # return json as string
        return json.dumps(_dict, cls=CustomEncoder)











