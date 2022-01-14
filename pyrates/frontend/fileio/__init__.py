"""Interfaces for loading and saving (dumping) templates and circuits from/to file."""
from typing import Optional

FILEIOMODES = ["pickle", "yaml"]


def save(_obj: object, filename: str, filetype: str, template_name: Optional[str] = None, **kwargs):
    """Save an object to file.

    Parameters
        ----------
        _obj
            The object to dump to file. It must either be pickleable or have a `to_dict` method for 'yaml' mode.
        filename
            Path to file (absolute or relative).
        filetype
            Chooses which loader to use to load the file. Allowed types: pickle, yaml
        template_name
            This argument is needed for filetype 'yaml'. The object is saved under that name within the file.

        Returns
        -------
        None
    """

    if filetype == "pickle":
        # make sure the directory exists
        from pyrates.utility import create_directory
        create_directory(filename)

        from pyrates.frontend.fileio import pickle
        pickle.dump(_obj, filename, **kwargs)
        print(f"{_obj} successfully pickled to {filename}")

    elif filetype == "yaml":
        try:
            _dict = _obj.to_dict()  # type: dict
        except AttributeError:
            try:
                _dict = dict(dict(nodes=_obj.nodes, edges=_obj.edges, base='CircuitTemplate'))
            except AttributeError:
                raise AttributeError(f"The object {_obj} does not have a `to_dict` method and is no instance of "
                                     f"`CircuitTemplate`. Please choose a different target filetype than 'yaml'.")

        # make sure the directory exists
        from pyrates.utility import create_directory
        create_directory(filename)

        from pyrates.frontend.fileio import yaml
        yaml.dump(_dict, filename, template_name, **kwargs)
        print(f"{_obj} successfully dumped to {filename} in YAML format.")

    else:

        raise ValueError(f"Unknown file format to save to. Allowed modes: {FILEIOMODES}")
