"""Interfaces for loading and saving (dumping) templates and circuits from/to file."""
from typing import Optional

FILEIOMODES = ["pickle", "yaml"]


def save(_obj: object, filename: str, filetype: str, **kwargs):
    """Save an object to file.

    Parameters
        ----------
        _obj
            The object to dump to file. It must either be pickleable or have a `to_dict` method for 'yaml' mode.
        filename
            Path to file (absolute or relative).
        filetype
            Chooses which loader to use to load the file. Allowed types: pickle, yaml

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

        from pyrates.frontend.fileio import yaml
        yaml.dump_to_yaml(_obj, filename, **kwargs)
        print(f"{_obj} successfully dumped to {filename} in YAML format.")

    else:

        raise ValueError(f"Unknown file format to save to. Allowed modes: {FILEIOMODES}")
