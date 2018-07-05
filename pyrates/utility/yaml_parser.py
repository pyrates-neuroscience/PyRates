""" Some utility functions for parsing YAML-based definitions of circuits and components.
"""

__author__ = "Daniel Rose"
__status__ = "Development"

import importlib


def import_template(path: str, child=None):
    """Function that imports a template defined in the form "path.to.template".
    Current implementation assumes that templates are only defined in YAML

    Parameters:
    -----------

    pointer:
        String of the form path.to.template
    """

    # Part 1: load template dictionary from yaml
    # ------------------------------------------

    if "." in path:
        parts = path.split(".")
        name = parts[-1]

        if parts[0] == "pyrates":
            # import built-in templates
            # get absolute path of pyrates library
            file = parts[-2]
            parentdir = ".".join(parts[:-2])
            module = importlib.import_module(parentdir)

            abspath = module.__path__[0]

            template = load_template_from_yaml(name, file, abspath)
        else:
            import os
            file = os.path.join(parts[:-1])
            template = load_template_from_yaml(name, file)
    elif child is not None:
        name = path
        template = load_template_from_yaml(name, child["__path__"])
    else:
        raise NotImplementedError
        # this should only happen, if "base" is specified, but empty

    # Part 2: load and replace base template
    # --------------------------------------

    # recursively load base template. most basic template does not have base
    if "base" in template:
        # child.pop("base")
        _, template = import_template(template["base"], child=template)

        # adapt template based on child template
    if child is not None:
        template = update_template(template, child)

    return name, template


def update_template(base, child):
    if isinstance(base, dict) and isinstance(child, dict):
        for key in child:
            if key in base:
                base[key] = update_template(base[key], child[key])
            else:
                base[key] = child[key]

    # this part is supposed to replace equations
    elif isinstance(child, dict) and isinstance(base, str):
        if "replace" in child:
            for old, new in child.pop("replace").items():
                base = base.replace(old, new)

        if "remove" in child:
            for old in child.pop("remove"):
                base = base.replace(old, "")

        if "append" in child:
            for new in child.pop("append"):
                base = f"{base} {new}"

        if "conditions" in child:
            child.pop("conditions")
            pass  # Not implemented yet

        for key in child:
            raise NotImplementedError(f"The string/equation manipulation option {key} is not supported")

    else:
        base = child
    return base


def load_template_from_yaml(name: str, filename: str, abspath: str = ""):
    from ruamel.yaml import YAML
    import os

    yaml = YAML(typ="safe", pure=True)

    if not filename.endswith(".yaml"):
        filename = f"{filename}.yaml"

    filepath = os.path.join(abspath, filename)

    with open(filepath, "r") as file:
        yaml_dict = yaml.load(file)

    if name in yaml_dict:
        template = yaml_dict[name]
        template["__path__"] = filepath
    else:
        raise AttributeError(f"Could not find {name} in {filename}.")

    return template