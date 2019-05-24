"""Collection of update functions that work on templates as they are loaded. In general, templates will inherit anything
from their base templates and then overwrite it. Any special keywords are mapped on special update functions.
"""
# from inspect import getargspec

# from pyrates.frontend.template.circuit.circuit import update_edges, update_dict

__author__ = "Daniel Rose"
__status__ = "Development"


# def update_template(tmp_cls, base_template, name: str, path: str, **kwargs):
#     """Update a template with special keywords mapped to special update functions."""
#
#     # use description of base, if no explicit description is given
#     description = kwargs.pop("description", base_template.__doc__)
#     getargspec
#     for
#
#     if not label:
#         label = base_template.label
#
#     if nodes:
#         nodes = update_dict(base_template.nodes, nodes)
#     else:
#         nodes = base_template.nodes
#
#     if circuits:
#         circuits = update_dict(base_template.circuits, circuits)
#     else:
#         circuits = base_template.circuits
#
#     if edges:
#         edges = update_edges(base_template.edges, edges)
#     else:
#         edges = base_template.edges
#
#     return tmp_cls(name=name, path=path, description=description,
#                    label=label, circuits=circuits, nodes=nodes,
#                    edges=edges)


