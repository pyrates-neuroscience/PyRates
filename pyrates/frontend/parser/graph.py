"""
"""

import networkx as nx
from pyrates.frontend import CircuitIR, EdgeIR
from . import type_mapping
from .dictionary import node_from_dict


__author__ = "Daniel Rose"
__status__ = "Development"


def circuit_from_graph(graph: nx.MultiDiGraph, label="circuit",
                       node_creator=node_from_dict):
    """Create a CircuitIR instance out of a networkx.MultiDiGraph"""

    circuit = CircuitIR(label)

    for name, data in graph.nodes(data=True):

        circuit.add_node(name, node=node_creator(data))

    required_keys = ["source_var", "target_var", "weight", "delay"]
    for source, target, data in graph.edges(data=True):

        if all([key in data for key in required_keys]):
            if "edge_ir" not in data:
                data["edge_ir"] = EdgeIR()
            circuit.add_edge(source, target, **data)
        else:
            raise KeyError(f"Missing a key out of {required_keys} in an edge with source `{source}` and target"
                           f"`{target}`")

    return circuit




