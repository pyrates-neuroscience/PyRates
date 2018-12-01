"""
"""
from copy import deepcopy

from pyrates.ir.circuit import CircuitIR
from pyrates.ir.node import NodeIR
from pyrates.ir.edge import EdgeIR
from pyrates.ir.operator import OperatorIR

__author__ = "Daniel Rose"
__status__ = "Development"


def node_from_dict(node_dict: dict):

    operators = {}
    operator_args = node_dict["operator_args"]

    for key, item in node_dict["operators"].items():
        operators[key] = {"operator": operator_from_dict(item),
                          "variables": {}}

    for key, item in operator_args.items():
        op_name, var_name = key.split("/")
        operators[op_name]["variables"][var_name] = item

    return NodeIR(operators=operators)


def operator_from_dict(op_dict: dict):

    return OperatorIR(equations=op_dict["equations"], inputs=op_dict["inputs"], output=op_dict["output"])


def circuit_to_dict(circuit: CircuitIR):
    """Reformat graph structure into a dictionary that can be saved as YAML template. The current implementation assumes
    that nodes and edges are given by as templates."""

    node_dict = {}
    for node_key, node_data in circuit.nodes(data=True):
        node = node_data["node"]
        if node.template:
            node_dict[node_key] = node.template.path
        else:
            # if no template is given, build and search deeper for node templates
            pass

    edge_list = []
    for source, target, edge_data in circuit.edges(data=True):
        edge_data = deepcopy(edge_data)
        edge = edge_data.pop("edge_ir")
        source = f"{source}/{edge_data['source_var']}"
        target = f"{target}/{edge_data['target_var']}"
        edge_list.append((source, target, edge.template.path, dict(weight=edge_data["weight"],
                                                                   delay=edge_data["delay"])))

    # use Python template as base, since inheritance from YAML templates is ambiguous for circuits
    base = "CircuitTemplate"

    return dict(nodes=node_dict, edges=edge_list, base=base)
