"""
"""
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
