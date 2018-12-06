"""
"""
from typing import List

from pyrates import PyRatesException
from pyrates.ir.node import NodeIR

__author__ = "Daniel Rose"
__status__ = "Development"


class EdgeIR(NodeIR):

    def __init__(self, operators: dict=None, template: str = None):

        super().__init__(operators, template)

    @property
    def input(self):
        """Detect input variable of edge, assuming only one input variable exists. This also references the operator
        the variable belongs to."""
        # noinspection PyTypeChecker
        in_op = [op for op, in_degree in self.op_graph.in_degree if in_degree == 0]  # type: List[str]

        # multiple input operations are possible, as long as they require the same singular input variable
        in_var = set()
        for op_key in in_op:
            for var in self[op_key].inputs:
                in_var.add(f"{op_key}/{var}")

        if len(in_var) == 1:
            return in_var.pop()
        elif len(in_var) == 0:
            return None
        else:
            raise PyRatesException("Too many input variables found. Exactly one or zero input variables are "
                                   "required per edge.")

    @property
    def input_var(self):
        return self.input

    @property
    def output(self):
        """Detect output variable of edge, assuming only one output variable exists."""

        # try to find single output variable
        # noinspection PyTypeChecker
        out_op = [op for op, out_degree in self.op_graph.out_degree if out_degree == 0]  # type: List[str]

        # only one single output operator allowed
        if len(out_op) == 1:
            out_var = self[out_op[0]].output
            return f"{out_op[0]}/{out_var}"
        elif len(out_op) == 0:
            return None
        else:
            raise PyRatesException("Too many or too little output operators found. Exactly one output operator and "
                                   "associated output variable is required per edge.")

    @property
    def output_var(self):
        return self.output

