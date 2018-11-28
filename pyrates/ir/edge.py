"""
"""
from pyrates import PyRatesException
from pyrates.ir.graph_entity import GraphEntityIR

__author__ = "Daniel Rose"
__status__ = "Development"


class EdgeIR(GraphEntityIR):

    def __init__(self, operators: dict=None, template: str = None, values: dict=None):

        if operators is None:
            operators = {}
        super().__init__(operators, template, values)

        # Step 1: Detect edge input variable
        ####################################
        # noinspection PyTypeChecker
        in_op = [op for op, in_degree in self.op_graph.in_degree if in_degree == 0]  # type: List[str]

        # multiple input operations are possible, as long as they require the same singular input variable
        in_var = set()
        for op_key in in_op:
            for var in self[op_key].inputs:
                in_var.add(f"{op_key}/{var}")

        if len(in_var) == 1:
            self.input = in_var.pop()
        elif len(in_var) == 0:
            self.input = None
        else:
            raise PyRatesException("Too many input variables found. Exactly one or zero input variables are "
                                   "required per edge.")

        # 2: get reference for target variable
        ######################################

        # try to find single output variable
        # noinspection PyTypeChecker
        out_op = [op for op, out_degree in self.op_graph.out_degree if out_degree == 0]  # type: List[str]

        # only one single output operator allowed
        if len(out_op) == 1:
            out_var = self[out_op[0]].output
            self.output = f"{out_op[0]}/{out_var}"
        elif len(out_op) == 0:
            self.output = None
        else:
            raise PyRatesException("Too many or too little output operators found. Exactly one output operator and "
                                   "associated output variable is required per edge.")

    @classmethod
    def from_template(cls, template, values: dict=None):

        instance = super().from_template(template, values)
        return instance