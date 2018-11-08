from typing import List

from pyrates import PyRatesException
from pyrates.frontend.graph_entity import GraphEntityTemplate, GraphEntityTemplateLoader, GraphEntityIR


class EdgeTemplate(GraphEntityTemplate):
    """Generic template for an edge in the computational network graph. A single edge may encompass several
    different operators. One template defines a typical structure of a given edge type."""

    pass


class EdgeIR(GraphEntityIR):

    def __init__(self, operators: dict, template: EdgeTemplate = None, values: dict=None):

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
        else:
            raise PyRatesException("Too many or too little input variables found. Exactly one input variable is "
                                   "required per edge.")

        # 2: get reference for target variable
        ######################################

        # try to find single output variable
        # noinspection PyTypeChecker
        out_op = [op for op, out_degree in self.op_graph.out_degree if out_degree == 0]  # type: List[str]

        # only one single output operator allowed
        if len(out_op) != 1:
            raise PyRatesException("Too many or too little output operators found. Exactly one output operator and "
                                   "associated output variable is required per edge.")

        out_var = self[out_op[0]].output

        self.output = f"{out_op[0]}/{out_var}"

    @classmethod
    def from_template(cls, template, values: dict=None):

        instance = super().from_template(template, values)
        return instance


class EdgeTemplateLoader(GraphEntityTemplateLoader):
    """Template loader specific to an EdgeOperatorTemplate. """

    def __new__(cls, path):
        return super().__new__(cls, path, EdgeTemplate)

    @classmethod
    def update_template(cls, *args, **kwargs):
        """Update all entries of a base node template to a more specific template."""

        return super().update_template(EdgeTemplate, *args, **kwargs)
