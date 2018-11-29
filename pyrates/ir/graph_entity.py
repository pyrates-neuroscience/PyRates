"""
"""
from networkx import DiGraph, find_cycle, NetworkXNoCycle

from pyrates import PyRatesException
from pyrates.frontend.operator import OperatorTemplate
from pyrates.ir.abc import AbstractBaseIR

__author__ = "Daniel Rose"
__status__ = "Development"


class GraphEntityIR(AbstractBaseIR):
    """Intermediate representation for nodes and edges."""

    def __init__(self, operators: dict, template: str =None, values: dict=None):

        self.op_graph = DiGraph()
        all_outputs = {}  # type: Dict[str, dict]
        self.template = template
        # op_inputs, op_outputs = set(), set()

        value_updates = {}
        if values:
            # values.pop("weight", None)
            # values.pop("delay", None)
            for key, value in values.items():
                op_name, var_name = key.split("/")
                if op_name not in value_updates:
                    value_updates[op_name] = {}
                value_updates[op_name][var_name] = value

        for key, item in operators.items():
            if isinstance(key, OperatorTemplate):
                op_template = key
                values_to_update = item

                if values_to_update is None:
                    values_to_update = {}
                if op_template.name in value_updates:
                    values_to_update.update(value_updates.pop(op_template.name, {}))
                op_instance, op_variables, key = op_template.apply(return_key=True,
                                                                   values=values_to_update)

            elif isinstance(key, str):
                op_instance = item["operator"]
                op_variables = item["variables"]

            else:
                raise TypeError(f"Unknown type of key `{key}` in operators dictionary")

            # add operator as node to local operator_graph
            # ToDo: separate variable def and operator def so one can be private and the other shared
            self.op_graph.add_node(key, operator=op_instance, variables=op_variables)

            # collect all output variables
            out_var = op_instance.output

            # check, if variable name exists in outputs and create empty list if it doesn't
            if out_var not in all_outputs:
                all_outputs[out_var] = {}

            all_outputs[out_var][key] = out_var
            # this assumes input and output variables map on each other by equal name
            # with additional information, non-equal names could also be mapped here

        # fail gracefully, if any variables remain in value_updates which means, the there is some typo
        if value_updates:
            raise PyRatesException("Found value updates that did not fit any operator by name. This may be due to a "
                                   "typo in specifying the operator or variable to update. Remaining variables:"
                                   f"{value_updates}")

        # link outputs to inputs
        for op_key in self.op_graph.nodes:
            for in_var in self[op_key].inputs:
                if in_var in all_outputs:
                    # link all collected outputs of given variable in inputs field of operator
                    for predecessor, out_var in all_outputs[in_var].items():
                        # add predecessor output as source; this would also work for non-equal variable names
                        self[op_key].inputs[in_var]["sources"].append(predecessor)
                        self.op_graph.add_edge(predecessor, op_key)
                else:
                    pass  # means, that 'source' will remain an empty list and no incoming edge will be added

        try:
            find_cycle(self.op_graph)
        except NetworkXNoCycle:
            pass
        else:
            raise PyRatesException("Found cyclic operator graph. Cycles are not allowed for operators within one node "
                                   "or edge.")

    def _getter(self, key: str):
        """
        Inoked by __getitem__. Returns operator specified by 'key'
        Parameters
        ----------
        key

        Returns
        -------
        operator
        """

        try:
            return self.op_graph.nodes[key]["operator"]
        except KeyError as e:
            if key in str(e):
                raise KeyError(f"Could not find operator '{key}''")
            else:
                raise e

    def __getitem__(self, key: str):
        """More specific implementation of __getitem__ that distinguishes between operator or variable as output"""

        # check type:
        if not isinstance(key, str):
            raise TypeError("Keys must be strings of format `key1/key2/...`.")

        try:
            if "/" in key:
                # assume it is operator/variable
                op, var = key.split("/")
                item = self.op_graph.nodes[op]["variables"][var]
            else:
                # assume it is only operator
                item = self.op_graph.nodes[key]["operator"]
        except KeyError as e:
            if hasattr(self, key):
                item = getattr(self, key)
            else:
                raise e

        return item

    @classmethod
    def from_template(cls, template, values: dict=None):

        return cls(operators=template.operators, template=template, values=values)