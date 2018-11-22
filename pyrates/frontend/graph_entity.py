from typing import Union, List, Type, Dict

from networkx import DiGraph, find_cycle, NetworkXNoCycle

from pyrates import PyRatesException
from pyrates.frontend.abc import AbstractBaseTemplate, AbstractBaseIR
from pyrates.frontend.operator import OperatorTemplate
from pyrates.frontend.parser.yaml import TemplateLoader


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
                    for predecessor in all_outputs[in_var].keys():
                        # add predecessor output as source; this would also work for non-equal variable names
                        if predecessor not in self[op_key].inputs[in_var]["sources"]:
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


    @classmethod
    def from_template(cls, template, values: dict=None):

        return cls(operators=template.operators, template=template, values=values)


class GraphEntityTemplate(AbstractBaseTemplate):

    def __init__(self, name: str, path: str, operators: Union[str, List[str], dict],
                 description: str = "A node or an edge.", label: str = None):
        """For now: only allow single equation in operator template."""

        super().__init__(name, path, description)

        if label:
            self.label = label
        else:
            self.label = self.name

        self.operators = {}  # dictionary with operator path as key and variations to the template as values
        if isinstance(operators, str):
            operator_template = self._load_operator_template(operators)
            self.operators[operator_template] = {}  # single operator path with no variations
        elif isinstance(operators, list):
            for op in operators:
                operator_template = self._load_operator_template(op)
                self.operators[operator_template] = {}  # multiple operator paths with no variations
        elif isinstance(operators, dict):
            for op, variations in operators.items():
                operator_template = self._load_operator_template(op)
                self.operators[operator_template] = variations
        # for op, variations in operators.items():
        #     if "." not in op:
        #         op = f"{path.split('.')[:-1]}.{op}"
        #     self.operators[op] = variations

    def _load_operator_template(self, path: str) -> OperatorTemplate:
        """Load an operator template based on a path"""
        path = self._format_path(path)
        return OperatorTemplate.from_yaml(path)

    def apply(self, values: dict=None):
        return super().apply(values)


class GraphEntityTemplateLoader(TemplateLoader):

    def __new__(cls, path, template_class):

        return super().__new__(cls, path, template_class)

    @classmethod
    def update_template(cls, template_cls: Type[GraphEntityTemplate], base, name: str, path: str, label: str,
                        operators: Union[str, List[str], dict] = None,
                        description: str = None):
        """Update all entries of a base edge template to a more specific template."""

        if operators:
            cls.update_operators(base.operators, operators)
        else:
            operators = base.operators

        if not description:
            description = base.__doc__  # or do we want to enforce documenting a template?

        return template_cls(name=name, path=path, label=label, operators=operators,
                            description=description)

    @staticmethod
    def update_operators(base_operators: dict, updates: Union[str, List[str], dict]):
        """Update operators of a given template. Note that currently, only the new information is
        propagated into the operators dictionary. Comparing or replacing operators is not implemented.

        Parameters:
        -----------

        base_operators:
            Reference to one or more operators in the base class.
        updates:
            Reference to one ore more operators in the child class
            - string refers to path or name of single operator
            - list refers to multiple operators of the same class
            - dict contains operator path or name as key
        """
        # updated = base_operators.copy()
        updated = {}
        if isinstance(updates, str):
            updated[updates] = {}  # single operator path with no variations
        elif isinstance(updates, list):
            for path in updates:
                updated[path] = {}  # multiple operator paths with no variations
        elif isinstance(updates, dict):
            for path, variations in updates.items():
                updated[path] = variations
            # dictionary with operator path as key and variations as sub-dictionary
        else:
            raise TypeError("Unable to interpret type of operator updates. Must be a single string,"
                            "list of strings or dictionary.")
        # # Check somewhere, if child operators have same input/output as base operators?
        #
        return updated