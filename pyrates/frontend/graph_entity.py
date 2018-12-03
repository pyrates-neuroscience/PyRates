from typing import Union, List, Type

from pyrates.frontend.abc import AbstractBaseTemplate
from pyrates.ir.graph_entity import GraphEntityIR
from pyrates.frontend.operator import OperatorTemplate
from pyrates.frontend.parser.yaml import TemplateLoader


class GraphEntityTemplate(AbstractBaseTemplate):
    target_ir = GraphEntityIR

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

    def apply(self, values: dict = None):

        # ToDo: change so that only IR classes are forwarded instead of templates.
        return self.target_ir(operators=self.operators, template=self.path, values=values)


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
