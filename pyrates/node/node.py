"""This module contains the node class used to create a network node from a set of operations.
"""

# external imports
from copy import deepcopy
from typing import Dict, List, Optional, Union, Type
import tensorflow as tf
from networkx import DiGraph, find_cycle, NetworkXNoCycle

# pyrates imports
from pyrates import PyRatesException
from pyrates.operator import Operator, OperatorTemplate
from pyrates.parser import parse_dict
from pyrates.abc import AbstractBaseTemplate
from pyrates.utility.yaml_parser import TemplateLoader

# meta infos
__author__ = "Richard Gast"
__status__ = "Development"


class Node(object):
    """Basic node class. Creates a node from a set of operations plus information about the variables contained therein.
    This node is a tensorflow sub-graph with an `update` operation that can be used to update all state variables
    described by the operators.

    Parameters
    ----------

    Attributes
    ----------

    Methods
    -------

    References
    ----------

    Examples
    --------

    """

    def __init__(self,
                 operations: Dict[str, List[str]],
                 operation_args: dict,
                 key: str,
                 tf_graph: Optional[tf.Graph] = None
                 ) -> None:
        """Instantiates node.
        """

        self.key = key
        self.operations = dict()
        self.tf_graph = tf_graph if tf_graph else tf.get_default_graph()

        # instantiate operations
        ########################

        with self.tf_graph.as_default():

            with tf.variable_scope(self.key):

                # get tensorflow variables and the variable names from operation_args
                tf_vars, var_names = parse_dict(var_dict=operation_args,
                                                var_scope=self.key,
                                                tf_graph=self.tf_graph)

                operator_args = dict()

                # bind tensorflow variables to node and save them in dictionary for the operator class
                for tf_var, var_name in zip(tf_vars, var_names):
                    setattr(self, var_name, tf_var)
                    operator_args[var_name] = tf_var

                tf_op = tf.no_op()

                for op_name, op in operations.items():

                    # store operator equations
                    self.operations[op_name] = op

                    with tf.control_dependencies([tf_op]):

                        # create operator
                        operator = Operator(expressions=op,
                                            expression_args=operator_args,
                                            tf_graph=self.tf_graph,
                                            key=op_name,
                                            variable_scope=self.key)

                        # collect tensorflow operator
                        tf_op = operator.create()

                    # bind newly created tf variables to node
                    for var_name, tf_var in operator.args.items():
                        if not hasattr(self, var_name):
                            setattr(self, var_name, tf_var)
                            operation_args[var_name] = tf_var

                # group tensorflow versions of all operators
                self.update = tf_op


class GraphEntityTemplate(AbstractBaseTemplate):

    def __init__(self, name: str, path: str, operators: Union[str, List[str], dict],
                 description: str, label: str = None, options: dict = None):
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

        self.options = options
        if options:
            raise NotImplementedError("Using options in node templates is not implemented yet.")

    def _load_operator_template(self, path: str) -> OperatorTemplate:
        """Load an operator template based on a path"""
        path = self._format_path(path)
        return OperatorTemplate.from_yaml(path)

    # def apply(self, options: dict = None, max_iter: int=10000):
    def apply(self, options: dict = None):

        if options:
            raise NotImplementedError("Applying options to a template is not implemented yet.")

        op_graph = DiGraph()
        all_outputs = {}  # type: Dict[str, dict]

        # op_inputs, op_outputs = set(), set()

        for op_template, op_options in self.operators.items():
            op_instance, op_values, key = op_template.apply(op_options, return_key=True)
            # add operator as node to local operator_graph
            # ToDo: separate variable def and operator def so one can be private and the other shared
            op_graph.add_node(key, **deepcopy(op_instance), values=op_values)

            # collect all output variables
            out_var = op_instance["output"]

            # check, if variable name exists in outputs and create empty list if it doesn't
            if out_var not in all_outputs:
                all_outputs[out_var] = {}

            all_outputs[out_var][key] = out_var
            # this assumes input and output variables map on each other by equal name
            # with additional information, non-equal names could also be mapped here

        # link outputs to inputs
        for op_key, data in op_graph.nodes(data=True):
            for in_var in data["inputs"]:
                if in_var in all_outputs:
                    # link all collected outputs of given variable in inputs field of operator
                    for predecessor, out_var in all_outputs[in_var].items():
                        # add predecessor output as source; this would also work for non-equal variable names
                        name = f"{predecessor}/{out_var}"
                        op_graph.nodes[op_key]["inputs"][in_var]["source"].append(name)
                        op_graph.add_edge(predecessor, op_key)
                else:
                    pass  # means, that 'source' will remain an empty list and no incoming edge will be added

        instance = {"operators": op_graph, "template": self}

        try:
            find_cycle(op_graph)
        except NetworkXNoCycle:
            pass
        else:
            raise PyRatesException("Found cyclic operator graph. Cycles are not allowed for operators within one node "
                                   "or edge.")

        return instance


class NodeTemplate(GraphEntityTemplate):
    """Generic template for a node in the computational network graph. A single node may encompass several
    different operators. One template defines a typical structure of a given node type."""

    pass


class GraphEntityTemplateLoader(TemplateLoader):

    def __new__(cls, path, template_class):

        return super().__new__(cls, path, template_class)

    @classmethod
    def update_template(cls, template_cls: Type[GraphEntityTemplate], base, name: str, path: str, label: str,
                        operators: Union[str, List[str], dict] = None,
                        description: str = None,
                        options: dict = None):
        """Update all entries of a base edge template to a more specific template."""

        if operators:
            cls.update_operators(base.operators, operators)
        else:
            operators = base.operators

        if options:
            # copy old options dict
            options = cls.update_options(base.options, options)
        else:
            options = base.options

        if not description:
            description = base.__doc__  # or do we want to enforce documenting a template?

        return template_cls(name=name, path=path, label=label, operators=operators,
                            description=description, options=options)

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
            - dict contains operator path or name as key and options/defaults as sub-dictionaries
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


class NodeTemplateLoader(GraphEntityTemplateLoader):
    """Template loader specific to an OperatorTemplate. """

    def __new__(cls, path):
        return super().__new__(cls, path, NodeTemplate)

    @classmethod
    def update_template(cls, *args, **kwargs):
        """Update all entries of a base node template to a more specific template."""

        return super().update_template(NodeTemplate, *args, **kwargs)
