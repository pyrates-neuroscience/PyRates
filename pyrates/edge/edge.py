"""This module contains an edge class that allows to connect variables on a source and a target node via an operator.
"""

# external imports
import tensorflow as tf
from typing import Optional, List
from typing import Union, List

# pyrates imports
from pyrates.node.node import NodeTemplateLoader, GraphEntityTemplate, GraphEntityTemplateLoader
from pyrates.operator import Operator
from pyrates.parser import parse_dict
from pyrates.node import Node
from pyrates.abc import AbstractBaseTemplate
from pyrates.operator import OperatorTemplate
from pyrates.utility.yaml_parser import TemplateLoader

# meta infos
__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


class Edge(object):
    """Base edge class to connect a tensorflow variable on source to a tensorflow variable on target via an
    operator.

    Parameters
    ----------
    source
    target
    coupling_op
    coupling_op_args
    key
    tf_graph

    Attributes
    ----------

    Methods
    -------

    Examples
    --------

    References
    ----------

    """

    def __init__(self,
                 source: Node,
                 target: Node,
                 coupling_op: List[str],
                 coupling_op_args: dict,
                 key: Optional[str] = None,
                 tf_graph: Optional[tf.Graph] = None):
        """Instantiates edge.
        """

        self.operator = coupling_op
        self.key = key if key else 'edge0'
        self.tf_graph = tf_graph if tf_graph else tf.get_default_graph()

        with self.tf_graph.as_default():

            with tf.variable_scope(self.key):

                # create operator
                #################

                # replace the coupling operator arguments with the fields from source and target where necessary
                co_args = {}
                for key, val in coupling_op_args.items():
                    if val['variable_type'] == 'source_var':
                        try:
                            var = getattr(source, val['name'])
                            co_args[key] = {'variable_type': 'raw', 'variable': var}
                        except AttributeError:
                            pass

                    elif val['variable_type'] == 'target_var':
                        try:
                            var = getattr(target, val['name'])
                            co_args[key] = {'variable_type': 'raw', 'variable': var}
                        except AttributeError:
                            pass
                    else:
                        co_args[key] = val

                # create tensorflow variables from the additional operator args
                tf_vars, var_names = parse_dict(var_dict=co_args,
                                                var_scope=self.key,
                                                tf_graph=self.tf_graph)

                operator_args = dict()

                # bind operator args to edge
                for tf_var, var_name in zip(tf_vars, var_names):
                    setattr(self, var_name, tf_var)
                    operator_args[var_name] = tf_var

                # instantiate operator
                operator = Operator(expressions=coupling_op,
                                    expression_args=operator_args,
                                    tf_graph=self.tf_graph,
                                    key=self.key,
                                    variable_scope=self.key)

                # bind newly created tf variables to edge
                for var_name, tf_var in operator.args.items():
                    if not hasattr(self, var_name):
                        setattr(self, var_name, tf_var)

                # connect source and target variables via operator
                self.project = operator.create()


class EdgeTemplate(GraphEntityTemplate):
    """Generic template for an edge in the computational network graph. A single edge may encompass several
    different operators. One template defines a typical structure of a given edge type."""

    pass


class EdgeTemplateLoader(GraphEntityTemplateLoader):
    """Template loader specific to an EdgeOperatorTemplate. """

    def __new__(cls, path):

        return super().__new__(cls, path, EdgeTemplate)

    @classmethod
    def update_template(cls, *args, **kwargs):
        """Update all entries of a base node template to a more specific template."""

        return super().update_template(EdgeTemplate, *args, **kwargs)
