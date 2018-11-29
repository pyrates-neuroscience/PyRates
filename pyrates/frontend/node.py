from pyrates.frontend.graph_entity import GraphEntityTemplate, GraphEntityTemplateLoader
from pyrates.ir.node import NodeIR


class NodeTemplate(GraphEntityTemplate):
    """Generic template for a node in the computational backend graph. A single node may encompass several
    different operators. One template defines a typical structure of a given node type."""

    target_ir = NodeIR


class NodeTemplateLoader(GraphEntityTemplateLoader):
    """Template loader specific to an OperatorTemplate. """

    def __new__(cls, path):
        return super().__new__(cls, path, NodeTemplate)

    @classmethod
    def update_template(cls, *args, **kwargs):
        """Update all entries of a base node template to a more specific template."""

        return super().update_template(NodeTemplate, *args, **kwargs)