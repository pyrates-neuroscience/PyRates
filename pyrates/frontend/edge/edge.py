from pyrates.frontend.graph_entity import GraphEntityTemplate, GraphEntityTemplateLoader
from pyrates.ir.edge import EdgeIR


class EdgeTemplateLoader(GraphEntityTemplateLoader):
    """Template loader specific to an EdgeOperatorTemplate. """

    def __new__(cls, path):
        return super().__new__(cls, path, EdgeTemplate)

    @classmethod
    def update_template(cls, *args, **kwargs):
        """Update all entries of a base node template to a more specific template."""

        return super().update_template(EdgeTemplate, *args, **kwargs)


class EdgeTemplate(GraphEntityTemplate):
    """Generic template for an edge in the computational backend graph. A single edge may encompass several
    different operators. One template defines a typical structure of a given edge type."""

    target_ir = EdgeIR
