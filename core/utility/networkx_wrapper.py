"""Defines a few custom functions on the network graph for convenience.
"""

__author__ = "Daniel Rose"
__status__ = "Development"

from networkx import MultiDiGraph
from core.population import Population


class WrappedMultiDiGraph(MultiDiGraph):
    """Wrapper for MultiDiGraph that has a few convenience customizations."""

    def add_edge(self, source, target, weight=1, delay=0, synapse=None):

        if synapse is None:
            raise ValueError("No Synapse was passed, please pass a target synapse.")

        super().add_edge(source, target, weight=weight, delay=delay, synapse=synapse)

        # add edge also to source population directly without breaking code
        source_pop = self.nodes[source]["data"]  # type: Population
        source_pop.connect(synapse, weight, delay)





