"""Defines a few custom functions on the backend graph for convenience.
"""

# external packages
from networkx import MultiDiGraph

# pyrates internal imports
from pyrates.population import Population

# meta infos
__author__ = "Daniel Rose, Richard Gast"
__status__ = "Development"


####################
# networkx wrapper #
####################


class WrappedMultiDiGraph(MultiDiGraph):
    """Wrapper for MultiDiGraph that has a few convenience customizations."""

    def add_edge(self, source, target, weight=1, delay=0, synapse=None):

        if synapse is None:

            # connect source to target population (directly)
            source_pop = self.nodes[source]["data"]  # type: Population
            source_pop.connect(self.nodes[target]["data"], weight, delay)

        else:

            # connect source population to synapse on target population
            source_pop = self.nodes[source]["data"]  # type: Population
            source_pop.connect(synapse, weight, delay)

        super().add_edge(source, target, weight=weight, delay=delay, synapse=synapse)



