"""
"""
from typing import Iterator

from pyrates.ir.abc import AbstractBaseIR
from pyrates.ir.operator_graph import OperatorGraph

__author__ = "Daniel Rose"
__status__ = "Development"


class NodeIR(AbstractBaseIR):

    def __init__(self, operators: dict=None, template: str=None):

        super().__init__(template)
        self.op_graph = OperatorGraph(operators)

    def getitem_from_iterator(self, key: str, key_iter: Iterator[str]):
        """Alias for self.op_graph.getitem_from_iterator"""

        return self.op_graph.getitem_from_iterator(key, key_iter)

    def __iter__(self):
        """Return an iterator containing all operator labels in the operator graph."""
        return iter(self.op_graph)

    @property
    def operators(self):
        return self.op_graph.operators
