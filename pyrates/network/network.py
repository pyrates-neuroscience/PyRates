"""

"""

# external imports
import tensorflow as tf
from typing import Optional

# pyrates imports
from pyrates.parser import EquationParser
from pyrates.node import Node


class Network(object):

    def __init__(self,
                 node_dict: dict,
                 dt: float = 1e-3,
                 vectorize: bool = True,
                 tf_graph: Optional[tf.Graph] = None,
                 key: Optional[str] = None
                 ) -> None:

        self.key = key if key else 'net0'
        self.dt = dt
        self.tf_graph = tf_graph if tf_graph else tf.get_default_graph()
        self.nodes = dict()

        with self.tf_graph.as_default():

            if vectorize:
                pass
            else:

                node_updates = []

                for node_name, node_info in node_dict.items():

                    node_ops = dict()
                    node_args = dict()
                    for key, val in node_info.items():
                        if 'operator' in key:
                            node_ops[key] = val
                        else:
                            node_args[key] = val

                    node = Node(node_ops, node_args, node_name, self.tf_graph)
                    self.nodes[node_name] = node
                    node_updates += node.update

                self.update = tf.tuple(node_updates)

