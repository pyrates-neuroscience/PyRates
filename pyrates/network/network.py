"""This module provides the network class that should be used to set-up any model. It creates a tensorflow graph that
manages all computations/operations and a networkx graph that represents the network structure (nodes + edges).
"""

# external imports
import tensorflow as tf
from typing import Optional, Tuple, Union, List
from pandas import DataFrame
import time as t
import numpy as np
from networkx import MultiDiGraph
from copy import deepcopy

# pyrates imports
from pyrates.parser import parse_dict, parse_equation

# meta infos
__author__ = "Richard Gast, Karim Ahmed"
__status__ = "development"


class Network(MultiDiGraph):
    """Network level class used to set up and simulate networks of nodes and edges defined by sets of operators.

    Parameters
    ----------
    net_config
        Networkx MultiDiGraph that defines the configuration of the network. Following information need to be contained
        on nodes and edges:
            nodes
                Key, value pairs defining the nodes. They node keys should look as follows:
                `node_type_or_template_name:node_specific_identifier`. Each node value is again a dictionary with a
                key-value pair for `operators` and `operator_args`. Each operator is a key-value pair. The operator keys
                should follow the same naming convention as the node keys. The operator values are dictionaries with the
                following keys:
                    - `equations`: List of equations in string format (e.g. ['a = b + c']).
                    - `inputs`: List of input variable names (e.g. ['b', 'c']). Does not include constants or
                       placeholders.
                    - `output`: Output variable name (e.g. 'a')
                Operator arguments need keys that are equivalent to the respective variable names in the equations and
                their values are dictionaries with the following keys:
                    - `vtype`: Can be one of the following:
                        a) `state_var` for variables that change across simulation steps
                        b) `constant` for constants
                        c) `placeholder` for external inputs fed in during simulation time.
                        d) `raw` for variables that should not be turned into a tensorflow variable
                    - `dtype`: data-type of the variable. See tensorflow datatypes for all options
                       (not required for raw vars).
                    - `name`: Name of the variable in the graph (not required for raw vars).
                    - `shape`: Shape of the variable. Either tuple or list (not required for raw vars).
                    - `value`: Initial value.
            edges
                Key-value pairs defining the edges, similarly to the nodes dict. The edge keys should look as follows:
                `edge_type_or_template_name:source_node_name:target_node_name:edge_specific_identifier`. Each edge value
                is a dictionary with key-value pairs for `operators` and `operator_args`. The operator dictionary
                follows the same structure as on the node level. Operator arguments do in principle as well. However,
                for variables that should be retrieved from the source or target node, the following key-value pairs
                should be used:
                    - `vtype`: `source_var` or `target_var`
                    - 'name`: Name of the respective attribute on the node
    dt
        Integration step-size [unit = s]. Required for systems with differential equations.
    tf_graph
        Instance of a tensorflow graph. Operations/variables will be stored on the default graph if not passed.
    vectorize
        If true, operations and variables on the graph will be automatically vectorized to some extend. Tends to speed
        up simulations substantially.
    key
        Name of the network.

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
                 net_config: MultiDiGraph,
                 dt: float = 1e-3,
                 tf_graph: Optional[tf.Graph] = None,
                 vectorize: bool = False,
                 key: Optional[str] = None
                 ) -> None:
        """Instantiation of network.
        """

        # call of super init
        ####################

        super().__init__()

        # additional object attributes
        ##############################

        self.key = key if key else 'net:0'
        self.dt = dt
        self.tf_graph = tf_graph if tf_graph else tf.get_default_graph()
        self.states = []

        # vectorize passed dictionaries
        ###############################

        if vectorize:
            net_config = self.vectorize(net_config=net_config, first_lvl_vec=vectorize, second_lvl_vec=False)

        # create objects on tensorflow graph
        ####################################

        with self.tf_graph.as_default():

            with tf.variable_scope(self.key):

                # initialize nodes
                ##################

                node_updates = []

                for node_name, node_info in net_config.nodes.items():

                    # add simulation step-size to node arguments
                    node_args = node_info['operator_args']
                    node_args['all_ops/dt'] = {'vtype': 'raw',
                                               'value': self.dt}

                    # add node to network
                    self.add_node(n=node_name,
                                  ops=node_info['operators'],
                                  op_args=node_args,
                                  op_order=node_info['op_order'])

                    # collect update operation of node
                    node_updates.append(self.nodes[node_name]['update'])

                # group the update operations of all nodes
                self.step = tf.tuple(node_updates, name='network_nodes_update')

                # initialize edges
                ##################

                with tf.control_dependencies(self.step):

                    edge_updates = []

                    for source_name, target_name, edge_idx in net_config.edges:

                        # add edge to network
                        edge_info = net_config.edges[source_name, target_name, edge_idx]
                        self.add_edge(u=source_name,
                                      v=target_name,
                                      delay=edge_info['delay'],
                                      weight=edge_info['weight'])

                        # collect project operation of edge
                        edge_updates.append(self.edges[source_name, target_name, edge_idx]['update'])

                    # group project operations of all edges
                    if len(edge_updates) > 0:
                        self.step = tf.tuple(edge_updates, name='network_update')

    def add_node(self, n: str, ops: dict, op_args: dict, op_order: list) -> None:
        """

        Parameters
        ----------
        n
            Node name.
        ops
            Dictionary containing key-value pairs for all operators on the node.
        op_args
            Dictionary containing key-value pairs for all variables used in the operators.
        op_order
            List that determines the order in which the operators are created.

        Returns
        -------
        None

        """

        # add node operations/variables to tensorflow graph
        ###################################################

        node_attr = {}

        with self.tf_graph.as_default():

            with tf.variable_scope(n):

                # handle operation arguments
                ############################

                op_args = {}

                # instantiate operations
                ########################

                tf_ops = []
                for op_name in op_order:

                    op = ops[op_name]

                    # extract operator-specific arguments from dict
                    op_args_raw = {}
                    for key, val in op_args.items():
                        op_name_tmp, var_name = key.split('/')
                        if op_name == op_name_tmp or 'all_ops' in op_name_tmp:
                            op_args_raw[var_name] = val

                    # get tensorflow variables and the variable names from operation_args
                    tf_vars, var_names = parse_dict(var_dict=op_args_raw,
                                                    var_scope=op_name,
                                                    tf_graph=self.tf_graph)

                    # bind tensorflow variables to node and save them in dictionary for the operator class
                    op_args_tf = {}
                    for tf_var, var_name in zip(tf_vars, var_names):
                        node_attr.update({f'{op_name}/{var_name}': tf_var})
                        op_args_tf[var_name] = {'var': tf_var, 'dependency': False}

                    # set input dependencies
                    ########################

                    for var_name, inp in op['inputs'].items():

                        # collect input variable calculation operations
                        out_ops = []
                        out_vars = []
                        out_var_idx = []

                        # go through inputs to variable
                        for i, inp_op in enumerate(inp['sources']):

                            if type(inp_op) is list and len(inp_op) == 1:
                                inp_op = inp_op[0]

                            # collect output variable and operation of the input operator(s)
                            ################################################################

                            # for a single input operator
                            if type(inp_op) is str:

                                # get name and extract index from it if necessary
                                out_name = ops[inp_op]['output']
                                if '[' in out_name:
                                    idx_start, idx_stop = out_name.find('['), out_name.find(']')
                                    out_var_idx.append(out_name[idx_start + 1:idx_stop])
                                    out_var = f"{inp_op}/{out_name[:idx_start]}"
                                else:
                                    out_var_idx.append(None)
                                    out_var = f"{inp_op}/{out_name}"
                                if out_var not in op_args.keys():
                                    raise ValueError(f"Invalid dependencies found in operator: {op['equations']}. Input"
                                                     f" Variable {var_name} has not been calculated yet.")

                                # append variable and operation to list
                                out_ops.append(op_args[out_var]['op'])
                                out_vars.append(op_args[out_var]['var'])

                            # for multiple input operators
                            else:

                                out_vars_tmp = []
                                out_var_idx_tmp = []
                                out_ops_tmp = []

                                for inp_op_tmp in inp_op:

                                    # get name and extract index from it if necessary
                                    out_name = ops[inp_op_tmp]['output']
                                    if '[' in out_name:
                                        idx_start, idx_stop = out_name.find('['), out_name.find(']')
                                        out_var_idx.append(out_name[idx_start + 1:idx_stop])
                                        out_var = f"{inp_op_tmp}/{out_name[:idx_start]}"
                                    else:
                                        out_var_idx_tmp.append(None)
                                        out_var = f"{inp_op_tmp}/{out_name}"
                                    if out_var not in op_args.keys():
                                        raise ValueError(
                                            f"Invalid dependencies found in operator: {op['equations']}. Input"
                                            f" Variable {var_name} has not been calculated yet.")
                                    out_ops_tmp.append(op_args[out_var]['op'])
                                    out_vars_tmp.append(op_args[out_var]['var'])

                                # add tensorflow operations for grouping the inputs together
                                tf_var_tmp = tf.parallel_stack(out_vars_tmp)
                                if inp['reduce_dim'][i]:
                                    tf_var_tmp = tf.reduce_sum(tf_var_tmp, 0)
                                else:
                                    tf_var_tmp = tf.reshape(tf_var_tmp,
                                                            shape=(tf_var_tmp.shape[0] * tf_var_tmp.shape[1],))

                                # append variable and operation to list
                                out_vars.append(tf_var_tmp)
                                out_var_idx.append(out_var_idx_tmp)
                                out_ops.append(tf.group(out_ops_tmp))

                        # add inputs to argument dictionary (in reduced or stacked form)
                        ################################################################

                        # for multiple multiple input operations
                        if len(out_vars) > 1:

                            # find shape of smallest input variable
                            min_shape = min([outvar.shape[0] for outvar in out_vars])

                            # append inpout variables to list and reshape them if necessary
                            out_vars_new = []
                            for out_var in out_vars:

                                shape = out_var.shape[0]

                                if shape > min_shape:
                                    if shape % min_shape != 0:
                                        raise ValueError(f"Shapes of inputs do not match: "
                                                         f"{inp['sources']} cannot be stacked.")
                                    multiplier = shape // min_shape
                                    for j in range(multiplier):
                                        out_vars_new.append(out_var[j * min_shape:(j + 1) * min_shape])
                                else:
                                    out_vars_new.append(out_var)

                            # stack input variables or sum them up
                            tf_var = tf.parallel_stack(out_vars_new)
                            if type(inp['reduce_dim']) is bool and inp['reduce_dim']:
                                tf_var = tf.reduce_sum(tf_var, 0)
                            else:
                                tf_var = tf.reshape(tf_var, shape=(tf_var.shape[0] * tf_var.shape[1],))

                            tf_op = tf.group(out_ops)

                        # for a single input variable
                        else:

                            tf_var = out_vars[0]
                            tf_op = out_ops[0]

                        # add input variable information to argument dictionary
                        op_args_tf[var_name] = {'dependency': True,
                                                'var': tf_var,
                                                'op': tf_op}

                    # create tensorflow operator
                    tf_op_new, op_args_tf = self.add_operator(expressions=op['equations'],
                                                              expression_args=op_args_tf,
                                                              variable_scope=op_name)
                    tf_ops.append(tf_op_new)

                    # bind newly created tf variables to node
                    for var_name, tf_var in op_args_tf.items():
                        if not hasattr(node_attr, var_name):
                            node_attr.update({var_name: tf_var['var']})
                        op_args[f'{op_name}/{var_name}'] = tf_var

                    # handle dependencies
                    op_args[f"{op_name}/{op['output']}"]['op'] = tf_ops[-1]
                    for arg in op_args.values():
                        arg['dependency'] = False

                # group tensorflow versions of all operators to a single 'step' operation
                node_attr.update({'step': tf.group(tf_ops, name=f"{self.key}_step")})

        # call super method
        super().add_node(n, **node_attr)

    def add_edge(self, u: str, v: str,
                 source_op: str,
                 target_var: str,
                 source_to_edge_map: Optional[np.ndarray] = None,
                 edge_to_target_map: Optional[np.ndarray] = None,
                 weight: Optional[Union[float, list, np.ndarray]] = 1.,
                 delay: Optional[Union[float, list, np.ndarray]] = 0.
                 ) -> None:
        """Add edge to the network that connects two variables from a source and a target node.

        Parameters
        ----------
        u
            Name of source node.
        v
            Name of target node.
        source_op
            Name of the output operator on the source node
        target_var
            Name of the target variable on the target node
        weight
            Weighting constant that will be applied to the source output.
        delay
            Time that it takes for the source output to arrive at the target.

        Returns
        -------
        None

        """

        edge_attr = {}

        # extract variables from source and target
        ##########################################

        source, target = self.nodes[u], self.nodes[v]

        # get source variable
        source_var_name = source['operators'][source_op]['output']
        source_var_tf = source['operator_args'][f"{source_op}/{source_var_name}"]

        # get target variable
        target_var_tf = target['operator_args'][target_var]

        # add projection operator of edge to tensorflow graph
        #####################################################

        with self.tf_graph.as_default():

            # set variable scope of edge
            edge_idx = len(self.edges[u, v])
            with tf.variable_scope(f'{u}_{v}_{edge_idx}'):

                # create edge operator dictionary
                source_to_edge_mapping = "" if source_to_edge_map is None else "smap @ "
                edge_to_target_mapping = "" if edge_to_target_map is None else "tmap @ "
                _, target_var_name = target_var.split('/')
                op = {'equations': [f"tvar[d] = {edge_to_target_mapping}"
                                    f"( c * {source_to_edge_mapping} svar)"],
                      'inputs': {},
                      'output': target_var}

                # create edge operator arguments dictionary
                op_args = {'c': {'vtype': 'constant',
                                 'dtype': 'float32',
                                 'shape': np.array(weight).shape,
                                 'value': weight},
                           'd': {'vtype': 'constant',
                                 'dtype': 'float32',
                                 'shape': np.array(delay).shape,
                                 'value': delay},
                           }
                if source_to_edge_map is not None:
                    op_args['smap'] = {'vtype': 'constant_sparse',
                                       'dtype': 'float32',
                                       'shape': source_to_edge_map.shape,
                                       'value': source_to_edge_map}
                if edge_to_target_map is not None:
                    op_args['tmap'] = {'vtype': 'constant_sparse',
                                       'dtype': 'float32',
                                       'shape': edge_to_target_map.shape,
                                       'value': edge_to_target_map}

                # parse into tensorflow graph
                op_args_tf = parse_dict(op_args, self.tf_graph)

                # add source and target variables
                op_args_tf['tvar'] = {'vtype': 'target_var',
                                      'name': target_var_name}
                op_args_tf['svar'] = {'vtype': 'source_var',
                                      'name': source_var_name}

                # add operator to tensorflow graph
                tf_op = self.add_operator(op['equations'], op_args, 'coupling_op')

        # add edge to networkx graph
        ############################

        super().add_edge(u, v, update=tf_op, **edge_attr)

    def add_operator(self, expressions: List[str], expression_args: dict, variable_scope: str
                     ) -> Tuple[Union[tf.Tensor, tf.Variable, tf.Operation], dict]:
        """

        Parameters
        ----------
        expressions
            String-based representations of operator's equations.
        expression_args
            Key-value pairs of operator's variables.
        variable_scope
            Operator name.

        Returns
        -------
        Tuple[Union[tf.Tensor, tf.Variable, tf.Operation], dict]

        """

        # add mathematical evaluations of each operator expression to the tensorflow graph
        ##################################################################################

        evals = []

        with self.tf_graph.as_default():

            # set variable scope
            with tf.variable_scope(variable_scope):

                # go through mathematical equations
                for i, expr in enumerate(expressions):

                    # parse equation
                    update, expression_args = parse_equation(expr, expression_args, tf_graph=self.tf_graph)

                    # store tensorflow operation in list
                    evals.append(update)

                # check which rhs evaluations still have to be assigned to their lhs variable
                evals_complete = []
                evals_uncomplete = []
                for ev in evals:
                    if ev[1] is None:
                        evals_complete.append(ev[0])
                    else:
                        evals_uncomplete.append(ev)

                # group the tensorflow operations across expressions
                with tf.control_dependencies(evals_complete):
                    for ev in evals_uncomplete:
                        evals_complete.append(ev[0].assign(ev[1]))

            return tf.group(evals_complete, name=f'{variable_scope}_eval')

    def vectorize(self, net_config: MultiDiGraph, first_lvl_vec: bool = True, second_lvl_vec: bool=True
                  ) -> MultiDiGraph:
        """Method that goes through the nodes and edges dicts and vectorizes those that are governed by the same
        operators/equations.

        Parameters
        ----------
        net_config
            See argument description at class level.
        first_lvl_vec
            If True, first stage (vectorization over nodes and edges) will be included.
        second_lvl_vec
            If True, second stage (vectorization over operations) will be performed as well.

        Returns
        -------
        Tuple[NodeView, EdgeView]

        """

        # First stage: Vectorize over nodes and edges
        #############################################

        if first_lvl_vec:

            new_net_config = MultiDiGraph()

            # node vectorization
            ####################

            # extract node-type related parts of node keys
            node_keys = []
            for key in net_config.nodes.keys():
                node_type, node_name = key.split('/')
                node_keys.append(node_type)

            # get unique node names
            node_keys_unique = list(set(node_keys))

            # create new node dict with nodes vectorized over multiples of the unique node keys
            node_arg_map = {}
            for new_node in node_keys_unique:

                op_args_new = {}
                n_nodes = 0
                arg_vals = {}
                arg_shape = {}
                op_order_new = []
                ops_new = {}

                # go through old nodes
                for node_name, node_info in net_config.nodes.items():

                    if new_node in node_name:

                        node_arg_map[node_name] = {}

                        # collect information of node operators and arguments
                        #####################################################

                        # use first match to collect operations and operation arguments
                        if len(op_args_new) == 0:

                            # collect operators, their args and their order
                            ops_new.update(deepcopy(node_info['operators']))
                            op_order_new = node_info['op_order']
                            op_args_new.update(deepcopy(node_info['operator_args']))

                            # collect arguments
                            for key, arg in op_args_new.items():

                                if 'value' in arg.keys() and 'shape' in arg.keys():

                                    # extract value and shape of argument
                                    if len(arg['shape']) == 2:
                                        raise ValueError(f"Automatic optimization of the graph (i.e. method `vectorize`"
                                                         f" cannot be applied to networks with variables of 2 or more"
                                                         f" dimensions. Variable {key} has shape {arg['shape']}. Please"
                                                         f" turn of the `vectorize` option.")
                                    if len(arg['shape']) == 1:
                                        if type(arg['value']) is float:
                                            arg['value'] = np.zeros(arg['shape']) + arg['value']
                                        arg_vals[key] = list(arg['value'])
                                        arg_shape[key] = tuple(arg['shape'])
                                    else:
                                        arg_vals[key] = [arg['value']]
                                        arg_shape[key] = (1, )

                                    # save position of old argument value in new, vectorized argument
                                    node_arg_map[node_name][key] = n_nodes

                        # collect shape and value information of other node's arguments
                        else:

                            for key, arg in op_args_new.items():

                                if 'value' in arg.keys() and 'shape' in arg.keys():

                                    # extract value and shape of argument
                                    if len(arg['shape']) == 0:
                                        arg_vals[key].append(arg['value'])
                                    else:
                                        if type(arg['value']) is float:
                                            arg['value'] = list(np.zeros(arg['shape']) + arg['value'])
                                        arg_vals[key] += arg['value']
                                        if arg['shape'][0] > arg_shape[key][0]:
                                            arg_shape[key] = tuple(arg['shape'])

                                    # save position of old argument value in new, vectorized argument
                                    node_arg_map[node_name][key] = n_nodes

                        # increment node counter
                        n_nodes += 1

                # go through new arguments and update shape and values
                for key, arg in op_args_new.items():

                    if 'value' in arg.keys():
                        arg.update({'value': arg_vals[key]})

                    if 'shape' in arg.keys():
                        arg.update({'shape': (n_nodes, ) + arg_shape[key]})

                # add new, vectorized node to dictionary
                if n_nodes > 0:
                    new_net_config.add_node(new_node,
                                            operators=ops_new,
                                            op_order=op_order_new,
                                            operator_args=op_args_new,
                                            inputs={})

            # edge vectorization
            ####################

            net_config = self.vectorize_edges(net_config=new_net_config,
                                              net_config_old=net_config,
                                              node_arg_map=node_arg_map)

        # Second stage: Vectorize over operators
        ########################################

        if second_lvl_vec:

            new_net_config = MultiDiGraph()
            new_net_config.add_node('net', operators={}, op_order=[], operator_args={}, inputs={})
            node_arg_map = {}

            # vectorize node operators
            ###########################

            # collect all operation keys, arguments and dependencies of each node
            op_info = {'keys': [], 'args': [], 'vectorized': [], 'deps': []}
            for node in net_config.nodes.values():
                op_info['keys'].append(node['op_order'])
                op_info['args'].append(list(node['operator_args'].keys()))
                op_info['vectorized'].append([False] * len(op_info['keys'][-1]))
                node_deps = []
                for op_name in node['op_order']:
                    op = node['operators'][op_name]
                    op_deps = []
                    for _, inputs in op['inputs'].items():
                        op_deps += inputs['sources']
                    node_deps.append(op_deps)
                op_info['deps'].append(node_deps)

            # go through nodes and vectorize over their operators
            node_idx = 0
            node_key = list(net_config.nodes.keys())[node_idx]
            while not all([all(vec) for vec in op_info['vectorized']]):

                node_changed = False

                for op_key in op_info['keys'][node_idx]:

                    op_idx = op_info['keys'][node_idx].index(op_key)

                    # check if operation needs to be vectorized
                    if not op_info['vectorized'][node_idx][op_idx]:

                        # check if dependencies are vectorized already
                        deps_vec = True
                        for dep in op_info['deps'][node_idx][op_idx]:
                            if not op_info['vectorized'][node_idx][op_info['keys'][node_idx].index(dep)]:
                                deps_vec = False

                        if deps_vec:

                            # check whether operation exists at other nodes
                            nodes_to_vec = []
                            op_indices = []
                            for node_idx_tmp, node_key_tmp in enumerate(net_config.nodes.keys()):

                                if node_key_tmp != node_key and op_key in op_info['keys'][node_idx_tmp]:

                                    op_idx_tmp = op_info['keys'][node_idx_tmp].index(op_key)

                                    # check if dependencies are vectorized already
                                    deps_vec = True
                                    for dep in op_info['deps'][node_idx_tmp][op_idx_tmp]:
                                        if not op_info['vectorized'][node_idx_tmp][op_info['keys']
                                                                                          [node_idx_tmp].index(dep)]:
                                            deps_vec = False

                                    if deps_vec:
                                        nodes_to_vec.append((node_key_tmp, node_idx_tmp))
                                        op_indices.append(op_idx_tmp)
                                    else:
                                        node_idx = node_idx_tmp
                                        node_key = node_key_tmp
                                        node_changed = True
                                        break

                            if nodes_to_vec and not node_changed:

                                nodes_to_vec.append((node_key, node_idx))
                                op_indices.append(op_idx)

                                # vectorize op
                                node_arg_map = self.vectorize_ops(new_node=new_net_config.nodes['net'],
                                                                  net_config=net_config,
                                                                  op_key=op_key,
                                                                  nodes=nodes_to_vec,
                                                                  node_arg_map=node_arg_map)

                                # indicate where vectorization was performed
                                for (node_key_tmp, node_idx_tmp), op_idx_tmp in zip(nodes_to_vec, op_indices):
                                    op_info['vectorized'][node_idx_tmp][op_idx_tmp] = True

                            elif node_changed:

                                break

                            else:

                                # add operation to new net configuration
                                node_arg_map = self.vectorize_ops(new_node=new_net_config.nodes['net'],
                                                                  net_config=net_config,
                                                                  op_key=op_key,
                                                                  nodes=[(node_key, 0)],
                                                                  node_arg_map=node_arg_map)

                                # mark operation on node as checked
                                op_info['vectorized'][node_idx][op_idx] = True

                # increment node
                if not node_changed:
                    node_idx = node_idx + 1 if node_idx < len(op_info['keys']) - 1 else 0
                    node_key = list(net_config.nodes.keys())[node_idx]

            # vectorize edge operators
            ##########################

            net_config = self.vectorize_edges(net_config=new_net_config,
                                              net_config_old=net_config,
                                              node_arg_map=node_arg_map)

        # Third Stage: Vectorize over multiple edges that map to the same input variable
        ################################################################################

        # 1. go through edges and extract target node input variables
        for source, target, edge in net_config.edges:

            # extract edge data
            edge_data = net_config.edges[source, target, edge]
            op_name = edge_data['op_order'][-1]
            op = edge_data['operators'][op_name]
            var_name = op['output']
            target_op_name = edge_data['operator_args'][f'{op_name}/{var_name}']['name']
            delay = op['delay']

            # find equation that maps to target variable
            for i, eq in enumerate(op['equations']):
                lhs, _ = eq.split(' = ')
                if var_name in lhs:
                    break

            # add output variable to inputs field of target node
            if target_op_name not in net_config.nodes[target]['inputs']:
                net_config.nodes[target]['inputs'][target_op_name] = {
                    'sources': [f'{source}/{target}/{edge}/{op_name}/{i}'],
                    'delays': [delay],
                    'reduce_dim': True}
            else:
                net_config.nodes[target]['inputs'][target_op_name]['sources'].append(
                    f'{source}/{target}/{edge}/{op_name}/{i}')
                net_config.nodes[target]['inputs'][target_op_name]['delays'].append(delay)

        # 2. go through nodes and create correct mapping for multiple inputs to same variable
        for node_name, node in net_config.nodes.items():

            # loop over input variables of node
            for i, (in_var, inputs) in enumerate(node['inputs'].items()):

                op_name, in_var_name = in_var.split('/')
                if len(inputs) > 1:

                    # get shape of output variables and define equation to be used for mapping
                    if '[' in in_var:

                        # extract index from variable name
                        idx_start, idx_stop = in_var_name.find('['), in_var_name.find(']')
                        in_var_name_tmp = in_var_name[0:idx_start]
                        idx = in_var_name[idx_start+1:idx_stop]

                        # define mapping equation
                        map_eq = f"{in_var_name} = sum(in_col_{i}, 1)"

                        # get shape of output variable
                        if ':' in idx:
                            idx_1, idx_2 = idx.split(':')
                            out_shape = int(idx_2) - int(idx_1)
                        else:
                            out_shape = ()

                        # add input buffer rotation to end of operation list
                        if ',' in idx:
                            eqs = [f"buffer_var_{i} = {in_var_name_tmp}[1:,:]",
                                   f"{in_var_name_tmp}[0:-1,:] = buffer_var_{i}",
                                   f"{in_var_name_tmp}[-1,:] = 0."]
                        else:
                            eqs = [f"buffer_var_{i} = {in_var_name_tmp}[1:]",
                                   f"{in_var_name_tmp}[0:-1] = buffer_var_{i}",
                                   f"{in_var_name_tmp}[-1] = 0."]
                        node['operators'][f'input_buffer_{i}'] = {
                            'equations': eqs,
                            'inputs': {in_var_name_tmp: {'sources': [f'{op_name}/{in_var_name_tmp}'],
                                                         'reduce_dim': True}},
                            'output': in_var_name_tmp}
                        node['op_order'].append(f'input_buffer_{i}')

                    else:

                        # get shape of output variable
                        in_var_name_tmp = in_var_name
                        out_shape = node['operator_args'][f'{op_name}/{in_var_name_tmp}']['shape']

                        # define mapping equation
                        if len(out_shape) < 2:
                            map_eq = f"{in_var_name_tmp} = sum(in_col_{i}, 1)"
                        elif out_shape[0] == 1:
                            map_eq = f"{in_var_name_tmp}[0,:] = sum(in_col_{i}, 1)"
                        elif out_shape[1] == 1:
                            map_eq = f"{in_var_name_tmp}[:,0] = sum(in_col_{i}, 1)"
                        else:
                            map_eq = f"{in_var_name_tmp} = squeeze(sum(in_col_{i}, 1))"

                    out_shape = None if len(out_shape) == 0 else out_shape[0]

                    # add new operator and its args to node
                    node['operators'][f'input_handling_{i}'] = {
                        'equations': [map_eq],
                        'inputs': {},
                        'output': in_var_name_tmp}
                    node['op_order'] = [f'input_handling_{i}'] + node['op_order']
                    node['operators'][op_name]['inputs'][in_var_name_tmp] = {
                        'sources': [f'input_handling_{i}'], 'reduce_dim': True}
                    node['operator_args'][f'input_handling_{i}/in_col_{i}'] = {
                        'name': f'in_col_{i}',
                        'shape': (out_shape, len(inputs)) if out_shape else (1, len(inputs)),
                        'dtype': 'float32',
                        'vtype': 'state_var',
                        'value': 0.}
                    node['operator_args'][f'input_handling_{i}/{in_var_name_tmp}'] = \
                        node['operator_args'][in_var].copy()
                    node['operator_args'].pop(in_var)

                    # adjust edges accordingly
                    for j, inp in enumerate(inputs['sources']):
                        source, target, edge, op, eq_idx = inp.split('/')
                        edge_data = net_config.edges[source, target, edge]
                        eq = edge_data['operators'][op]['equations'][int(eq_idx)]
                        lhs, rhs = eq.split(' = ')
                        edge_data['operators'][op]['equations'][int(eq_idx)] = \
                            eq.replace(lhs, lhs.replace(in_var_name, f'in_col_{i}[:,{j}:{j+1}]'))
                        edge_data['operators'][op]['output'] = f'input_handling_{i}/{in_var_name_tmp}'
                        edge_data['operator_args'][f'{op}/in_col_{i}'] = {'vtype': 'target_var',
                                                                          'name': f'input_handling_{i}/in_col_{i}'}
                        edge_data['operator_args'].pop(f'{op}/{in_var_name_tmp}')

        # Final Stage: Vectorize operator inputs correctly
        ##################################################

        # go through nodes
        # for node_name, node in net_config.nodes.items():
        #
        #     # go through node's operators
        #     for op_name in node['op_order']:
        #
        #         op = node['operators'][op_name]
        #
        #         # go through operator's inputs
        #         for var_name, inputs in op['inputs'].items():
        #
        #             if len(inputs['sources']) > 1:
        #
        #                 # extract shapes
        #                 arg_name = f'{op_name}/{var_name}'
        #                 source_shapes = []
        #                 for source in inputs['sources']:
        #                     source_var = f"{source}/{node['operators'][source]['output']}"
        #                     source_shapes.append(node['operator_args'][source_var]['shape'])
        #
        #                 # map source variables to target variable
        #                 #########################################
        #
        #                 if inputs['reduce_dim']:
        #
        #                     if not len(list(set(source_shapes))) == 1:
        #
        #                         # add indices from node_arg_map to operator outputs
        #                         for source in inputs['sources']:
        #                             idx = node_arg_map[node_name][arg_name]
        #                             if type(idx) is tuple:
        #                                 idx = f'{idx[0]}:{idx[1]}'
        #                             node['operators'][source]['output'] += f"[{idx}]"

        return net_config

    def vectorize_edges(self,
                        net_config: MultiDiGraph,
                        net_config_old: MultiDiGraph,
                        node_arg_map: dict
                        ) -> MultiDiGraph:
        """Vectorizes edges according to node structure of net.

        Parameters
        ----------
        net_config
        net_config_old
        node_arg_map

        Returns
        -------
        MultiDiGraph

        """

        # extract edge-type related parts of edge keys
        ##############################################

        edge_keys = {}

        # go through edges
        for source, target, key in net_config_old.edges:

            # extract dege type from key
            if '/' in key:
                edge_type, edge_name = key.split('/')
            else:
                edge_type = key

            # check whether network contains one or multiple nodes and extract source + target node
            if len(net_config.nodes) == 1:
                s, t = list(net_config.nodes.keys())[0], list(net_config.nodes.keys())[0]
            else:
                s = source.split('/')[0]
                t = target.split('/')[0]

            # add edge type information to dictionary
            if s in edge_keys.keys():
                if t in edge_keys[s].keys():
                    edge_keys[s][t].append(edge_type)
                else:
                    edge_keys[s][t] = [edge_type]
            else:
                edge_keys[s] = {t: [edge_type]}

        # get unique edge names
        for s, sval in edge_keys.items():
            for t, tval in sval.items():
                edge_keys[s][t] = list(set(tval))

        # create new edge dict with edges vectorized over multiples of the unique edge keys
        ###################################################################################

        # go through each edge of each pair of nodes
        for s, sval in edge_keys.items():
            for t, tval in sval.items():
                for e in tval:

                    op_args_new = {}
                    n_edges = 0
                    arg_vals = {}
                    arg_shape = {}
                    snode_idx = {}
                    tnode_idx = {}
                    ops_new = {}
                    op_order_new = []
                    out_var_shape = 1
                    edge_delays = []

                    # go through old edges
                    for source, target, edge in net_config_old.edges:

                        edge_info = net_config_old.edges[source, target, edge]

                        # check if source, target and edge of old net config belong to the to-be created edge
                        if (s in source and t in target and e in edge) or (s == t and e in edge):

                            if n_edges == 0:

                                # use first match to collect operations and operation arguments
                                ###############################################################

                                # collect operators, their arguments and their order
                                ops_new.update(deepcopy(edge_info['operators']))
                                op_order_new = edge_info['op_order']
                                op_args_new.update(deepcopy(edge_info['operator_args']))

                                # go through operator arguments
                                for key, arg in op_args_new.items():

                                    # set shape and value of argument
                                    if 'value' in arg.keys() and 'shape' in arg.keys():
                                        if len(arg['shape']) == 2:
                                            arg_shape[key] = list(arg['shape'])
                                            arg_vals[key] = list(arg['value'])
                                        elif len(arg['shape']) == 1:
                                            if type(arg['value']) is float:
                                                arg_vals[key] = list(np.zeros(arg['shape']) + arg['value'])
                                            else:
                                                arg_vals[key] = list(arg['value'])
                                            arg_shape[key] = list(arg['shape'])
                                        else:
                                            arg_vals[key] = [arg['value']]
                                            arg_shape[key] = [1]

                                    # collect indices and shapes of source and target variables
                                    if arg['vtype'] == 'source_var':
                                        snode_idx[key] = ([node_arg_map[source][arg['name']]],
                                                          net_config.nodes[s]['operator_args'][arg['name']]['shape'])
                                    elif arg['vtype'] == 'target_var':
                                        tnode_idx[key] = ([node_arg_map[target][arg['name']]],
                                                          net_config.nodes[t]['operator_args'][arg['name']]['shape'])

                                    # check whether output shape needs to be updated
                                    if 'all_ops/map_' in key:
                                        out_var_shape = op_args_new[key]['shape'][0]
                                    elif 'all_ops/tmap' in key:
                                        out_var_shape = op_args_new[key]['shape'][1]

                            else:

                                # add argument values to dictionary
                                for key, arg in op_args_new.items():

                                    # add value and shape information
                                    if 'value' in arg.keys() and 'shape' in arg.keys():
                                        if len(arg['shape']) == 0:
                                            arg_vals[key].append(arg['value'])
                                            arg_shape[key][0] += 1
                                        else:
                                            if type(arg['value']) is float:
                                                val = list(np.zeros(arg['shape']) + arg['value'])
                                            else:
                                                val = list(arg['value'])
                                            arg_vals[key] += val
                                            arg_shape[key][0] += len(val)
                                    if arg['vtype'] == 'source_var':
                                        snode_idx[key][0].append(node_arg_map[source][arg['name']])
                                    elif arg['vtype'] == 'target_var':
                                        tnode_idx[key][0].append(node_arg_map[target][arg['name']])

                            # extract edge delay
                            for op in edge_info['operators'].values():
                                if 'delay' in op.keys():
                                    if type(op['delay']) is float:
                                        edge_delays.append(int(op['delay'] / self.dt))
                                    else:
                                        edge_delays.append(op['delay'])
                                    break

                            # increment edge counter
                            n_edges += 1

                    # create mapping from source to edge space and edge to target space
                    ###################################################################

                    if n_edges > 0:

                        # go through source variables
                        for n, (key, val) in enumerate(snode_idx.items()):

                            edge_map = np.zeros((out_var_shape * n_edges, val[1][0]))

                            # set indices of source variables in mapping
                            for i, idx in enumerate(val[0]):
                                if type(idx) is tuple:
                                    for idx2 in range(idx[0], idx[1]):
                                        edge_map[i, idx2] = 1.
                                        i += 1
                                else:
                                    edge_map[i, idx] = 1.

                            # add edge mapping to edge operator equation and operator arguments
                            if not n_edges == val[1] or not edge_map == np.eye(n_edges):

                                for op_key in op_order_new:

                                    op = ops_new[op_key]
                                    _, var_name = key.split('/')

                                    for i, eq in enumerate(op['equations']):

                                        if f'map_{var_name}' not in eq:

                                            # insert mapping into equation
                                            eq = eq.replace(var_name, f'(map_{var_name} @ {var_name})')
                                            ops_new[op_key]['equations'][i] = eq

                                # add mapping to operator arguments
                                op_args_new[f'all_ops/map_{var_name}'] = {'vtype': 'constant',
                                                                          'dtype': 'float32',
                                                                          'name': f'map_{var_name}',
                                                                          'shape': edge_map.shape,
                                                                          'value': edge_map}

                        # go through target variables
                        for key, val in tnode_idx.items():

                            target_map = np.zeros((val[1][0], out_var_shape * n_edges))

                            # set indices of target variables in mapping
                            for i, idx in enumerate(val[0]):
                                if type(idx) is tuple:
                                    for idx2 in range(idx[0], idx[1]):
                                        target_map[idx2, i] = 1.
                                        i += 1
                                else:
                                    target_map[idx, i] = 1.

                            # add target mapping to edge operator equation and operator arguments
                            if not n_edges == val[1] or not target_map == np.eye(n_edges):

                                for op_key in op_order_new:

                                    op = ops_new[op_key]

                                    for i, eq in enumerate(op['equations']):

                                        _, rhs = eq.split(' = ')

                                        if 'tmap' not in rhs:

                                            # insert mapping into equation
                                            eq = eq.replace(rhs, f'tmap @ ({rhs})')
                                            ops_new[op_key]['equations'][i] = eq

                                # add mapping to operator arguments
                                op_args_new['all_ops/tmap'] = {'vtype': 'constant',
                                                               'dtype': 'float32',
                                                               'name': 'tmap',
                                                               'shape': target_map.shape,
                                                               'value': target_map}

                    # TODO: vectorize delays of old edges
                    ops_new[op_order_new[-1]]['delay'] = 0.

                    # go through new arguments and change shape and values
                    for key, arg in op_args_new.items():

                        if 'map_' not in key and 'tmap' not in key:

                            if 'value' in arg.keys():
                                arg.update({'value': arg_vals[key]})

                            if 'shape' in arg.keys():
                                arg.update({'shape': tuple(arg_shape[key])})

                    if n_edges > 0:

                        # add new, vectorized edge to networkx graph
                        net_config.add_edge(s, t, e,
                                            operators=ops_new,
                                            operator_args=op_args_new,
                                            op_order=op_order_new)

        return net_config

    def vectorize_ops(self,
                      new_node: dict,
                      net_config: MultiDiGraph,
                      op_key: str,
                      nodes: list,
                      node_arg_map: dict
                      ) -> dict:
        """Vectorize all instances of an operation across nodes and put them into a single-node network.

        Parameters
        ----------
        new_node
        net_config
        op_key
        nodes
        node_arg_map

        Returns
        -------
        dict

        """

        # extract operation in question
        node_name_tmp = nodes[0][0]
        ref_node = net_config.nodes[node_name_tmp]
        op = ref_node['operators'][op_key]
        n_nodes = len(nodes)

        # collect input dependencies of all nodes
        #########################################

        for node, _ in nodes:
            for key, arg in net_config.nodes[node]['operators'][op_key]['inputs'].items():
                if type(op['inputs'][key]['reduce_dim']) is bool:
                    op['inputs'][key]['reduce_dim'] = [op['inputs'][key]['reduce_dim']]
                    op['inputs'][key]['sources'] = [op['inputs'][key]['sources']]
                else:
                    if arg['sources'] not in op['inputs'][key]['sources']:
                        op['inputs'][key]['sources'].append(arg['sources'])
                        op['inputs'][key]['reduce_dim'].append(arg['reduce_dim'])

        # extract operator-specific arguments and change their shape and value
        ######################################################################

        op_args = {}
        for key, arg in ref_node['operator_args'].items():

            arg = arg.copy()

            if op_key in key:

                # go through nodes and extract shape and value information for arg
                arg['value'] = np.array(arg['value'])
                for node_name, node_idx in nodes:

                    if node_name not in node_arg_map.keys():
                        node_arg_map[node_name] = {}

                    # extract node arg value and shape
                    arg_tmp = net_config.nodes[node_name_tmp]['operator_args'][key]
                    val = arg_tmp['value']
                    dims = tuple(arg_tmp['shape'])

                    # append value to argument dictionary
                    idx = (0, len(val))
                    if node_name != node_name_tmp:
                        if len(dims) > 0 and type(val) is float:
                            val = np.zeros(dims) + val
                        else:
                            val = np.array(val)
                        idx_old = 0 if len(dims) == 0 else arg['value'].shape[0]
                        arg['value'] = np.append(arg['value'], val, axis=0)
                        idx_new = arg['value'].shape[0]
                        idx = (idx_old, idx_new)

                    # add information to node_arg_map
                    node_arg_map[node_name][key] = idx

                    # change shape of argument
                    arg['shape'] = arg['value'].shape

                op_args[key] = arg

        # add operator information to new node
        ######################################

        # add operator
        new_node['operators'][op_key] = op
        new_node['op_order'].append(op_key)

        # add operator args
        for key, arg in op_args.items():
            new_node['operator_args'][key] = arg

        return node_arg_map

    def run(self, simulation_time: Optional[float] = None, inputs: Optional[dict] = None,
            outputs: Optional[dict] = None, sampling_step_size: Optional[float] = None) -> Tuple[DataFrame, float]:
        """Simulate the network behavior over time via a tensorflow session.

        Parameters
        ----------
        simulation_time
        inputs
        outputs
        sampling_step_size

        Returns
        -------
        list

        """

        # prepare simulation
        ####################

        # basic simulation parameters initialization
        if simulation_time:
            sim_steps = int(simulation_time / self.dt)
        else:
            sim_steps = 1

        if not sampling_step_size:
            sampling_step_size = self.dt
        sampling_steps = int(sampling_step_size / self.dt)

        # define output variables
        if not outputs:
            outputs = dict()
            for name, node in self.nodes.items():
                outputs[name + '_v'] = getattr(node['handle'], 'name')

        # linearize input dictionary
        if inputs:
            inp = list()
            for step in range(sim_steps):
                inp_dict = dict()
                for key, val in inputs.items():
                    shape = key.shape
                    if len(val) > 1:
                        inp_dict[key] = np.reshape(val[step], shape)
                    else:
                        inp_dict[key] = np.reshape(val, shape)
                inp.append(inp_dict)
        else:
            inp = [None for _ in range(sim_steps)]

        # run simulation
        ################

        time = 0.
        times = []
        with tf.Session(graph=self.tf_graph) as sess:

            # writer = tf.summary.FileWriter('/tmp/log/', graph=self.tf_graph)

            # initialize all variables
            sess.run(tf.global_variables_initializer())

            results = []
            t_start = t.time()

            # simulate network behavior for each time-step
            for step in range(sim_steps):

                sess.run(self.step, inp[step])

                # save simulation results
                if step % sampling_steps == 0:
                    results.append([np.squeeze(var.eval()) for var in outputs.values()])
                    time += sampling_step_size
                    times.append(time)

            t_end = t.time()

            # display simulation time
            if simulation_time:
                print(f"{simulation_time}s of network behavior were simulated in {t_end - t_start} s given a "
                      f"simulation resolution of {self.dt} s.")
            else:
                print(f"Network computations finished after {t_end - t_start} seconds.")
            # writer.close()

            # store results in pandas dataframe
            results = np.array(results)
            if len(results.shape) > 2:
                n_steps, n_vars, n_nodes = results.shape
            else:
                n_steps, n_vars = results.shape
                n_nodes = 1
            results = np.reshape(results, (n_steps, n_nodes * n_vars))
            columns = []
            for var in outputs.keys():
                columns += [f"{var}_{i}" for i in range(n_nodes)]
            results_final = DataFrame(data=results, columns=columns, index=times)

        return results_final, (t_end - t_start)
