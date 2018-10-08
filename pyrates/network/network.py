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
    node_arg_map
        Used internally for vectorization.

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
        self.node_arg_map = {}

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
                    self.add_node(node=node_name,
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
                        self.add_edge(source_node=source_name,
                                      target_node=target_name,
                                      **edge_info)

                        # collect project operation of edge
                        edge_updates.append(self.edges[source_name, target_name, edge_idx]['update'])

                    # group project operations of all edges
                    if len(edge_updates) > 0:
                        self.step = tf.tuple(edge_updates, name='network_update')

    def add_node(self, node: str, ops: dict, op_args: dict, op_order: list) -> None:
        """

        Parameters
        ----------
        node
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

            # initialize variable scope of node
            with tf.variable_scope(node):

                tf_ops = []

                # go through operations
                for op_name in op_order:

                    # intialize variable scope of operator
                    with tf.variable_scope(op_name):

                        op = ops[op_name]

                        # extract operator-specific arguments from dict
                        op_args_raw = {}
                        for key, val in op_args.items():
                            op_name_tmp, var_name = key.split('/')
                            if op_name == op_name_tmp or 'all_ops' in op_name_tmp:
                                op_args_raw[var_name] = val

                        # get tensorflow variables and the variable names from operation_args
                        op_args_tf = parse_dict(var_dict=op_args_raw,
                                                tf_graph=self.tf_graph)

                        # bind tensorflow variables to node
                        for var_name, var in op_args_tf.items():
                            node_attr.update({f'{op_name}/{var_name}': var})
                            op_args_tf[var_name] = {'var': var, 'dependency': False}

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
                                        raise ValueError(f"Invalid dependencies found in operator: {op['equations']}. "
                                                         f"Input Variable {var_name} has not been calculated yet.")

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
                                min_shape = min([outvar.shape[0] if len(outvar.shape) > 0 else 0 for outvar in out_vars])

                                # append inpout variables to list and reshape them if necessary
                                out_vars_new = []
                                for out_var in out_vars:

                                    shape = out_var.shape[0] if len(out_var.shape) > 0 else 0

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
                            attr_name = f"{op_name}/{var_name}"
                            if attr_name not in node_attr.keys():
                                node_attr[attr_name] = tf_var['var']
                            op_args[attr_name] = tf_var

                        # handle dependencies
                        op_args[f"{op_name}/{op['output']}"]['op'] = tf_ops[-1]
                        for arg in op_args.values():
                            arg['dependency'] = False

                # group tensorflow versions of all operators to a single 'step' operation
                node_attr['update'] = tf.group(tf_ops, name=f"{self.key}_update")

        # call super method
        super().add_node(node, **node_attr)

    def add_edge(self,
                 source_node: str,
                 target_node: str,
                 source_var: str,
                 target_var: str,
                 source_to_edge_map: Optional[np.ndarray] = None,
                 edge_to_target_map: Optional[np.ndarray] = None,
                 weight: Optional[Union[float, list, np.ndarray]] = 1.,
                 delay: Optional[Union[float, list, np.ndarray]] = 0.
                 ) -> None:
        """Add edge to the network that connects two variables from a source and a target node.

        Parameters
        ----------
        source_node
            Name of source node.
        target_node
            Name of target node.
        source_var
            Name of the output variable on the source node (including the operator name).
        target_var
            Name of the target variable on the target node (including the operator name).
        weight
            Weighting constant that will be applied to the source output.
        delay
            Time that it takes for the source output to arrive at the target.

        Returns
        -------
        None

        """

        # extract variables from source and target
        ##########################################

        source, target = self.nodes[source_node], self.nodes[target_node]

        # get source and target variables
        source_var_tf = source[source_var]
        target_var_tf = target[target_var]

        # add projection operation of edge to tensorflow graph
        ######################################################

        with self.tf_graph.as_default():

            # set variable scope of edge
            edge_idx = self.number_of_edges(source_node, target_node)
            with tf.variable_scope(f'{source_node}_{target_node}_{edge_idx}'):

                # create edge operator dictionary
                source_to_edge_mapping = "" if source_to_edge_map is None else "smap @ "
                edge_to_target_mapping = "" if edge_to_target_map is None else "tmap @ "
                if edge_to_target_map is not None and len(target_var_tf.shape) == 1:
                    idx = "[:,0]"
                else:
                    idx = ""
                weight = np.array(weight)
                delay = np.array(delay) if delay else np.array([0])
                _, target_var_name = target_var.split('/')
                delay_idx = "" if np.sum(delay) == 0 else "[d]"
                op = {'equations': [f"tvar{delay_idx} = ({edge_to_target_mapping}"
                                    f"( c * ({source_to_edge_mapping} svar))){idx}"],
                      'inputs': {},
                      'output': target_var}

                # create edge operator arguments dictionary
                op_args = {'c': {'vtype': 'constant',
                                 'dtype': 'float32',
                                 'shape': weight.shape if len(weight.shape) == 2 else weight.shape + (1,),
                                 'value': weight}
                           }
                if np.sum(delay) > 0:
                    op_args['d'] = {'vtype': 'constant',
                                    'dtype': 'int32',
                                    'shape': delay.shape,
                                    'value': delay}
                if source_to_edge_map is not None:
                    op_args['smap'] = {'vtype': 'constant_sparse',
                                       'dtype': 'float32',
                                       'shape': source_to_edge_map.shape,
                                       'value': np.array(source_to_edge_map, dtype=np.float32)}
                if edge_to_target_map is not None:
                    op_args['tmap'] = {'vtype': 'constant_sparse',
                                       'dtype': 'float32',
                                       'shape': edge_to_target_map.shape,
                                       'value': np.array(edge_to_target_map, dtype=np.float32)}

                # parse into tensorflow graph
                op_args_tf = parse_dict(op_args, self.tf_graph)

                # add source and target variables
                op_args_tf['tvar'] = target_var_tf
                op_args_tf['svar'] = source_var_tf

                # bring tensorflow variables in parser format
                for var_name, var in op_args_tf.items():
                    op_args_tf[var_name] = {'var': var, 'dependency': False}

                # add operator to tensorflow graph
                tf_op, op_args_tf = self.add_operator(op['equations'], op_args_tf, 'coupling_op')

        # add edge to networkx graph
        ############################

        super().add_edge(source_node, target_node, update=tf_op, **op_args_tf)

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

            return tf.group(evals_complete, name=f'{variable_scope}_eval'), expression_args

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

        # First stage: Vectorize over nodes
        ###################################

        if first_lvl_vec:

            new_net_config = MultiDiGraph()

            # extract operations existing on each node
            node_ops = []
            node_names = []
            for node_name, node in net_config.nodes.items():
                node_ops.append(set(node['op_order']))
                node_names.append(node_name)

            # get unique node names
            node_ops_unique = []
            for node_op in node_ops:
                if node_op not in node_ops_unique:
                    node_ops_unique.append(node_op)

            # create new nodes that combine all original nodes that have the same operators
            for node_op_list in node_ops_unique:

                # collect nodes with a specific set of operators
                nodes = []
                i = 0
                while i < len(node_ops):
                    node_op_list_tmp = node_ops[i]
                    if node_op_list_tmp == node_op_list:
                        nodes.append(node_names.pop(i))
                        node_ops.pop(i)
                    else:
                        i += 1

                # vectorize over nodes
                new_net_config = self.vectorize_nodes(new_net_config=new_net_config,
                                                      old_net_config=net_config,
                                                      nodes=nodes)

            # adjust edges accordingly
            ##########################

            # go through new nodes
            for source_new in new_net_config.nodes.keys():
                for target_new in new_net_config.nodes.keys():

                    # collect old edges that connect the same source and target variables
                    edge_col = {}
                    for source, target, edge in net_config.edges:
                        if source.split('/')[0] in source_new and target.split('/')[0] in target_new:
                            edge_tmp = net_config.edges[source, target, edge]
                            source_var, target_var = edge_tmp['source_var'], edge_tmp['target_var']
                            key = f"{source_var}{target_var}"
                            if key in edge_col.keys():
                                edge_col[key].append((source, target, edge))
                            else:
                                edge_col[key] = [(source, target, edge)]

                    # vectorize over edges
                    for edges in edge_col.values():
                        new_net_config = self.vectorize_edges(edges=edges,
                                                              source=source_new,
                                                              target=target_new,
                                                              new_net_config=new_net_config,
                                                              old_net_config=net_config)

            net_config = new_net_config.copy()
            new_net_config.clear()

        # Second stage: Vectorize over operators
        ########################################

        if second_lvl_vec:

            new_net_config = MultiDiGraph()
            new_net_config.add_node('net', operators={}, op_order=[], operator_args={}, inputs={})

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
                            node_indices = []
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
                                        nodes_to_vec.append(node_key_tmp)
                                        op_indices.append(op_idx_tmp)
                                        node_indices.append(node_idx_tmp)
                                    else:
                                        node_idx = node_idx_tmp
                                        node_key = node_key_tmp
                                        node_changed = True
                                        break

                            if nodes_to_vec and not node_changed:

                                nodes_to_vec.append(node_key)
                                op_indices.append(op_idx)
                                node_indices.append(node_idx)

                                # vectorize op
                                self.vectorize_ops(new_node=new_net_config.nodes['net'],
                                                   net_config=net_config,
                                                   op_key=op_key,
                                                   nodes=nodes_to_vec.copy())

                                # indicate where vectorization was performed
                                for node_key_tmp, node_idx_tmp, op_idx_tmp in \
                                        zip(nodes_to_vec, node_indices, op_indices):
                                    op_info['vectorized'][node_idx_tmp][op_idx_tmp] = True

                            elif node_changed:

                                break

                            else:

                                # add operation to new net configuration
                                self.vectorize_ops(new_node=new_net_config.nodes['net'],
                                                   net_config=net_config,
                                                   op_key=op_key,
                                                   nodes=[node_key])

                                # mark operation on node as checked
                                op_info['vectorized'][node_idx][op_idx] = True

                # increment node
                if not node_changed:
                    node_idx = node_idx + 1 if node_idx < len(op_info['keys']) - 1 else 0
                    node_key = list(net_config.nodes.keys())[node_idx]

            # adjust edges accordingly
            ##########################

            # set source and target
            source_new, target_new = 'net', 'net'

            # collect old edges that connect the same source and target variables
            edge_col = {}
            for source, target, edge in net_config.edges:
                edge_tmp = net_config.edges[source, target, edge]
                source_var, target_var = edge_tmp['source_var'], edge_tmp['target_var']
                key = f"{source_var}{target_var}"
                if key in edge_col.keys():
                    edge_col[key].append((source, target, edge))
                else:
                    edge_col[key] = [(source, target, edge)]

            # vectorize over edges
            for edges in edge_col.values():
                new_net_config = self.vectorize_edges(edges=edges,
                                                      source=source_new,
                                                      target=target_new,
                                                      new_net_config=new_net_config,
                                                      old_net_config=net_config)

            net_config = new_net_config.copy()
            new_net_config.clear()

        # Third Stage: Finalize edge vectorization
        ##########################################

        # 1. Go through edges and create mappings between source, edge and target variables
        for source_name, target_name, edge_name in net_config.edges:

            # extract edge information
            edge = net_config.edges[source_name, target_name, edge_name]
            source_var = net_config.nodes[source_name]['operator_args'][edge['source_var']]
            target_var = net_config.nodes[target_name]['operator_args'][edge['target_var']]
            n_sources = source_var['shape'][0]
            n_targets = target_var['shape'][0]
            n_edges = edge['edge_idx'][-1][1]

            # create mapping from source to edge variables
            ##############################################

            # create mapping from lower dimensional source to higher dimensional edge space
            smap = np.zeros((n_edges, n_sources))
            for edge_idx, source_idx in zip(edge['edge_idx'], edge['source_idx']):
                if type(source_idx) is tuple:
                    for e_idx, s_idx in zip(range(edge_idx[0], edge_idx[1]), range(source_idx[0], source_idx[1])):
                        smap[e_idx, s_idx] = 1.
                elif type(source_idx) is list:
                    for e_idx, s_idx in zip(range(edge_idx[0], edge_idx[1]), source_idx):
                        smap[e_idx, s_idx] = 1.
                else:
                    smap[edge_idx[0]:edge_idx[1], source_idx] = 1.

            # add mapping to edge attributs
            if n_edges != n_sources or np.sum(smap.flatten()) != n_edges:
                edge['source_to_edge_map'] = smap

            # create mapping from edge to target variables
            ##############################################

            # create mapping from lower dimensional source to higher dimensional edge space
            tmap = np.zeros((n_targets, n_edges))
            for edge_idx, target_idx in zip(edge['edge_idx'], edge['target_idx']):
                if type(target_idx) is tuple:
                    for t_idx, e_idx in zip(range(target_idx[0], target_idx[1]), range(edge_idx[0], edge_idx[1])):
                        tmap[t_idx, e_idx] = 1.
                elif type(target_idx) is list:
                    for t_idx, e_idx in zip(target_idx, range(edge_idx[0], edge_idx[1])):
                        tmap[t_idx, e_idx] = 1.
                else:
                    tmap[target_idx, edge_idx[0]:edge_idx[1]] = 1.

            # add mapping to edge attributs
            if n_edges != n_sources or np.sum(tmap.flatten()) != n_edges:
                edge['edge_to_target_map'] = tmap

            # add information about edge projection to target node
            ######################################################

            if edge['target_var'] not in net_config.nodes[target_name]['inputs']:
                net_config.nodes[target_name]['inputs'][edge['target_var']] = {
                    'sources': [(source_name, target_name, edge_name)],
                    'delay': [np.max(edge['delay'])],
                    'reduce_dim': True}
            else:
                net_config.nodes[target_name]['inputs'][edge['target_var']]['sources'].append(
                    (source_name, target_name, edge_name))
                net_config.nodes[target_name]['inputs'][edge['target_var']]['delay'].append(np.max(edge['delay']))

            # clean up edge
            ###############

            for key in ['edge_idx', 'source_idx', 'target_idx']:
                if key in edge.keys():
                    edge.pop(key)

        # 2. go through nodes and create mapping for their inputs
        for node_name, node in net_config.nodes.items():

            # loop over input variables of node
            for i, (in_var, input_info) in enumerate(node['inputs'].items()):

                # extract info for input variable connections
                op_name, var_name = in_var.split('/')
                target_shape = node['operator_args'][in_var]['shape']
                n_inputs = len(input_info['sources'])

                # loop over different input sources
                for j in range(n_inputs):

                    # handle delays
                    ###############

                    max_delay = input_info['delay'][j]

                    if max_delay:

                        # create buffer equations
                        if len(target_shape) == 1:
                            eqs = [f"{var_name} = {var_name}_buffer_{j}[0]",
                                   f"{var_name}_buffer_{j}_tmp = {var_name}_buffer_{j}[1:]",
                                   f"{var_name}_buffer_{j}[0:-1] = {var_name}_buffer_{j}_tmp",
                                   f"{var_name}_buffer_{j}[-1] = 0."]
                        else:
                            eqs = [f"{var_name} = {var_name}_buffer_{j}[:, 0]",
                                   f"{var_name}_buffer_{j}_tmp = {var_name}_buffer_{j}[:,1:]",
                                   f"{var_name}_buffer_{j}[:,0:-1] = {var_name}_buffer_{j}_tmp",
                                   f"{var_name}_buffer_{j}[:,-1] = 0."]

                        # add buffer operator to node
                        node['operators'][f'{var_name}_buffering_{j}'] = {
                            'equations': eqs,
                            'inputs': {},
                            'output': var_name}
                        node['op_order'] = [f'{var_name}_buffering_{j}'] + node['op_order']

                        # add buffer variable to node arguments
                        node['operator_args'][f'{var_name}_buffering_{j}/{var_name}_buffer_{j}'] = {
                            'vtype': 'state_var',
                            'dtype': 'float32',
                            'shape': target_shape + (max_delay, ),
                            'value': 0.
                            }

                        # handle operator dependencies
                        if var_name in node['operators'][op_name]['inputs'].keys():
                            node['operators'][op_name]['inputs'][var_name]['sources'].append(
                                f'{var_name}_buffering_{j}')
                        else:
                            node['operators'][op_name]['inputs'][var_name] = {'sources': [f'{var_name}_buffering_{j}'],
                                                                              'reduce_dim': True}

                        # update edge information
                        edge = net_config.edges[input_info['sources'][j]]
                        edge['target_var'] = f'{var_name}_buffering_{j}/{var_name}_buffer_{j}'

                    # handle multiple projections to same variable
                    elif n_inputs > 1:

                        # create buffer equations
                        eqs = [f"{var_name} = {var_name}_col_{j}"]

                        # add buffer operator to node
                        node['operators'][f'{var_name}_collector_{j}'] = {
                            'equations': eqs,
                            'inputs': {},
                            'output': var_name}
                        node['op_order'] = [f'{var_name}_collector_{j}'] + node['op_order']

                        # add buffer variable to node arguments
                        node['operator_args'][f'{var_name}_collector_{j}/{var_name}_col_{j}'] = {
                            'vtype': 'state_var',
                            'dtype': 'float32',
                            'shape': target_shape,
                            'value': 0.
                        }

                        # handle operator dependencies
                        if var_name in node['operators'][op_name]['inputs'].keys():
                            node['operators'][op_name]['inputs'][var_name]['sources'].append(
                                f'{var_name}_collector_{j}')
                        else:
                            node['operators'][op_name]['inputs'][var_name] = {'sources': [f'{var_name}_collector_{j}'],
                                                                              'reduce_dim': True}

                        # update edge information
                        edge = net_config.edges[input_info['sources'][j]]
                        edge['target_var'] = f'{var_name}_collector_{j}/{var_name}_col_{j}'

        return net_config

    def vectorize_nodes(self,
                        nodes: List[str],
                        new_net_config: MultiDiGraph,
                        old_net_config: MultiDiGraph
                        ) -> MultiDiGraph:
        """Combines all nodes in list to a single node and adds node to new net config.

        Parameters
        ----------
        new_net_config
            Networkx MultiDiGraph with nodes and edges as defined in init.
        old_net_config
            Networkx MultiDiGraph with nodes and edges as defined in init.
        nodes
            Names of the nodes to be combined.

        Returns
        -------
        MultiDiGraph

        """

        n_nodes = len(nodes)

        # instantiate new node
        ######################

        node_ref = nodes.pop()
        new_node = deepcopy(old_net_config.nodes[node_ref])
        self.node_arg_map[node_ref] = {}

        # go through node attributes and store their value and shape
        arg_vals, arg_shapes = {}, {}
        for arg_name, arg in new_node['operator_args'].items():

            # extract argument value and shape
            if len(arg['shape']) == 2:
                raise ValueError(f"Automatic optimization of the graph (i.e. method `vectorize`"
                                 f" cannot be applied to networks with variables of 2 or more"
                                 f" dimensions. Variable {arg_name} has shape {arg['shape']}. Please"
                                 f" turn of the `vectorize` option or change the dimensionality of the argument.")
            if len(arg['shape']) == 1:
                if type(arg['value']) is float:
                    arg['value'] = np.zeros(arg['shape']) + arg['value']
                arg_vals[arg_name] = list(arg['value'])
                arg_shapes[arg_name] = tuple(arg['shape'])
            else:
                arg_vals[arg_name] = [arg['value']]
                arg_shapes[arg_name] = (1,)

            # save index of original node's attributes in new nodes attributes
            self.node_arg_map[node_ref][arg_name] = 0

        # go through rest of the nodes and extract their argument shapes and values
        ###########################################################################

        for i, node_name in enumerate(nodes):

            node = deepcopy(old_net_config.nodes[node_name])
            self.node_arg_map[node_name] = {}

            # go through arguments
            for arg_name in arg_vals.keys():

                arg = node['operator_args'][arg_name]

                # extract value and shape of argument
                if len(arg['shape']) == 0:
                    arg_vals[arg_name].append(arg['value'])
                else:
                    if type(arg['value']) is float:
                        arg['value'] = np.zeros(arg['shape']) + arg['value']
                    arg_vals[arg_name] += list(arg['value'])
                    if arg['shape'][0] > arg_shapes[arg_name][0]:
                        arg_shapes[arg_name] = tuple(arg['shape'])

                # save index of original node's attributes in new nodes attributes
                self.node_arg_map[node_name][arg_name] = i + 1

        # go through new arguments and update shape and values
        for arg_name, arg in new_node['operator_args'].items():

            if 'value' in arg.keys():
                arg.update({'value': arg_vals[arg_name]})

            if 'shape' in arg.keys():
                arg.update({'shape': (n_nodes,) + arg_shapes[arg_name]})

        # add new, vectorized node to dictionary
        ########################################

        # define name of new node
        if '/'in node_ref:
            node_ref = node_ref.split('/')[0]
        if all([node_ref in node_name for node_name in nodes]):
            node_name = node_ref
        else:
            node_name = node_ref
            for n in nodes:
                node_name += f'_{n}'

        # add new node to new net config
        node_idx = 0
        while True:
            new_node_name = f"{node_name}/{node_idx}"
            if new_node_name in new_net_config.nodes.keys():
                node_idx += 1
            else:
                new_net_config.add_node(new_node_name, **new_node)
                break

        return new_net_config

    def vectorize_edges(self,
                        edges: list,
                        source: str,
                        target:str,
                        new_net_config: MultiDiGraph,
                        old_net_config: MultiDiGraph
                        ) -> MultiDiGraph:
        """Combines edges in list and adds a new edge to the new net config.

        Parameters
        ----------
        edges
        source
        target
        new_net_config
        old_net_config

        Returns
        -------
        MultiDiGraph

        """

        n_edges = len(edges)

        if n_edges > 0:

            # go through edges and extract weight and delay
            ###############################################

            weight_col = []
            delay_col = []
            old_edge_idx = []
            old_svar_idx = []
            old_tvar_idx = []

            for edge in edges:

                edge_dict = old_net_config.edges[edge]

                # check delay of edge
                if 'delay' not in edge_dict.keys():
                    delays = None
                elif 'int' in str(type(edge_dict['delay'])):
                    delays = [edge_dict['delay']]
                elif type(edge_dict['delay']) is np.ndarray:
                    if len(edge_dict['delay'].shape) > 1:
                        raise ValueError(f"Automatic optimization of the graph (i.e. method `vectorize`"
                                         f" cannot be applied to networks with variables of 2 or more"
                                         f" dimensions. Delay of edge between {edge[0]} and {edge[1]} has shape"
                                         f" {edge_dict['delay'].shape}. Please turn of the `vectorize` option or change the"
                                         f" edges' dimensionality.")
                    delays = list(edge_dict['delay'])
                else:
                    delays = edge_dict['delay']

                # check weight of edge
                if 'weight' not in edge_dict.keys():
                    weights = None
                elif 'float' in str(type(edge_dict['weight'])):
                    weights = [edge_dict['weight']]
                elif type(edge_dict['weight']) is np.ndarray:
                    if len(edge_dict['weight'].shape) > 1:
                        raise ValueError(f"Automatic optimization of the graph (i.e. method `vectorize`"
                                         f" cannot be applied to networks with variables of 2 or more"
                                         f" dimensions. Weight of edge between {edge[0]} and {edge[1]} has shape"
                                         f" {edge_dict['weight'].shape}. Please turn of the `vectorize` option or change the"
                                         f" edges' dimensionality.")
                    weights = list(edge_dict['weight'])
                else:
                    weights = edge_dict['weight']

                # match dimensionality of delay and weight
                if not delays or not weights:
                    pass
                elif len(delays) > 1 and len(weights) == 1:
                    weights = weights * len(delays)
                elif len(weights) > 1 and len(delays) == 1:
                    delays = delays * len(weights)
                elif len(delays) != len(weights):
                    raise ValueError(f"Dimensionality of weights and delays of edge between {edge[0]} and {edge[1]}"
                                     f" do not match. They are of length {len(weights)} and {len(delays)}."
                                     f" Please turn of the `vectorize` option or change the dimensionality of the"
                                     f" edge's attributes.")

                # add weight, delay and variable indices to collector lists
                if weights:
                    old_edge_idx.append((len(weight_col), len(weight_col) + len(weights)))
                    weight_col += weights
                else:
                    old_edge_idx.append((old_edge_idx[-1][1], old_edge_idx[-1][1] + 1))
                if delays:
                    delay_col += delays
                if 'source_idx' in edge_dict.keys():
                    idx = self.node_arg_map[edge[0]][old_net_config.edges[edge]['source_var']]
                    if type(idx) is tuple:
                        idx = range(idx[0], idx[1])
                        idx_new = []
                        for i in edge_dict['source_idx']:
                            if type(i) is tuple:
                                idx_new.append(idx[i[0]:i[1]])
                            else:
                                idx_new.append(idx[i])
                    else:
                        idx_new = idx
                    old_svar_idx.append(idx_new)
                else:
                    old_svar_idx.append(self.node_arg_map[edge[0]][old_net_config.edges[edge]['source_var']])
                if 'target_idx' in edge_dict.keys():
                    idx = self.node_arg_map[edge[1]][old_net_config.edges[edge]['target_var']]
                    if type(idx) is tuple:
                        idx = range(idx[0], idx[1])
                        idx_new = []
                        for i in edge_dict['target_idx']:
                            if type(i) is tuple:
                                idx_new.append(idx[i[0]:i[1]])
                            else:
                                idx_new.append(idx[i])
                    else:
                        idx_new = idx
                    old_tvar_idx.append(idx_new)
                else:
                    old_tvar_idx.append(self.node_arg_map[edge[1]][old_net_config.edges[edge]['target_var']])

            # create new, vectorized edge
            #############################

            # extract edge
            edge_ref = edges[0]
            new_edge = deepcopy(old_net_config.edges[edge_ref])

            # change delay and weight attributes
            new_edge['delay'] = delay_col if delay_col else None
            new_edge['weight'] = weight_col if weight_col else None
            new_edge['edge_idx'] = old_edge_idx
            new_edge['source_idx'] = old_svar_idx
            new_edge['target_idx'] = old_tvar_idx

            # add new edge to new net config
            new_net_config.add_edge(source, target, **new_edge)

        return new_net_config

    def vectorize_ops(self,
                      op_key: str,
                      nodes: list,
                      new_node: dict,
                      net_config: MultiDiGraph
                      ) -> None:
        """Vectorize all instances of an operation across nodes and put them into a single-node network.

        Parameters
        ----------
        op_key
        nodes
        new_node
        net_config

        """

        # extract operation in question
        node_name_tmp = nodes.pop(0)
        ref_node = net_config.nodes[node_name_tmp]
        op = ref_node['operators'][op_key]

        if node_name_tmp not in self.node_arg_map.keys():
            self.node_arg_map[node_name_tmp] = {}

        # collect input dependencies of all nodes
        #########################################

        for node in [node_name_tmp] + nodes:
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

                if nodes:

                    for node_name in nodes:

                        if node_name not in self.node_arg_map.keys():
                            self.node_arg_map[node_name] = {}

                        # extract node arg value and shape
                        arg_tmp = net_config.nodes[node_name]['operator_args'][key]
                        val = arg_tmp['value']
                        dims = tuple(arg_tmp['shape'])

                        # append value to argument dictionary
                        idx = (0, len(val))
                        if len(dims) > 0 and type(val) is float:
                            val = np.zeros(dims) + val
                        else:
                            val = np.array(val)
                        idx_old = 0 if len(dims) == 0 else arg['value'].shape[0]
                        arg['value'] = np.append(arg['value'], val, axis=0)
                        idx_new = arg['value'].shape[0]
                        idx = (idx_old, idx_new)

                        # add information to node_arg_map
                        self.node_arg_map[node_name][key] = idx

                else:

                    # add information to node_arg_map
                    self.node_arg_map[node_name_tmp][key] = (0, arg['value'].shape[0])

                # change shape of argument
                arg['shape'] = arg['value'].shape

                # add argument to new node's argument dictionary
                op_args[key] = arg

        # add operator information to new node
        ######################################

        # add operator
        new_node['operators'][op_key] = op
        new_node['op_order'].append(op_key)

        # add operator args
        for key, arg in op_args.items():
            new_node['operator_args'][key] = arg

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
            #for name, node in self.nodes.items():
            #    outputs[name + '_v'] = getattr(node['handle'], 'name')

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
