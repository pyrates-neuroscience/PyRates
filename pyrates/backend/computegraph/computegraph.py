"""This module provides the backend class that should be used to set-up any backend. It creates a tensorflow graph that
manages all computations/operations and a networkx graph that represents the backend structure (nodes + edges).
"""

# external imports
import tensorflow as tf
from typing import Optional, Tuple, Union, List
from pandas import DataFrame
import time as t
import numpy as np
from networkx import MultiDiGraph
from copy import copy

# pyrates imports
from pyrates.backend.parser import parse_equation, parse_dict
from pyrates.backend.backend_wrapper import TensorflowBackend

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


class ComputeGraph(MultiDiGraph):
    """ComputeGraph level class used to set up and simulate networks of nodes and edges defined by sets of operators.

    Parameters
    ----------
    net_config
        Networkx MultiDiGraph that defines the configuration of the backend. Following information need to be contained
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
        Name of the backend.

    Attributes
    ----------
    _node_arg_map
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
                 vectorize: str = 'nodes',
                 key: Optional[str] = None
                 ) -> None:
        """Instantiation of backend.
        """

        # call of super init
        ####################

        super().__init__()

        # additional object attributes
        ##############################

        self.key = key if key else 'net/0'
        self.dt = dt
        self._tf_graph = tf_graph if tf_graph else tf.get_default_graph()
        self.states = []
        self._node_arg_map = {}

        # vectorize passed dictionaries
        ###############################

        net_config = self._vectorize(net_config=net_config, vectorization_mode=vectorize)

        # create objects on tensorflow graph
        ####################################

        with self._tf_graph.as_default():

            with tf.variable_scope(self.key):

                # initialize nodes
                ##################

                node_updates = []

                for node_name, node_info in net_config.nodes.items():

                    # add simulation step-size to node arguments
                    node_args = node_info['operator_args']
                    node_args['all_ops/dt'] = {'vtype': 'raw',
                                               'value': self.dt}

                    # add node to backend
                    self.add_node(node=node_name,
                                  ops=node_info['operators'],
                                  op_args=node_args,
                                  op_order=node_info['operator_order'])

                    # collect update operation of node
                    node_updates += self.nodes[node_name]['update']

                # initialize edges
                ##################
                edge_updates = []
                for source_name, target_name, edge_idx in net_config.edges:

                    # add edge to backend
                    edge_info = net_config.edges[source_name, target_name, edge_idx]
                    self.add_edge(source_node=source_name,
                                  target_node=target_name,
                                  dependencies=node_updates,
                                  **edge_info)

                    # collect project operation of edge
                    edge_updates.append(self.edges[source_name, target_name, edge_idx]['update'])

                # create backend update operation
                #################################

                if len(edge_updates) > 0:
                    self.step = tf.group(edge_updates, name='network_update')
                else:
                    self.step = tf.group(node_updates, name='network_update')

    def run(self,
            simulation_time: Optional[float] = None,
            inputs: Optional[dict] = None,
            outputs: Optional[dict] = None,
            sampling_step_size: Optional[float] = None,
            out_dir: Optional[str] = None,
            verbose: bool=True,
            ) -> Tuple[DataFrame, float]:
        """Simulate the backend behavior over time via a tensorflow session.

        Parameters
        ----------
        simulation_time
            Simulation time in seconds.
        inputs
            Inputs for placeholder variables. Each key is a tuple that specifies a placeholder variable in the graph
            in the following format: (node_name, op_name, var_name). Each value is an array that defines the input for
            the placeholder variable over time (first dimension).
        outputs
            Output variables that will be returned. Each key is the desired name of an output variable and each value is
            a tuple that specifies a variable in the graph in the following format: (node_name, op_name, var_name).
        sampling_step_size
            Time in seconds between sampling points of the output variables.
        out_dir
            Directory in which to store outputs.
        verbose
            If true, status updates will be printed to the console.

        Returns
        -------
        tuple
            First entry of the tuple contains the output variables in a pandas dataframe, the second contains the
            simulation time in seconds.

        """

        # prepare simulation
        ####################

        # basic simulation parameters initialization
        if not simulation_time:
            simulation_time = self.dt
        sim_steps = int(simulation_time / self.dt)

        if not sampling_step_size:
            sampling_step_size = self.dt
        sampling_steps = int(sampling_step_size / self.dt)

        # define output variables
        if not outputs:

            outputs_tmp = dict()

        else:

            outputs_tmp = dict()

            # go through  output variables
            for key, val in outputs.items():

                if val[0] == 'all':

                    # collect output variable from every node in backend
                    for node in self.nodes.keys():
                        outputs_tmp[f'{node}/{key}'] = self.get_var(node=node, op=val[1], var=val[2])

                elif val[0] in self.nodes.keys() or val[0] in self._node_arg_map.keys():

                    # get output variable of specific backend node
                    outputs_tmp[key] = self.get_var(node=val[0], op=val[1], var=val[2])

                elif any([val[0] in key for key in self.nodes.keys()]):

                    # get output variable from backend nodes of a certain type
                    for node in self.nodes.keys():
                        if val[0] in node:
                            outputs_tmp[f'{key}/{node}'] = self.get_var(node=node, op=val[1], var=val[2])
                else:

                    # get output variable of specific, vectorized  backend node
                    for node in self._node_arg_map.keys():
                        if val[0] in node and 'comb' not in node:
                            outputs_tmp[f'{key}/{node}'] = self.get_var(node=node, op=val[1], var=val[2])

        # add output collector variables to graph
        output_col = {}
        store_ops = []
        with self._tf_graph.as_default():
            with tf.variable_scope(f'{self.key}_output_col'):

                # create counting index for collector variables
                out_idx = tf.Variable(0, dtype=tf.int32, name='out_var_idx')

                # add collector variables to the graph
                for key, var in outputs_tmp.items():
                    output_col[key] = tf.get_variable(name=key,
                                                      dtype=tf.float32,
                                                      shape=[int(sim_steps / sampling_steps) + 1] + list(var.shape),
                                                      initializer=tf.constant_initializer())

                    # add collect operation to the graph
                    store_ops.append(tf.scatter_update(output_col[key], out_idx, var))

                # create increment operator for counting index
                with tf.control_dependencies(store_ops):
                    sample = out_idx.assign_add(1)

        # linearize input dictionary
        if inputs:
            inp = list()
            for step in range(sim_steps):
                inp_dict = dict()
                for key, val in inputs.items():

                    if self.key in self.nodes.keys():

                        # fully vectorized case: add vectorized placeholder variable to input dictionary
                        var = self.get_var(node=self.key, op=key[1], var=key[2])
                        inp_dict[var] = np.reshape(val[step], var.shape)

                    elif any(['comb' in key_tmp for key_tmp in self.nodes.keys()]):

                        # node-vectorized case
                        if key[0] == 'all':

                            # go through all nodes, extract the variable and add it to input dict
                            for i, node in enumerate(self.nodes.keys()):
                                var = self.get_var(node=node, op=key[1], var=key[2])
                                inp_dict[var] = np.reshape(val[step, i], var.shape)

                        elif key[0] in self.nodes.keys():

                            # add placeholder variable of node(s) to input dictionary
                            var = self.get_var(node=key[0], op=key[1], var=key[2])
                            inp_dict[var] = np.reshape(val[step], var.shape)

                        elif any([key[0] in key_tmp for key_tmp in self.nodes.keys()]):

                            # add vectorized placeholder variable of specified node type to input dictionary
                            for node in list(self.nodes.keys()):
                                if key[0] in node:
                                    break
                            var = self.get_var(node=node, op=key[1], var=key[2])
                            inp_dict[var] = np.reshape(val[step], var.shape)

                    else:

                        # non-vectorized case
                        if key[0] == 'all':

                            # go through all nodes, extract the variable and add it to input dict
                            for i, node in enumerate(self.nodes.keys()):
                                var = self.get_var(node=node, op=key[1], var=key[2])
                                inp_dict[var] = np.reshape(val[step, i], var.shape)

                        elif any([key[0] in key_tmp for key_tmp in self.nodes.keys()]):

                            # extract variables from nodes of specified type
                            i = 0
                            for node in self.nodes.keys():
                                if key[0] in node:
                                    var = self.get_var(node=node, op=key[1], var=key[2])
                                    inp_dict[var] = np.reshape(val[step, i], var.shape)
                                    i += 1

                # add input dictionary to placeholder input liust
                inp.append(inp_dict)

        else:

            # create list of nones (no placeholder variables should exist)
            inp = [None for _ in range(sim_steps)]

        # run simulation
        ################

        with tf.Session(graph=self._tf_graph) as sess:

            # initialize session log
            if out_dir:
                writer = tf.summary.FileWriter(out_dir, graph=self._tf_graph)

            # initialize all variables
            sess.run(tf.global_variables_initializer())

            sess.run(sample)
            t_start = t.time()

            # simulate backend behavior for each time-step
            for step in range(sim_steps):
                sess.run(self.step, inp[step])
                if step % sampling_steps == 0:
                    sess.run(sample)

            # display simulation time
            t_end = t.time()
            if verbose:
                if simulation_time:
                    print(f"{simulation_time}s of backend behavior were simulated in {t_end - t_start} s given a "
                          f"simulation resolution of {self.dt} s.")
                else:
                    print(f"ComputeGraph computations finished after {t_end - t_start} seconds.")

            # close session log
            if out_dir:
                writer.close()

            # store output variables in data frame
            for i, (key, var) in enumerate(output_col.items()):
                if i == 0:
                    var = np.squeeze(var.eval())
                    try:
                        out_vars = DataFrame(data=var,
                                             columns=[f'{key}_{j}' for j in range(var.shape[1])])
                    except IndexError:
                        out_vars = DataFrame(data=var, columns=[key])
                else:
                    var = np.squeeze(var.eval())
                    try:
                        for j in range(var.shape[1]):
                            out_vars[f'{key}_{j}'] = var[:, j]
                    except IndexError:
                        out_vars[key] = var

            if not outputs:
                out_vars = DataFrame()
            else:
                out_vars['time'] = np.arange(0., simulation_time + sampling_step_size * 0.5, sampling_step_size)

        return out_vars, (t_end - t_start)

    def get_var(self, var: str, node: str, op: str) -> Union[tf.Tensor, tf.Variable]:
        """Extracts variable from a specific operator of a specific node in the graph.


        Parameters
        ----------
        var
            Variable name.
        node
            Node name.
        op
            Operator name.

        Returns
        -------
        Union[tf.Tensor, tf.Variable]
            Tensorflow representation of the variable.

        """

        try:
            tf_var = self.nodes[node][f'{op}/{var}']
        except KeyError:
            node, var, idx = self._node_arg_map[node][f'{op}/{var}']
            if node in self._node_arg_map.keys():
                op, var = var.split('/')
                try:
                    return self.get_var(var, node, op)[idx[0]:idx[1]]
                except TypeError:
                    return self.get_var(var, node, op)[idx]
            try:
                tf_var = self.nodes[node][var][idx[0]: idx[1]]
            except TypeError:
                tf_var = self.nodes[node][var][idx]

        return tf_var

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
        final_node_ops = []

        with self._tf_graph.as_default():

            # initialize variable scope of node
            with tf.variable_scope(node):

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
                                                tf_graph=self._tf_graph)

                        # bind tensorflow variables to node
                        for var_name, var in op_args_tf.items():
                            node_attr.update({f'{op_name}/{var_name}': var})
                            op_args_tf[var_name] = var

                        # set input dependencies
                        ########################

                        for var_name, inp in op['inputs'].items():

                            if inp['sources']:

                                # collect input variable calculation operations
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
                                            raise ValueError(f"Invalid dependencies found in operator: "
                                                             f"{op['equations']}. Input Variable {var_name} has not "
                                                             f"been calculated yet. Consider changes")

                                        # append operator output variable to list
                                        out_vars.append(op_args[out_var])
                                        if out_var in final_node_ops:
                                            final_node_ops.pop(final_node_ops.index(out_var))

                                    # for multiple input operators
                                    else:

                                        out_vars_tmp = []
                                        out_var_idx_tmp = []

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
                                            out_vars_tmp.append(op_args[out_var])
                                            if out_var in final_node_ops:
                                                final_node_ops.pop(final_node_ops.index(out_var))

                                        # add tensorflow operations for grouping the inputs together
                                        tf_var_tmp = tf.parallel_stack(out_vars_tmp)
                                        if inp['reduce_dim'][i]:
                                            tf_var_tmp2 = tf.reduce_sum(tf_var_tmp, 0)
                                        else:
                                            tf_var_tmp2 = tf.reshape(tf_var_tmp,
                                                                     shape=(tf_var_tmp.shape[0] *
                                                                            tf_var_tmp.shape[1],))

                                        # append variable and operation to list
                                        out_vars.append(tf_var_tmp2)
                                        out_var_idx.append(out_var_idx_tmp)

                                # add inputs to argument dictionary (in reduced or stacked form)
                                ################################################################

                                # for multiple multiple input operations
                                if len(out_vars) > 1:

                                    # find shape of smallest input variable
                                    min_shape = min([outvar.shape[0] if len(outvar.shape) > 0 else 0
                                                     for outvar in out_vars])

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
                                    tf_var = tf.stack(out_vars_new)
                                    if type(inp['reduce_dim']) is bool and inp['reduce_dim']:
                                        tf_var_inp = tf.reduce_sum(tf_var, 0)
                                    else:
                                        tf_var_inp = tf.reshape(tf_var, shape=(tf_var.shape[0] * tf_var.shape[1],))

                                # for a single input variable
                                else:

                                    tf_var_inp = out_vars[0]

                                # add input variable information to argument dictionary
                                if var_name not in op_args_tf.keys():
                                    op_args_tf[var_name] = tf.get_variable(name=var_name,
                                                                           shape=tf_var_inp.shape,
                                                                           dtype=tf_var_inp.dtype)
                                op_args_tf[var_name] = self._assign_to_var(op_args_tf[var_name], tf_var_inp)

                        # create tensorflow operator
                        tf_ops, op_args_tf = self.add_operator(expressions=op['equations'],
                                                               expression_args=op_args_tf,
                                                               variable_scope=op_name)

                        # bind newly created tf variables to node
                        for var_name, tf_var in op_args_tf.items():
                            attr_name = f"{op_name}/{var_name}"
                            if attr_name not in node_attr.keys():
                                node_attr[attr_name] = tf_var
                            op_args[attr_name] = tf_var

                        # handle dependencies
                        out_var = f"{op_name}/{op['output']}"
                        op_args[out_var] = tf_ops
                        final_node_ops.append(out_var)

                # collect remaining operator update operations
                node_attr['update'] = [op_args[out_var] for out_var in final_node_ops]

        # call super method
        super().add_node(node, **node_attr)

    def add_edge(self,
                 source_node: str,
                 target_node: str,
                 source_var: str,
                 target_var: str,
                 source_idx: Optional[list] = None,
                 target_idx: Optional[list] = None,
                 weight: Optional[Union[float, list, np.ndarray]] = 1.,
                 delay: Optional[Union[float, list, np.ndarray]] = 0.,
                 dependencies: Optional[list] = None
                 ) -> None:
        """Add edge to the backend that connects two variables from a source and a target node.

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
        source_idx
        target_idx
        weight
            Weighting constant that will be applied to the source output.
        delay
            Time that it takes for the source output to arrive at the target.
        dependencies
            List with tensorflow operations which the edge has to wait for.

        Returns
        -------
        None

        """

        # check data type of delay
        ##########################

        if delay:
            if type(delay) is list:
                if 'float' in str(type(delay[0])):
                    delay = [int(d/self.dt) + 1 for d in delay]
            if 'float' in str(type(delay)):
                delay = int(delay/self.dt) + 1

        # add projection operation of edge to tensorflow graph
        ######################################################

        with self._tf_graph.as_default():

            # set variable scope of edge

            edge_idx = self.number_of_edges(source_node, target_node)

            with tf.variable_scope(f'{source_node}_{target_node}_{edge_idx}'):

                # create edge operator arguments
                source, target = self.nodes[source_node], self.nodes[target_node]
                svar = source[source_var]
                tvar = target[target_var]
                c = tf.constant(weight, name='c', dtype=svar.dtype)

                # add operator to tensorflow graph
                ##################################

                if not dependencies:
                    dependencies = []

                with tf.control_dependencies(dependencies):

                    # apply edge weights to source variable
                    if not source_idx:
                        source_val = svar
                    else:
                        try:
                            source_val = tf.gather_nd(svar, source_idx)
                        except ValueError:
                            try:
                                source_val = tf.gather_nd(svar[:, 0], source_idx)
                            except ValueError:
                                source_val = tf.gather_nd(tf.squeeze(svar), source_idx)
                    if source_val.shape == c.shape:
                        edge_val = source_val * c
                    elif len(c.shape) > 0 and (len(c.shape) < len(source_val.shape)):
                        edge_val = source_val * tf.reshape(c, [c.shape[0], source_val.shape[1]])
                    elif len(source_val.shape) > 0 and (len(source_val.shape) < len(c.shape)):
                        edge_val = source_val * tf.squeeze(c)
                    else:
                        edge_val = source_val * c

                    # apply update to target variable
                    if delay and target_idx:

                        # apply update with delay to indices in buffer
                        target_idx = [idx + [d] for idx, d in zip(target_idx, delay)]
                        try:
                            tf_op = tf.scatter_nd_add(tvar, target_idx, edge_val)
                        except ValueError:
                            try:
                                tf_op = tf.scatter_nd_add(tvar, target_idx, tf.squeeze(edge_val))
                            except ValueError:
                                tf_op = tf.scatter_nd_add(tvar, target_idx, edge_val[:, 0])

                    elif not target_idx and delay is None:

                        # apply update directly to target variable
                        tf_op = self._assign_to_var(tvar, edge_val, add=False, solve=False)

                    else:

                        # apply update via target or delay indices
                        if not target_idx:
                            if type(delay) is list:
                                target_idx = [d for d in delay]
                            elif 'float' in str(type(delay)):
                                target_idx = [int(delay/self.dt) + 1]
                            else:
                                target_idx = [delay]
                        else:
                            target_idx = [idx[0] if type(idx) is list else idx for idx in target_idx]
                        if not delay:
                            tvar = tvar.assign(np.zeros(tvar.shape), use_locking=True)
                        try:
                            tf_op = tf.scatter_add(tvar, target_idx, edge_val)
                        except ValueError:
                            try:
                                tf_op = tf.scatter_add(tvar[:, 0], target_idx, edge_val)
                            except (ValueError, AttributeError):
                                try:
                                    tf_op = tf.scatter_add(tvar, target_idx, tf.squeeze(edge_val))
                                except ValueError:
                                    tf_op = tf.scatter_add(tvar,
                                                           target_idx,
                                                           tf.reshape(edge_val, [edge_val.shape[0], tvar.shape[1]])
                                                           )

        # add edge to networkx graph
        ############################

        super().add_edge(source_node, target_node, update=tf_op, svar=svar, tvar=tvar, weight=c, delay=delay)

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

        with self._tf_graph.as_default():

            # set variable scope
            with tf.variable_scope(variable_scope):

                # go through mathematical equations
                for i, expr in enumerate(expressions):

                    # parse equation
                    update, expression_args = parse_equation(expr, expression_args, tf_graph=self._tf_graph)

                    # store tensorflow operation in list
                    evals.append(update)

                # go through evals and assign right-hand side updates to left-hand sides for all equations except DEs
                evals_complete = []
                de_lhs = []
                de_rhs = []
                for ev in evals:
                    if ev[1] is None:
                        evals_complete.append(ev[0])
                    elif not ev[2]:
                        assign_op = self._assign_to_var(ev[0], ev[1], add=False, dependencies=evals_complete)
                        evals_complete.append(assign_op)
                    else:
                        de_lhs.append(ev[0])
                        de_rhs.append(ev[1])

                # go through DEs and assign rhs to lhs
                deps = de_rhs + evals_complete
                de_complete = []
                for lhs, rhs in zip(de_lhs, de_rhs):
                    de_complete.append(self._assign_to_var(lhs, rhs, add=True, solve=True, dependencies=deps))

                if de_complete:
                    return de_complete, expression_args
                else:
                    return evals_complete, expression_args

    def _vectorize(self, net_config: MultiDiGraph, vectorization_mode: Optional[str] = 'nodes') -> MultiDiGraph:
        """Method that goes through the nodes and edges dicts and vectorizes those that are governed by the same
        operators/equations.

        Parameters
        ----------
        net_config
            See argument description at class level.
        vectorization_mode
            Can be 'none' for no vectorization, 'nodes' for vectorization over nodes and 'ops' for vectorization over
            nodes and operations.

        Returns
        -------
        Tuple[NodeView, EdgeView]

        """

        # First stage: Vectorize over nodes
        ###################################

        if vectorization_mode in 'nodesops':

            new_net_config = MultiDiGraph()

            # extract operations existing on each node
            node_ops = []
            node_names = []
            for node_name, node in net_config.nodes.items():
                node_ops.append(set(node['operator_order']))
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
                new_net_config = self._vectorize_nodes(new_net_config=new_net_config,
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
                        new_net_config = self._vectorize_edges(edges=edges,
                                                               source=source_new,
                                                               target=target_new,
                                                               new_net_config=new_net_config,
                                                               old_net_config=net_config)

            net_config = new_net_config.copy()
            new_net_config.clear()

        # Second stage: Vectorize over operators
        ########################################

        if vectorization_mode == 'ops':

            new_net_config = MultiDiGraph()
            new_net_config.add_node(self.key, operators={}, operator_order=[], operator_args={}, inputs={})

            # vectorize node operators
            ###########################

            # collect all operation keys, arguments and dependencies of each node
            op_info = {'keys': [], 'args': [], 'vectorized': [], 'deps': []}
            for node in net_config.nodes.values():
                op_info['keys'].append(node['operator_order'])
                op_info['args'].append(list(node['operator_args'].keys()))
                op_info['vectorized'].append([False] * len(op_info['keys'][-1]))
                node_deps = []
                for op_name in node['operator_order']:
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
                                self._vectorize_ops(new_node=new_net_config.nodes[self.key],
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
                                self._vectorize_ops(new_node=new_net_config.nodes[self.key],
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
            source_new, target_new = self.key, self.key

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
                new_net_config = self._vectorize_edges(edges=edges,
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

            # add information about edge projection to target node
            edge = net_config.edges[source_name, target_name, edge_name]
            if edge['target_var'] not in net_config.nodes[target_name]['inputs']:
                net_config.nodes[target_name]['inputs'][edge['target_var']] = {
                    'sources': [(source_name, target_name, edge_name)],
                    'delay': np.max(edge['delay']),
                    'reduce_dim': True}
            else:
                net_config.nodes[target_name]['inputs'][edge['target_var']]['sources'].append(
                    (source_name, target_name, edge_name))
                max_delay = np.max(edge['delay'])
                if max_delay:
                    if not net_config.nodes[target_name]['inputs'][edge['target_var']]['delay'] \
                            or max_delay > net_config.nodes[target_name]['inputs'][edge['target_var']]['delay']:
                        net_config.nodes[target_name]['inputs'][edge['target_var']]['delay'] = max_delay

        # 2. go through nodes and create mapping for their inputs
        for node_name, node in net_config.nodes.items():

            # loop over input variables of node
            for i, (in_var, input_info) in enumerate(node['inputs'].items()):

                # extract info for input variable connections
                op_name, var_name = in_var.split('/')
                target_shape = node['operator_args'][in_var]['shape']
                n_inputs = len(input_info['sources'])

                max_delay = input_info['delay']

                # loop over different input sources
                for j in range(n_inputs):

                    # handle delays
                    ###############

                    if max_delay is not None:

                        # create buffer equations
                        if len(target_shape) < 1:
                            eqs_op_read = [f"{var_name} = {var_name}_buffer_{j}[0]"]
                            eqs_op_rotate = [f"{var_name}_buffer_{j}_tmp = {var_name}_buffer_{j}[1:]",
                                             f"{var_name}_buffer_{j}[0:-1] = {var_name}_buffer_{j}_tmp",
                                             f"{var_name}_buffer_{j}[-1] = 0."]
                        else:
                            eqs_op_read = [f"{var_name} = {var_name}_buffer_{j}[:, 0]"]
                            eqs_op_rotate = [f"{var_name}_buffer_{j}_tmp = {var_name}_buffer_{j}[:, 1:]",
                                             f"{var_name}_buffer_{j}[:, 0:-1] = {var_name}_buffer_{j}_tmp",
                                             f"{var_name}_buffer_{j}[:, -1] = 0."]

                        # add buffer operators to node
                        node['operators'][f'{op_name}_{var_name}_rotate_buffer_{j}'] = {
                            'equations': eqs_op_rotate,
                            'inputs': {},
                            'output': f'{var_name}_buffer_{j}'}
                        node['operators'][f'{op_name}_{var_name}_read_buffer_{j}'] = {
                            'equations': eqs_op_read,
                            'inputs': {f'{var_name}_buffer_{j}':
                                           {'sources': [f'{op_name}_{var_name}_rotate_buffer_{j}'],
                                            'reduce_dim': False}},
                            'output': var_name}
                        node['operator_order'] = [f'{op_name}_{var_name}_rotate_buffer_{j}',
                                                  f'{op_name}_{var_name}_read_buffer_{j}',
                                                  ] + node['operator_order']

                        # add buffer variable to node arguments
                        if 'float' in str(type(max_delay)):
                            max_delay = int(max_delay / self.dt) + 1
                        if len(target_shape) > 0:
                            buffer_shape = [target_shape[0], max_delay + 1]
                        else:
                            buffer_shape = [max_delay + 1]
                        node['operator_args'][f'{op_name}_{var_name}_rotate_buffer_{j}/{var_name}_buffer_{j}'] = {
                            'vtype': 'state_var',
                            'dtype': 'float32',
                            'shape': buffer_shape,
                            'value': 0.
                        }

                        # handle operator dependencies
                        if var_name in node['operators'][op_name]['inputs'].keys():
                            node['operators'][op_name]['inputs'][var_name]['sources'].append(
                                f'{op_name}_{var_name}_read_buffer_{j}')
                        else:
                            node['operators'][op_name]['inputs'][var_name] = {'sources':
                                                                              [f'{op_name}_{var_name}_read_buffer_{j}'],
                                                                              'reduce_dim': True}

                        # update edge information
                        edge = net_config.edges[input_info['sources'][j]]
                        edge['target_var'] = f'{op_name}_{var_name}_rotate_buffer_{j}/{var_name}_buffer_{j}'

                    # handle multiple projections to same variable
                    elif n_inputs > 1:

                        # create buffer equations
                        eqs = [f"{var_name} = {var_name}_col_{j}"]

                        # add buffer operator to node
                        node['operators'][f'{op_name}_{var_name}_col_{j}'] = {
                            'equations': eqs,
                            'inputs': {},
                            'output': var_name}
                        node['operator_order'] = [f'{op_name}_{var_name}_col_{j}'] + node['operator_order']

                        # add buffer variable to node arguments
                        node['operator_args'][f'{op_name}_{var_name}_col_{j}/{var_name}_col_{j}'] = {
                            'vtype': 'state_var',
                            'dtype': 'float32',
                            'shape': target_shape,
                            'value': 0.
                        }

                        # handle operator dependencies
                        if var_name in node['operators'][op_name]['inputs'].keys():
                            node['operators'][op_name]['inputs'][var_name]['sources'].append(
                                f'{op_name}_{var_name}_col_{j}')
                        else:
                            node['operators'][op_name]['inputs'][var_name] = {'sources': [f'{op_name}_'
                                                                                          f'{var_name}_col_{j}'],
                                                                              'reduce_dim': True}

                        # update edge information
                        edge = net_config.edges[input_info['sources'][j]]
                        edge['target_var'] = f'{op_name}_{var_name}_col_{j}/{var_name}_col_{j}'

        return net_config

    def _vectorize_nodes(self,
                         nodes: List[str],
                         new_net_config: MultiDiGraph,
                         old_net_config: MultiDiGraph
                         ) -> MultiDiGraph:
        """Combines all nodes in list to a single node and adds node to new net config.

        Parameters
        ----------
        nodes
            Names of the nodes to be combined.
        new_net_config
            Networkx MultiDiGraph with nodes and edges as defined in init.
        old_net_config
            Networkx MultiDiGraph with nodes and edges as defined in init.

        Returns
        -------
        MultiDiGraph
            Updated new net configuration.

        """

        n_nodes = len(nodes)

        # instantiate new node
        ######################

        node_ref = nodes.pop()
        new_node = old_net_config.nodes[node_ref]
        if node_ref not in self._node_arg_map.keys():
            self._node_arg_map[node_ref] = {}

        # define new node's name
        node_idx = 0
        node_ref_tmp = node_ref.split('/')[0] if '/'in node_ref else node_ref
        if all([node_ref_tmp in node_name for node_name in nodes]):
            node_name = f'{node_ref_tmp}_comb'
        else:
            node_name = node_ref_tmp
            for n in nodes:
                n_tmp = n.split('/')[0] if '/' in n else n
                node_name += f'_{n_tmp}'
                test_names = []
                for test_name in nodes:
                    test_name_tmp = test_name.split('/')[0] if '/' in test_name else test_name
                    test_names.append(test_name_tmp)
                if all([test_name in node_name for test_name in test_names]):
                    break
            node_name += '_comb'

        while True:
            new_node_name = f"{node_name}/{node_idx}"
            if new_node_name in new_net_config.nodes.keys():
                node_idx += 1
            else:
                break

        # go through node attributes and store their value and shape
        arg_vals, arg_shapes = {}, {}
        for arg_name, arg in new_node['operator_args'].items():

            # extract argument value and shape
            if arg['vtype'] == 'raw':
                if type(arg['value']) is list or type(arg['value']) is tuple:
                    arg_vals[arg_name] = list(arg['value'])
                else:
                    arg_vals[arg_name] = arg['value']
            else:
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
            self._node_arg_map[node_ref][arg_name] = [new_node_name, arg_name, 0]

        # go through rest of the nodes and extract their argument shapes and values
        ###########################################################################

        for i, node_name in enumerate(nodes):

            node = old_net_config.nodes[node_name]
            if node_name not in self._node_arg_map.keys():
                self._node_arg_map[node_name] = {}

            # go through arguments
            for arg_name in arg_vals.keys():

                arg = node['operator_args'][arg_name]

                # extract value and shape of argument
                if arg['vtype'] == 'raw':
                    if type(arg['value']) is list or type(arg['value']) is tuple:
                        for j, val in enumerate(arg['value']):
                            arg_vals[arg_name][j] += val
                else:
                    if len(arg['shape']) == 0:
                        arg_vals[arg_name].append(arg['value'])
                    else:
                        if type(arg['value']) is float:
                            arg['value'] = np.zeros(arg['shape']) + arg['value']
                        arg_vals[arg_name] += list(arg['value'])
                        if arg['shape'][0] > arg_shapes[arg_name][0]:
                            arg_shapes[arg_name] = tuple(arg['shape'])

                # save index of original node's attributes in new nodes attributes
                self._node_arg_map[node_name][arg_name] = [new_node_name, arg_name, i + 1]

        # go through new arguments and update shape and values
        for arg_name, arg in new_node['operator_args'].items():

            if 'value' in arg.keys():
                arg.update({'value': arg_vals[arg_name]})

            if 'shape' in arg.keys():
                arg.update({'shape': (n_nodes,) + arg_shapes[arg_name]})

        # add new, vectorized node to dictionary
        ########################################

        # add new node to new net config
        new_net_config.add_node(new_node_name, **new_node)

        return new_net_config

    def _vectorize_edges(self,
                         edges: list,
                         source: str,
                         target: str,
                         new_net_config: MultiDiGraph,
                         old_net_config: MultiDiGraph
                         ) -> MultiDiGraph:
        """Combines edges in list and adds a new edge to the new net config.

        Parameters
        ----------
        edges
            Identifiers of the edges that should be vectorized/added to new net config.
        source
            Name of the source node
        target
            Name of the target node
        new_net_config
            Networkx MultiDiGraph containing the new, vectorized nodes.
        old_net_config
            Networkx MultiDiGraph containing the old, non-vectorized backend configuration.

        Returns
        -------
        MultiDiGraph
            Updated backend graph (with new, vectorized edges added).

        """

        n_edges = len(edges)

        if n_edges > 0:

            # go through edges and extract weight and delay
            ###############################################

            weight_col = []
            delay_col = []
            old_svar_idx = []
            old_tvar_idx = []

            for edge in edges:

                edge_dict = old_net_config.edges[edge]

                # check delay of edge
                if 'delay' not in edge_dict.keys():
                    delays = None
                elif 'int' in str(type(edge_dict['delay'])):
                    delays = [edge_dict['delay']]
                elif'float' in str(type(edge_dict['delay'])):
                    delays = [int(edge_dict['delay']/self.dt) + 1]
                elif type(edge_dict['delay']) is np.ndarray:
                    if len(edge_dict['delay'].shape) > 1:
                        raise ValueError(f"Automatic optimization of the graph (i.e. method `vectorize`"
                                         f" cannot be applied to networks with variables of 2 or more"
                                         f" dimensions. Delay of edge between {edge[0]} and {edge[1]} has shape"
                                         f" {edge_dict['delay'].shape}. Please turn of the `vectorize` option or change"
                                         f" the edges' dimensionality.")
                    if 'float' in edge_dict['delay'].dtype:
                        edge_dict['delay'] = np.ndarray(edge_dict['delay'] / self.dt + 1, dtype=np.int32)
                    delays = list(edge_dict['delay'])
                elif edge_dict['delay']:
                    delays = [int(d/self.dt) + 1 if 'float' in str(type(d)) else d for d in edge_dict['delay']]
                else:
                    delays = None

                # check weight of edge
                if 'weight' not in edge_dict.keys():
                    weights = None
                elif 'float' in str(type(edge_dict['weight'])):
                    weights = [edge_dict['weight']]
                elif 'int' in str(type(edge_dict['weight'])):
                    weights = [float(edge_dict['weight'])]
                elif type(edge_dict['weight']) is np.ndarray:
                    if len(edge_dict['weight'].shape) > 1:
                        raise ValueError(f"Automatic optimization of the graph (i.e. method `vectorize`"
                                         f" cannot be applied to networks with variables of 2 or more"
                                         f" dimensions. Weight of edge between {edge[0]} and {edge[1]} has shape"
                                         f" {edge_dict['weight'].shape}. Please turn of the `vectorize` option or"
                                         f" change the edges' dimensionality.")
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
                if delays:
                    delay_col += delays
                if weights:
                    weight_col += weights
                if 'source_idx' in edge_dict.keys():
                    _, _, idx = self._node_arg_map[edge[0]][old_net_config.edges[edge]['source_var']]
                    if type(idx) is tuple:
                        idx = range(idx[0], idx[1])
                        idx_new = []
                        for i in edge_dict['source_idx']:
                            if type(i) is tuple:
                                idx_new.append([idx[i[0]:i[1]]])
                            elif type(i) is list:
                                idx_new.append([idx[i[0]]])
                            else:
                                idx_new.append([idx[i]])
                    else:
                        idx_new = [[idx]]
                    old_svar_idx += idx_new
                else:
                    _, _, idx_new = self._node_arg_map[edge[0]][old_net_config.edges[edge]['source_var']]
                    if type(idx_new) is int:
                        idx_new = [idx_new]
                    else:
                        idx_new = list(idx_new)
                    old_svar_idx.append(idx_new)
                if 'target_idx' in edge_dict.keys():
                    _, _, idx = self._node_arg_map[edge[1]][old_net_config.edges[edge]['target_var']]
                    if type(idx) is tuple:
                        idx = range(idx[0], idx[1])
                        idx_new = []
                        for i in edge_dict['target_idx']:
                            if type(i) is tuple:
                                idx_new.append([idx[i[0]:i[1]]])
                            elif type(i) is list:
                                idx_new.append([idx[i[0]]])
                            else:
                                idx_new.append([idx[i]])
                    else:
                        idx_new = [[idx]]
                    old_tvar_idx += idx_new
                else:
                    _, _, idx_new = self._node_arg_map[edge[1]][old_net_config.edges[edge]['target_var']]
                    if type(idx_new) is int:
                        idx_new = [idx_new]
                    else:
                        idx_new = list(idx_new)
                    old_tvar_idx.append(idx_new)

            # create new, vectorized edge
            #############################

            # extract edge
            edge_ref = edges[0]
            new_edge = old_net_config.edges[edge_ref]

            # change delay and weight attributes
            new_edge['delay'] = delay_col if delay_col else None
            new_edge['weight'] = weight_col if weight_col else None
            new_edge['source_idx'] = old_svar_idx
            new_edge['target_idx'] = old_tvar_idx

            # add new edge to new net config
            new_net_config.add_edge(source, target, **new_edge)

        return new_net_config

    def _vectorize_ops(self,
                       op_key: str,
                       nodes: list,
                       new_node: dict,
                       net_config: MultiDiGraph
                       ) -> None:
        """Vectorize all instances of an operation across nodes and put them into a single-node backend.

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

        if node_name_tmp not in self._node_arg_map.keys():
            self._node_arg_map[node_name_tmp] = {}

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

                if arg['vtype'] == 'raw':

                    if type(arg['value']) is list:

                        self._node_arg_map[node_name_tmp][key] = [self.key, key, (0, len(arg['value']))]

                        for node_name in nodes:

                            if node_name not in self._node_arg_map.keys():

                                self._node_arg_map[node_name] = {}

                                new_val = net_config.nodes[node_name]['operator_args'][key]['value']
                                old_idx = len(arg['value'])
                                arg['value'].append(new_val)

                                self._node_arg_map[node_name_tmp][key] = [self.key,
                                                                          key,
                                                                          (old_idx, arg['value'].shape[0])]

                else:

                    if not hasattr(arg['value'], 'shape'):
                        arg['value'] = np.array(arg['value'])
                    if arg['value'].shape != arg['shape']:
                        try:
                            arg['value'] = np.reshape(arg['value'], arg['shape'])
                        except ValueError:
                            arg['value'] = np.zeros(arg['shape']) + arg['value']

                    # go through nodes and extract shape and value information for arg
                    if nodes:

                        self._node_arg_map[node_name_tmp][key] = [self.key, key, (0, arg['value'].shape[0])]

                        for node_name in nodes:

                            if node_name not in self._node_arg_map.keys():
                                self._node_arg_map[node_name] = {}

                            # extract node arg value and shape
                            arg_tmp = net_config.nodes[node_name]['operator_args'][key]
                            val = arg_tmp['value']
                            dims = tuple(arg_tmp['shape'])

                            # append value to argument dictionary
                            if len(dims) > 0 and type(val) is float:
                                val = np.zeros(dims) + val
                            else:
                                val = np.reshape(np.array(val), dims)
                            if len(dims) == 0:
                                idx_old = 0
                            else:
                                idx_old = arg['value'].shape[0]
                            arg['value'] = np.append(arg['value'], val, axis=0)
                            idx_new = arg['value'].shape[0]
                            idx = (idx_old, idx_new)

                            # add information to _node_arg_map
                            self._node_arg_map[node_name][key] = [self.key, key, idx]

                    else:

                        # add information to _node_arg_map
                        self._node_arg_map[node_name_tmp][key] = [self.key, key, (0, arg['value'].shape[0])]

                    # change shape of argument
                    arg['shape'] = arg['value'].shape

                # add argument to new node's argument dictionary
                op_args[key] = arg

        # add operator information to new node
        ######################################

        # add operator
        new_node['operators'][op_key] = op
        new_node['operator_order'].append(op_key)

        # add operator args
        for key, arg in op_args.items():
            new_node['operator_args'][key] = arg

    def _assign_to_var(self,
                       var: tf.Variable,
                       val: Union[tf.Variable, tf.Operation, tf.Tensor],
                       add: Optional[bool] = False,
                       solve: Optional[bool] = False,
                       dependencies: Optional[list] = None
                       ) -> Union[tf.Variable, tf.Operation, tf.Tensor]:
        """

        Parameters
        ----------
        var
            Tensorflow variable.
        val
            New value that should be assigned to variable.
        add
            If true, assign_add will be used instead of assign.
        solve
            If true, assign process will be treated as update of a differential equation.
        dependencies
            List of tensorflow operations the assign op should wait for to finish.

        Returns
        -------
        Union[tf.Variable, tf.Operation, tf.Tensor]
            Handle for assign operation.

        """

        if not dependencies:
            dependencies = []

        with tf.control_dependencies(dependencies):

            # check whether val needs to be solved first
            if solve:
                val = self._solve(val)

            if add:
                try:
                    return var.assign_add(val)
                except ValueError:
                    try:
                        return var[:, 0].assign(var[:, 0] + val)
                    except ValueError:
                        return var.assign_add(tf.squeeze(val))
            else:
                if hasattr(var, 'shape') and not hasattr(val, 'shape'):
                    return var.assign(tf.zeros(var.shape) + val)
                else:
                    try:
                        return var.assign(val)
                    except ValueError:
                        try:
                            return var[:, 0].assign(val)
                        except ValueError:
                            return var.assign(tf.squeeze(val))

    def _solve(self, rhs):
        """Solves right-hand side of a differential equation.
        """

        return rhs * self.dt


class Network(MultiDiGraph):
    """Creates an RNN cell that contains all nodes in the network plus their recurrent connections.

    Parameters
    ----------
    net_config
    name

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self,
                 net_config: MultiDiGraph,
                 dt: float=1e-3,
                 vectorize: str='nodes',
                 name: Optional[str]=None,
                 build_in_place=True):
        """Instantiates operator.
        """

        super().__init__(name=name)

        # further basic attributes
        self.backend = TensorflowBackend()
        self._net_config_map = {}
        self.dt = dt

        # process the network configuration
        self.net_config = self._preprocess_net_config(net_config) if build_in_place \
            else self._preprocess_net_config(copy(net_config))
        self.net_config = self._vectorize(net_config=copy(self.net_config), vectorization_mode=vectorize)

        # parse node operations
        #######################

        self.node_updates = []

        for node_name, node in net_config.nodes.items():

            # check operators for cyclic relationships
            op_graph = node['node'].op_graph
            #try:
            #    find_cycle(op_graph)
            #except NetworkXNoCycle:
            #    pass
            #else:
            #    raise PyRatesException(
            #        "Found cyclic operator graph. Cycles are not allowed for operators within one node.")

            graph = op_graph.copy()

            # first, parse operators that have no dependencies on other operators
            # noinspection PyTypeChecker
            primary_ops = [op for op, in_degree in graph.in_degree if in_degree == 0]
            self._add_ops(primary_ops, True, node_name)

            # remove parsed operators from graph
            graph.remove_nodes_from(primary_ops)

            # now, pass all other operators on the node
            while graph.nodes:

                # get all operators that have no dependencies on other operators
                # noinspection PyTypeChecker
                secondary_ops = [op for op, in_degree in graph.in_degree if in_degree == 0]
                self._add_ops(secondary_ops, False, node_name)

                # remove parsed operators from graph
                graph.remove_nodes_from(secondary_ops)

        self.node_updates = tf.group(self.node_updates)

        # parse edges
        #############

        self.edge_updates = []
        for source_node, target_node, edge_idx in net_config.edges:

            # extract edge information
            weight = self._get_edge_attr(source_node, target_node, edge_idx, 'weight')
            delay = self._get_edge_attr(source_node, target_node, edge_idx, 'delay')
            svar = self._get_edge_attr(source_node, target_node, edge_idx, 'source_var')
            tvar = self._get_edge_attr(source_node, target_node, edge_idx, 'target_var')
            sidx = self._get_edge_attr(source_node, target_node, edge_idx, 'source_idx')
            tidx = self._get_edge_attr(source_node, target_node, edge_idx, 'target_idx')

            # define target index
            if delay:
                if type(delay) is list:
                    if 'float' in str(type(delay[0])):
                        delay = [int(d / self.dt) + 1 for d in delay]
                if 'float' in str(type(delay)):
                    delay = int(delay / self.dt) + 1
            if delay and tidx:
                tidx = [idx + [d] for idx, d in zip(tidx, delay)]

            # create mapping equation and its arguments
            d = "[target_idx]" if tidx else ""
            idx = "[source_idx]" if sidx else ""
            eq = f"target_var{d} = source_var{idx} * weight"
            args = {'vars': {}, 'inputs': {}}
            args['vars']['weight'] = {'vtype': 'constant', 'dtype': svar.dtype, 'value': weight}
            if tidx:
                args['vars']['target_idx'] = {'vtype': 'constant', 'dtype': 'int32', 'value': tidx}
            if sidx:
                args['vars']['source_idx'] = {'vtype': 'constant', 'dtype': 'int32', 'value': sidx}
            args['vars']['target_var'] = {'vtype': 'state_var', 'dtype': tvar.dtype, 'value': np.zeros(tvar.shape)}
            args['inputs']['source_var'] = svar

            # parse mapping
            args = parse_equation(eq, args, backend=self.backend, scope=(source_node, target_node, edge_idx))
            args.pop('lhs_evals')

            # store information in network config
            edge = self.net_config.edges[source_node, target_node, edge_idx]

            # update edge attributes
            for fields in args.values():
                edge.update(fields)

            # add projection to edge updates
            self.edge_updates.append(edge['target_var'])

        # add differential equation updates
        for node in self.net_config.nodes:
            op_graph = self._get_node_attr(node, 'op_graph')
            for op in op_graph.nodes.values():
                for key, var in op['variables'].items():
                    if '_old' in key:
                        key_tmp = key.replace('_old', '')
                        val = op['variables'][key_tmp]
                        self.edge_updates.append(
                            tf.keras.layers.Lambda(lambda x: tf.keras.backend.update(x[0], x[1]))([var, val]))

        self.edge_updates = tf.group(self.edge_updates)

    def run(self,
            simulation_time: Optional[float] = None,
            inputs: Optional[dict] = None,
            outputs: Optional[dict] = None,
            sampling_step_size: Optional[float] = None,
            out_dir: Optional[str] = None,
            verbose: bool=True,
            ) -> Tuple[DataFrame, float]:
        """Simulate the backend behavior over time via a tensorflow session.

        Parameters
        ----------
        simulation_time
            Simulation time in seconds.
        inputs
            Inputs for placeholder variables. Each key is a tuple that specifies a placeholder variable in the graph
            in the following format: (node_name, op_name, var_name). Each value is an array that defines the input for
            the placeholder variable over time (first dimension).
        outputs
            Output variables that will be returned. Each key is the desired name of an output variable and each value is
            a tuple that specifies a variable in the graph in the following format: (node_name, op_name, var_name).
        sampling_step_size
            Time in seconds between sampling points of the output variables.
        out_dir
            Directory in which to store outputs.
        verbose
            If true, status updates will be printed to the console.

        Returns
        -------
        tuple
            First entry of the tuple contains the output variables in a pandas dataframe, the second contains the
            simulation time in seconds.

        """

        # prepare simulation
        ####################

        # basic simulation parameters initialization
        if not simulation_time:
            simulation_time = self.dt
        sim_steps = int(simulation_time / self.dt)

        if not sampling_step_size:
            sampling_step_size = self.dt
        sampling_steps = int(sampling_step_size / self.dt)

        # define output variables
        outputs_tmp = dict()
        if outputs:

            # go through  output variables
            for key, val in outputs.items():

                if val[0] == 'all':

                    # collect output variable from every node in backend
                    for node in self.net_config.nodes.keys():
                        outputs_tmp[f'{node}/{key}'] = self._get_node_attr(node=node, op=val[1], attr=val[2])

                elif val[0] in self.net_config.nodes.keys() or val[0] in self._net_config_map.keys():

                    # get output variable of specific backend node
                    outputs_tmp[key] = self._get_node_attr(node=val[0], op=val[1], attr=val[2])

                elif any([val[0] in key for key in self.net_config.nodes.keys()]):

                    # get output variable from backend nodes of a certain type
                    for node in self.net_config.nodes.keys():
                        if val[0] in node:
                            outputs_tmp[f'{key}/{node}'] = self._get_node_attr(node=node, op=val[1], attr=val[2])
                else:

                    # get output variable of specific, vectorized  backend node
                    for node in self._net_config_map.keys():
                        if val[0] in node and 'comb' not in node:
                            outputs_tmp[f'{key}/{node}'] = self._get_node_attr(node=node, op=val[1], attr=val[2])

        # add output collector variables to graph
        output_col = {}
        store_ops = []

        # create counting index for collector variables
        out_idx = self.backend.add_variable(name='out_var_idx', dtype=tf.int32, shape=(), value=0,
                                            scope="output_collection")

        # add collector variables to the graph
        for key, var in outputs_tmp.items():
            shape = [int(sim_steps / sampling_steps) + 1] + list(var.shape)
            output_col[key] = self.backend.add_variable(name=key, dtype=tf.float32, shape=shape, value=np.zeros(shape),
                                                        scope="output_collection")

            # add collect operation to the graph
            store_ops.append(self.backend.add_op('scatter_update', output_col[key], out_idx, var))

        store_output = tf.group(store_ops)

        # create increment operator for counting index
        out_idx_incr = self.backend.add_op('+=', out_idx, 1)

        # linearize input dictionary
        if inputs:
            inp = list()
            for step in range(sim_steps):
                inp_dict = dict()
                for key, val in inputs.items():

                    if self.name in self.net_config.nodes.keys():

                        # fully vectorized case: add vectorized placeholder variable to input dictionary
                        var = self._get_node_attr(node=self.name, op=key[1], attr=key[2])
                        inp_dict[var] = np.reshape(val[step], var.shape)

                    elif any(['comb' in key_tmp for key_tmp in self.net_config.nodes.keys()]):

                        # node-vectorized case
                        if key[0] == 'all':

                            # go through all nodes, extract the variable and add it to input dict
                            for i, node in enumerate(self.net_config.nodes.keys()):
                                var = self._get_node_attr(node=node, op=key[1], attr=key[2])
                                inp_dict[var] = np.reshape(val[step, i], var.shape)

                        elif key[0] in self.net_config.nodes.keys():

                            # add placeholder variable of node(s) to input dictionary
                            var = self._get_node_attr(node=key[0], op=key[1], attr=key[2])
                            inp_dict[var] = np.reshape(val[step], var.shape)

                        elif any([key[0] in key_tmp for key_tmp in self.net_config.nodes.keys()]):

                            # add vectorized placeholder variable of specified node type to input dictionary
                            for node in list(self.net_config.nodes.keys()):
                                if key[0] in node:
                                    break
                            var = self._get_node_attr(node=node, op=key[1], attr=key[2])
                            inp_dict[var] = np.reshape(val[step], var.shape)

                    else:

                        # non-vectorized case
                        if key[0] == 'all':

                            # go through all nodes, extract the variable and add it to input dict
                            for i, node in enumerate(self.net_config.nodes.keys()):
                                var = self._get_node_attr(node=node, op=key[1], attr=key[2])
                                inp_dict[var] = np.reshape(val[step, i], var.shape)

                        elif any([key[0] in key_tmp for key_tmp in self.net_config.nodes.keys()]):

                            # extract variables from nodes of specified type
                            i = 0
                            for node in self.net_config.nodes.keys():
                                if key[0] in node:
                                    var = self._get_node_attr(node=node, op=key[1], attr=key[2])
                                    inp_dict[var] = np.reshape(val[step, i], var.shape)
                                    i += 1

                # add input dictionary to placeholder input liust
                inp.append(inp_dict)

        else:

            # create list of nones (no placeholder variables should exist)
            inp = [None for _ in range(sim_steps)]

        # run simulation
        ################

        output_col, sim_time = self.backend.run(steps=sim_steps, ops=[self.node_updates, self.edge_updates], inputs=inp,
                                                outputs=output_col, sampling_steps=sampling_steps,
                                                sampling_ops=[store_output, out_idx_incr], out_dir=out_dir)

        # store output variables in data frame
        for i, (key, var) in enumerate(output_col.items()):
            if i == 0:
                var = np.squeeze(var)
                try:
                    out_vars = DataFrame(data=var,
                                         columns=[f'{key}_{j}' for j in range(var.shape[1])])
                except IndexError:
                    out_vars = DataFrame(data=var, columns=[key])
            else:
                var = np.squeeze(var)
                try:
                    for j in range(var.shape[1]):
                        out_vars[f'{key}_{j}'] = var[:, j]
                except IndexError:
                    out_vars[key] = var
        if not outputs:
            out_vars = DataFrame()
        else:
            out_vars['time'] = np.arange(0., simulation_time + sampling_step_size * 0.5, sampling_step_size)

        # display simulation time
        if verbose:
            if simulation_time:
                print(f"{simulation_time}s of backend behavior were simulated in {sim_time} s given a "
                      f"simulation resolution of {self.dt} s.")
            else:
                print(f"ComputeGraph computations finished after {sim_time} seconds.")

        return out_vars, sim_time

    def _add_ops(self, ops, primary_ops, node_name):
        """

        Parameters
        ----------
        ops
        primary_ops
        node_name

        Returns
        -------

        """

        for op_name in ops:

            # retrieve operator and operator args
            op_args = dict()
            op_args['vars'] = self._get_op_attr(node_name, op_name, 'variables')
            op_args['vars']['dt'] = {'vtype': 'constant', 'dtype': 'float32', 'shape': (1,), 'value': self.dt}
            op_args['inputs'] = {}
            op_info = self._get_op_attr(node_name, op_name, 'operator')

            # handle operator inputs
            for var_name, inp in op_info['inputs'].items():

                # go through inputs to variable
                if inp['sources']:
                    in_ops_col = []
                    i = 0
                    for in_op in inp['sources']:

                        if type(in_op) is list and len(in_op) == 1:

                            if primary_ops:
                                raise ValueError(f'Input found in primary operator {op_name} on node {node_name}. '
                                                 f'This operator should have no node-internal inputs. '
                                                 f'Please move the operator inputs to the node level or change the '
                                                 f'input-output relationships between the node operators.')
                            else:
                                var = self._get_op_attr(node_name, in_op[0], 'output')
                                if type(var) is str:
                                    raise ValueError(f'Wrong operator order on node {node_name}. Operator {op_name} '
                                                     f'needs input from operator {in_op[0]} which has not been '
                                                     f'processed yet. Please consider changing the operator order '
                                                     f'or dependencies.')
                            in_ops_col.append(var)
                            i += 1

                        elif type(in_op) is tuple:

                            in_ops = in_op[0]
                            reduce_dim = in_op[1]

                            in_ops_tmp = []
                            for op in in_ops:
                                if primary_ops:
                                    raise ValueError(f'Input found in primary operator {op_name} on node {node_name}. '
                                                     f'This operator should have no node-internal inputs. '
                                                     f'Please move the operator inputs to the node level or change the '
                                                     f'input-output relationships between the node operators.')
                                else:
                                    var = self._get_op_attr(node_name, in_op[0], 'output')
                                    if type(var) is str:
                                        raise ValueError(
                                            f'Wrong operator order on node {node_name}. Operator {op_name} '
                                            f'needs input from operator {op} which has not been '
                                            f'processed yet. Please consider changing the operator order '
                                            f'or dependencies.')
                                in_ops_tmp.append(var)
                                i += 1

                            in_ops_col.append(self._map_multiple_inputs(in_ops_tmp, reduce_dim))

                        else:

                            if primary_ops:
                                raise ValueError(f'Input found in primary operator {op_name} on node {node_name}. '
                                                 f'This operator should have no node-internal inputs. '
                                                 f'Please move the operator inputs to the node level or change the '
                                                 f'input-output relationships between the node operators.')
                            else:
                                var = self._get_op_attr(node_name, in_op, 'output')
                                if type(var) is str:
                                    raise ValueError(f'Wrong operator order on node {node_name}. Operator {op_name} '
                                                     f'needs input from operator {in_op[0]} which has not been '
                                                     f'processed yet. Please consider changing the operator order '
                                                     f'or dependencies.')
                            in_ops_col.append(var)
                            i += 1

                    # for multiple multiple input operations
                    if len(in_ops_col) > 1:

                        # find shape of smallest input variable
                        min_shape = min([op.shape[0] if len(op.shape) > 0 else 0 for op in in_ops_col])

                        # append input variables to list and reshape them if necessary
                        in_ops = []
                        for op in in_ops_col:
                            shape = op.shape[0] if len(op.shape) > 0 else 0
                            if shape > min_shape:
                                if shape % min_shape != 0:
                                    raise ValueError(f"Shapes of inputs do not match: "
                                                     f"{inp['sources']} cannot be stacked.")
                                multiplier = shape // min_shape
                                for j in range(multiplier):
                                    in_ops.append(op[j * min_shape:(j + 1) * min_shape])
                            else:
                                in_ops.append(op)

                        # map inputs to target
                        in_ops = self._map_multiple_inputs(in_ops, inp['reduce_dim'])

                    # for a single input variable
                    else:
                        in_ops = in_ops_col[0]

                    # add input variable to dictionary
                    op_args['inputs'][var_name] = in_ops

            # parse equations into tensorflow
            for eq in op_info['equations']:
                op_args = parse_equation(eq, op_args, backend=self.backend, scope=(node_name, op_name))

            # store operator variables in net config
            op_vars = self._get_op_attr(node_name, op_name, 'variables')
            op_vars.update(op_args['inputs'])
            op_vars.update(op_args['vars'])
            op_vars.update(op_args['updates'])

            # if the operator does not project to others, store its update operations
            if self._get_node_attr(node_name, 'op_graph').out_degree(op_name) == 0:
                for var_update in op_args['lhs_evals']:
                    self.node_updates.append(op_args['updates'][var_update])

    def _map_multiple_inputs(self, inputs, reduce_dim):
        """
        """

        inp = self.backend.add_op('stack', inputs)
        if reduce_dim:
            return self.backend.add_op('sum', inp, axis=0)
        else:
            return self.backend.add_op('reshape', inp, shape=(inp.shape[0] * inp.shape[1],))

    def _connect_edge_to_op(self, in_op, node, op, var, n):
        """
        """

        # create input tensor
        source, target, edge = in_op
        var_info = self._get_edge_attr(source, target, edge, 'source_var')
        inp_tensor = tf.keras.layers.Input(shape=var_info['shape'], dtype=var_info['dtype'])
        self.inputs[f'{node}/{op}/{var}.{n}'] = inp_tensor

        # update edge to map to input tensor
        self._set_edge_attr(source, target, edge, 'target_var',
                            f'{op}/{var}.{n}')

        return inp_tensor

    def _get_node_attr(self, node, attr, op=None, net_config=None):
        """

        Parameters
        ----------
        node
        attr
        op
        net_config

        Returns
        -------

        """

        if not net_config:
            net_config = self.net_config

        if op:
            return self._get_op_attr(node, op, attr, net_config=net_config)
        try:
            return net_config.nodes[node]['node'][attr]
        except KeyError:
            vals = []
            for op in net_config.nodes[node]['node']['op_graph'].nodes:
                vals.append(self._get_op_attr(node, op, attr, net_config=net_config))
            return vals

    def _set_node_attr(self, node, attr, val, op=None, net_config=None):
        """

        Parameters
        ----------
        node
        attr
        op
        net_config

        Returns
        -------

        """

        if not net_config:
            net_config = self.net_config

        if op:
            return self._set_op_attr(node, op, attr, val, net_config=net_config)

        try:
            net_config.nodes[node]['node'][attr] = val
        except KeyError:
            vals_updated = []
            for op in net_config.nodes[node]['node']['op_graph'].nodes:
                vals_updated.append(self._set_op_attr(node, op, attr, val, net_config=net_config))
            return vals_updated

    def _get_edge_attr(self, source, target, edge, attr, net_config=None):
        """

        Parameters
        ----------
        source
        target
        edge
        attr
        net_config

        Returns
        -------

        """

        if not net_config:
            net_config = self.net_config

        try:
            attr_val = net_config.edges[source, target, edge][attr]
            if 'var' in attr and type(attr_val) is str:
                op, var = attr_val.split('/')
                if 'source' in attr:
                    attr_val = self._get_op_attr(source, op, var, net_config=net_config)
                else:
                    attr_val = self._get_op_attr(target, op, var, net_config=net_config)
        except KeyError:
            attr_val = None

        return attr_val

    def _set_edge_attr(self, source, target, edge, attr, val, net_config=None):
        """

        Parameters
        ----------
        source
        target
        edge
        attr
        val
        net_config

        Returns
        -------

        """

        if not net_config:
            net_config = self.net_config

        net_config.edges[source, target, edge][attr] = val
        return net_config.edges[source, target, edge][attr]

    def _get_op_attr(self, node, op, attr, net_config=None):
        """

        Parameters
        ----------
        node
        op
        attr
        net_config

        Returns
        -------

        """

        if not net_config:
            net_config = self.net_config

        op = net_config.nodes[node]['node'].op_graph.nodes[op]
        if attr in op['variables'].keys():
            return op['variables'][attr]
        elif attr == 'output':
            return op['variables'][op['operator']['output']]
        elif hasattr(op['operator'], attr) or (hasattr(op['operator'], 'keys') and attr in op['operator'].keys()):
            return op['operator'][attr]
        else:
            try:
                return op[attr]
            except KeyError:
                return None

    def _set_op_attr(self, node, op, attr, val, net_config=None):
        """

        Parameters
        ----------
        node
        op
        attr
        val

        Returns
        -------

        """

        if not net_config:
            net_config = self.net_config

        op = net_config.nodes[node]['node'].op_graph.nodes[op]
        if attr in op['variables'].keys():
            op['variables'][attr] = val
            return op['variables'][attr]
        elif attr == 'output':
            op['variables'][op[attr]] = val
            return op['variables'][op[attr]]
        elif hasattr(op['operator'], attr):
            setattr(op['operator'], attr, val)
            return getattr(op['operator'], attr)
        else:
            try:
                op[attr] = val
                return op[attr]
            except KeyError:
                return None

    def _get_nodes_with_attr(self, attr, val, net_config=None):
        """

        Parameters
        ----------
        attr
        val

        Returns
        -------

        """

        if not net_config:
            net_config = self.net_config

        return [node for node in net_config.nodes.keys() if self._get_node_attr(node, attr,
                                                                                net_config=net_config) == val]

    def _contains_node(self, node, target_node):
        """

        Parameters
        ----------
        node
        target_node

        Returns
        -------

        """

        try:
            node_map = self._net_config_map[target_node]
            for op in node_map.values():
                for var in op.values():
                    if var[0] == node:
                        return True
            else:
                return False
        except KeyError:
            raise KeyError(f'Could not identify target node {target_node}.')

    def _get_edges_between(self, source, target, net_config=None):
        """

        Parameters
        ----------
        source
        target

        Returns
        -------

        """

        if not net_config:
            net_config = self.net_config

        try:
            return [net_config.edges[source, target, edge]
                    for edge in range(net_config.graph.number_of_edges(source, target))]
        except KeyError:
            edges = []
            for source_tmp in self._net_config_map.keys():
                for target_tmp in self._net_config_map.keys():
                    if self._contains_node(source, source_tmp) and self._contains_node(target, target_tmp):
                        edges += [net_config.edges[source_tmp, target_tmp, edge]
                                  for edge in range(net_config.graph.number_of_edges(source_tmp, target_tmp))]
            return edges

    def _get_edge_conn(self, source, target, edge, net_config=None):
        """

        Parameters
        ----------
        source
        target
        edge
        net_config

        Returns
        -------

        """

        if not net_config:
            net_config = self.net_config

        # check delay of edge
        delay = self._get_edge_attr(source, target, edge, 'delay', net_config=net_config)
        if delay.ndim > 1:
            raise ValueError(f"Automatic optimization of the graph (i.e. method `vectorize`"
                             f" cannot be applied to networks with variables of 2 or more"
                             f" dimensions. Delay of edge {edge} between {source} and {target} has shape"
                             f" {delay.shape}. Please turn of the `vectorize` option or change"
                             f" the edges' dimensionality.")

        # check weight of edge
        weight = self._get_edge_attr(source, target, edge, 'delay', net_config=net_config)
        if weight.ndim > 1:
            raise ValueError(f"Automatic optimization of the graph (i.e. method `vectorize`"
                             f" cannot be applied to networks with variables of 2 or more"
                             f" dimensions. Weight of edge {edge} between {source} and {target} has shape"
                             f" {weight.shape}. Please turn of the `vectorize` option or"
                             f" change the edges' dimensionality.")

        # match dimensionality of delay and weight
        if not delay or not weight:
            pass
        elif delay.shape[0] > 1 and weight.shape[0] == 1:
            weight = np.tile(weight, (delay.shape[0], 1))
        elif weight.shape[0] > 1 and delay.shape[0] == 1:
            delay = np.tile(delay, (weight.shape[0], 1))
        elif delay.shape != weight.shape:
            raise ValueError(f"Dimensionality of weights and delays of edge between {source} and {target}"
                             f" do not match. They are of shape {weight.shape} and {delay.shape}."
                             f" Please turn of the `vectorize` option or change the dimensionality of the"
                             f" edge's attributes.")

        return weight, delay

    def _get_edge_var_idx(self, source, target, edge, idx_type, net_config=None):
        """

        Parameters
        ----------
        source
        target
        edge
        net_config

        Returns
        -------

        """

        if not net_config:
            net_config = self.net_config

        if idx_type == 'source':
            var = 'source_idx'
            node_to_idx = source
        elif idx_type == 'target':
            var = 'target_idx'
            node_to_idx = target
        else:
            raise ValueError('Wrong `idx_type`. Please, choose either `source` or `target`.')

        var_idx = self._get_edge_attr(source, target, edge, var, net_config=net_config)
        if var_idx:
            op, var = self.net_config.edges[source, target, edge][var].split('/')
            _, _, _, idx = self._net_config_map[node_to_idx][op][var]
            if type(idx) is tuple:
                idx = range(idx[0], idx[1])
                idx_new = []
                for i in var_idx:
                    if type(i) is tuple:
                        idx_new.append([idx[i[0]:i[1]]])
                    elif type(i) is list:
                        idx_new.append([idx[i[0]]])
                    else:
                        idx_new.append([idx[i]])
            else:
                idx_new = [[idx]]
        else:
            op, var = self.net_config.edges[source, target, edge][var].split('/')
            _, _, _, idx_new = self._net_config_map[node_to_idx][op][var]
            if type(idx_new) is int:
                idx_new = [idx_new]
            else:
                idx_new = list(idx_new)

        return idx_new

    def _sort_edges(self, edges, attr, net_config=None):
        """

        Parameters
        ----------
        edges
        attr
        net_config

        Returns
        -------

        """

        if not net_config:
            net_config = self.net_config

        edges_new = {}
        for edge in edges:
            if len(edge) == 3:
                source, target, edge = edge
            else:
                source, target = edge
                edge = 0
            edge_info = net_config.edges[source, target, edge]
            if edge_info[attr] not in edges_new.keys():
                edges_new[edge_info[attr]] = [(source, target, edge)]
            else:
                edges_new[edge_info[attr]].append((source, target, edge))

        return edges_new

    def _add_synaptic_buffer(self, node, op, var, idx, buffer_length, edge, net_config=None):
        """

        Parameters
        ----------
        node
        op
        var
        idx
        buffer_length
        edge
        net_config

        Returns
        -------

        """

        if not net_config:
            net_config = self.net_config
        target_shape = self._get_op_attr(node, op, var, net_config=net_config)['shape']
        op_graph = self._get_node_attr(node, 'op_graph', net_config=net_config)

        # create buffer equations
        if len(target_shape) < 1:
            eqs_op_read = [f"{var} = {var}_buffer_{idx}[0]"]
            eqs_op_rotate = [f"{var}_buffer_{idx}_tmp = {var}_buffer_{idx}[1:]",
                             f"{var}_buffer_{idx}[0:-1] = {var}_buffer_{idx}_tmp",
                             f"{var}_buffer_{idx}[-1] = 0."]
        else:
            eqs_op_read = [f"{var} = {var}_buffer_{idx}[:, 0]"]
            eqs_op_rotate = [f"{var}_buffer_{idx}_tmp = {var}_buffer_{idx}[:, 1:]",
                             f"{var}_buffer_{idx}[:, 0:-1] = {var}_buffer_{idx}_tmp",
                             f"{var}_buffer_{idx}[:, -1] = 0."]

        # create buffer variable definitions
        if 'float' in str(type(buffer_length)):
            buffer_length = int(buffer_length / self.dt) + 1
        if len(target_shape) > 0:
            buffer_shape = [target_shape[0], buffer_length + 1]
        else:
            buffer_shape = [buffer_length + 1]
        var_dict = {f'{var}_buffer_{idx}': {'vtype': 'state_var',
                                            'dtype': 'float32',
                                            'shape': buffer_shape,
                                            'value': 0.
                                            }}

        # add buffer operators to operator graph
        op_graph.add_node(f'{op}_{var}_buffer_rotate_{idx}',
                          operator={'inputs': {},
                                    'output': f'{var}_buffer_{idx}',
                                    'equations': eqs_op_rotate},
                          variables=var_dict)
        op_graph.add_node(f'{op}_{var}_buffer_read_{idx}',
                          operator={'inputs': {var: {'sources': [f'{op}_{var}_buffer_rotate_{idx}'],
                                                     'reduce_dim': False}},
                                    'output': var,
                                    'equations': eqs_op_read},
                          variables={})

        # connect operators to rest of the graph
        op_graph.add_edge(f'{op}_{var}_buffer_rotate_{idx}', f'{op}_{var}_buffer_read_{idx}')
        op_graph.add_edge(f'{op}_{var}_buffer_read_{idx}', op)

        # add input information to target operator
        inputs = self._get_op_attr(node, op, 'inputs', net_config=net_config)
        if var in inputs.keys():
            inputs[var]['sources'].append(f'{op}_{var}_buffer_read_{idx}')
        else:
            inputs[var] = {'sources': [f'{op}_{var}_buffer_read_{idx}'],
                           'reduce_dim': True}

        # update edge target information
        s, t, e = edge
        self._set_edge_attr(s, t, e, 'target_var', f'{op}_{var}_buffer_rotate_{idx}/{var}_buffer_{idx}',
                            net_config=net_config)

    def _add_synaptic_input_collector(self, node, op, var, idx, edge, net_config=None):
        """

        Parameters
        ----------
        node
        op
        var
        idx
        edge
        net_config

        Returns
        -------

        """

        if not net_config:
            net_config = self.net_config
        target_shape = self._get_op_attr(node, op, var, net_config=net_config)['shape']
        op_graph = self._get_node_attr(node, 'op_graph', net_config=net_config)

        # create collector equation
        eqs = [f"{var} = {var}_col_{idx}"]

        # create collector variable definition
        var_dict = {f'{op}_{var}_col_{idx}/{var}_col_{idx}': {'vtype': 'state_var',
                                                              'dtype': 'float32',
                                                              'shape': target_shape,
                                                              'value': 0.
                                                              }}

        # add collector operator to operator graph
        op_graph.add_node(f'{op}_{var}_col_{idx}',
                          operator={'inputs': {},
                                    'output': var,
                                    'equations': eqs},
                          variables=var_dict)
        op_graph.add_edge(f'{op}_{var}_col_{idx}', op)

        # add input information to target operator
        if var in op_graph.nodes[op]['inputs'].keys():
            op_graph.nodes[op]['inputs']['sources'].append(f'{op}_{var}_col_{idx}')
        else:
            op_graph.nodes[op]['inputs'][var] = {'sources': [f'{op}_{var}_col_{idx}'],
                                                 'reduce_dim': True}

        # update edge target information
        s, t, e = edge
        self._set_edge_attr(s, t, e, 'target_var', f'{op}_{var}_col_{idx}/{var}_col_{idx}', net_config=net_config)

    def _preprocess_net_config(self, net_config: MultiDiGraph) -> MultiDiGraph:
        """

        Parameters
        ----------
        net_config

        Returns
        -------
        MultiDiGraph

        """

        # check node attributes
        #######################

        # go through each node in the network  config
        for node_name, node in net_config.nodes.items():

            # check whether an operation graph exists
            try:
                op_graph = node['node'].op_graph
            except KeyError:
                raise KeyError(f'Key `node` not found on node {node_name}. Every node in the network configuration '
                               f'needs a field with the key `node` under which '
                               f'its operation graph and its template is stored.')
            except AttributeError:
                raise AttributeError(f'Attribute `op_graph` not found on node {node_name}. Every node in the network '
                                     f'configuration needs a graph object stored under its `node` key, which contains '
                                     f'all information regarding its operators, variables,'
                                     f'and input-output relationships.')

            # go through the operations on the node
            for op_name, op in op_graph.nodes.items():

                # check whether the variable field exists on the operator
                try:
                    variables = op['variables']
                except KeyError:
                    raise KeyError(f'Key `variables` not found in operator {op_name} of node {node_name}. Every '
                                   f'operator on a node needs a field `variables` under which all necessary '
                                   'variables are defined via key-value pairs.')

                # check whether the operator field exists on the operator
                try:
                    op_info = op['operator']
                except KeyError:
                    raise KeyError(f'Key `operator` not found in operator {op_name} of node {node_name}. Every '
                                   f'operator on a node needs a field `operator` under which all the equations, '
                                   'inputs and output are defined.')

                # go through the variables
                for var_name, var in variables.items():

                    # check definition of each variable
                    var_def = {'state_var': {'value': np.zeros((1,), dtype=np.float32),
                                             'dtype': 'float32',
                                             'shape': (1,),
                                             },
                               'constant': {'value': KeyError,
                                            'shape': (1,),
                                            'dtype': 'float32'},
                               'placeholder': {'dtype': 'float32',
                                               'shape': (1,),
                                               'value': None},
                               'raw': {'dtype': None,
                                       'shape': None,
                                       'value': KeyError}
                               }
                    try:
                        vtype = var['vtype']
                    except KeyError:
                        raise KeyError(f'Field `vtype` not found in variable definition of variable {var_name} of '
                                       f'operator {op_name} on node {node_name}. Each variable needs a field `vtype`'
                                       f'that indicates whether the variable is a `state_var`, `constant` or '
                                       f'`placeholder`.')
                    var_defs = var_def[vtype]

                    # check the value of the variable
                    try:
                        val = var['value']
                    except KeyError:
                        if var_defs['value'] is KeyError:
                            raise KeyError(f'Field `value` not found in variable definition of variable {var_name} of '
                                           f'operator {op_name} on node {node_name}, but it is needed for variables of '
                                           f'type {vtype}.')
                        elif var_defs['value'] is not None:
                            var['value'] = var_defs['value']

                    # check the shape of the variable
                    try:
                        val = var['shape']
                    except KeyError:
                        var['shape'] = var_defs['shape']

                    # check the data type of the variable
                    try:
                        val = var['dtype']
                    except KeyError:
                        var['dtype'] = var_defs['dtype']

                    if 'value' in var.keys() and not hasattr(var['value'], 'shape'):
                        var['value'] = np.zeros(var['shape'], dtype=var['dtype']) + var['value']

                # check whether the equations, inputs and output fields exist on the operator field
                op_fields = ['equations', 'inputs', 'output']
                for field in op_fields:
                    try:
                        attr = getattr(op_info, field)
                    except KeyError:
                        if field == 'equations':
                            raise KeyError(f'Field {field} not found in `operators` field of operator {op_name} on '
                                           f'node {node_name}. Each operator should follow a list of equations that '
                                           f'needs to be provided at this position.')
                        elif field == 'inputs':
                            op_info['inputs'] = {}
                        else:
                            raise KeyError(f'Field {field} not found in `operators` field of operator {op_name} on '
                                           f'node {node_name}. Each operator should have an output, the name of which '
                                           f'needs to be provided at this position.')

        # check edge attributes
        #######################

        # go through edges
        for source, target, idx in net_config.edges:

            edge = net_config.edges[source, target, idx]

            # check whether source and target variable information were provided
            try:
                svar = edge['source_var']
            except KeyError:
                raise KeyError(f'Field `source_var` not found on edge {idx} between {source} and {target}. Every edge '
                               f'needs information about the variable on the source node it should access that needs '
                               f'to be provided at this position.')
            try:
                tvar = edge['target_var']
            except KeyError:
                raise KeyError(f'Field `target_var` not found on edge {idx} between {source} and {target}. Every edge '
                               f'needs information about the variable on the target node it should access that needs '
                               f'to be provided at this position.')

            # check weight of edge
            try:
                weight = edge['weight']
                if not hasattr(weight, 'shape'):
                    weight = np.ones((1,), dtype=np.float32) * weight
                elif 'float' not in str(weight.dtype):
                    weight = np.asarray(weight, dtype=np.float32)
                edge['weight'] = weight
            except KeyError:
                edge['weight'] = np.ones((1,), dtype=np.float32)

            # check delay of edge
            try:
                delay = edge['delay']
                if delay is not None:
                    if not hasattr(delay, 'shape'):
                        delay = np.zeros((1,)) + delay
                    if 'float' in str(delay.dtype):
                        delay = np.asarray(delay / self.dt, dtype=np.int32)
                    edge['delay'] = delay
            except KeyError:
                edge['delay'] = None

        return net_config

    def _vectorize(self, net_config: MultiDiGraph, vectorization_mode: Optional[str] = 'nodes') -> MultiDiGraph:
        """Method that goes through the nodes and edges dicts and vectorizes those that are governed by the same
        operators/equations.

        Parameters
        ----------
        net_config
            See argument description at class level.
        vectorization_mode
            Can be 'none' for no vectorization, 'nodes' for vectorization over nodes and 'ops' for vectorization over
            nodes and operations.

        Returns
        -------
        Tuple[NodeView, EdgeView]

        """

        # First stage: Vectorize over nodes
        ###################################

        if vectorization_mode in 'nodesfull':

            nodes = list(net_config.nodes.keys())

            # go through each node in net config and vectorize it with nodes that have the same operator structure
            ######################################################################################################

            while nodes:

                # get nodes with same operators
                op_graph = self._get_node_attr(nodes[0], 'op_graph', net_config=net_config)
                nodes_tmp = self._get_nodes_with_attr('op_graph', op_graph, net_config=net_config)

                # vectorize those nodes
                _ = self._vectorize_nodes(nodes_tmp.copy(), net_config)

                # delete vectorized nodes from graph and list
                for n in nodes_tmp:
                    net_config.graph.remove_node(n)
                    nodes.pop(nodes.index(n))

            # adjust edges accordingly
            ##########################

            # go through new nodes
            for source in net_config.nodes.keys():
                for target in net_config.nodes.keys():
                    _ = self._vectorize_edges(source, target, net_config)

            # save changes to net config
            self.net_config = copy(net_config)

        # Second stage: Vectorize over operators
        ########################################

        if vectorization_mode == 'full':

            # create dictionary of operators of each node that will be used to check whether they have been vectorized
            vec_info = {}
            for node in net_config.nodes:
                vec_info[node] = {}
                for op in self._get_node_attr(node, 'op_graph').nodes:
                    vec_info[op] = False

            # go through nodes and vectorize over their operators
            node_idx = 0
            node_key = list(vec_info.keys())[node_idx]
            while not all([all([vec for vec in node.values]) for node in vec_info.keys()]):

                change_node = False
                op_graph = self._get_node_attr(node_key, 'op_graph').copy()

                # vectorize nodes on the operator in their hierarchical order
                while op_graph.nodes:

                    # get all operators that have no dependencies on other operators on node
                    # noinspection PyTypeChecker
                    primary_ops = [op for op, in_degree in op_graph.in_degree if in_degree == 0]

                    for op_key in primary_ops:

                        if not vec_info[node_key][op_key]:

                            # check whether operator exists at other nodes
                            nodes_to_vec = []

                            for node_key_tmp in enumerate(vec_info.keys()):

                                if node_key_tmp != node_key and op_key in vec_info[node_key_tmp].keys():

                                    # check if operator dependencies are vectorized already
                                    deps_vectorized = True
                                    op_graph_tmp = self._get_node_attr(node_key_tmp, 'op_graph')
                                    for op_key_tmp in op_graph_tmp.nodes:
                                        if op_key_tmp != op_key:
                                            for edge in op_graph_tmp.in_edges(op_key_tmp):
                                                if not vec_info[node_key_tmp][edge[0]]:
                                                    deps_vectorized = False

                                    if deps_vectorized:
                                        nodes_to_vec.append(node_key_tmp)
                                    else:
                                        change_node = True
                                        break

                            if nodes_to_vec and not change_node:

                                # vectorize op
                                nodes_to_vec.append(node_key)
                                self._vectorize_ops(net_config=net_config,
                                                    op_key=op_key,
                                                    nodes=nodes_to_vec.copy())

                                # indicate where vectorization was performed
                                for node_key_tmp in nodes_to_vec:
                                    vec_info[node_key_tmp][op_key] = True

                            elif change_node:

                                break

                            else:

                                # add operation to new net configuration
                                self._vectorize_ops(net_config=net_config,
                                                    op_key=op_key,
                                                    nodes=[node_key])

                                # mark operation on node as checked
                                vec_info[node_key][op_key] = True

                    # remove parsed operators from graph
                    op_graph.remove_nodes_from(primary_ops)

                # increment node
                if not change_node:
                    node_idx = node_idx + 1 if node_idx < len(vec_info) - 1 else 0
                    node_key = list(vec_info.keys())[node_idx]

            # adjust edges accordingly
            ##########################

            _ = self._vectorize_edges(self.key, self.key, net_config)

        # Third Stage: Finalize edge vectorization
        ##########################################

        # go through nodes and create mapping for their inputs
        for node_name, node in net_config.nodes.items():

            node_inputs = net_config.graph.in_edges(node_name)
            node_inputs = self._sort_edges(node_inputs, 'target_var')

            # loop over input variables of node
            for i, (in_var, edges) in enumerate(node_inputs.items()):

                # extract info for input variable connections
                n_inputs = len(edges)
                op_name, var_name = in_var.split('/')
                max_delay = np.max([self._get_edge_attr(s, t, e, 'delay', net_config=net_config) for s, t, e in edges])

                # loop over different input sources
                for j in range(n_inputs):

                    if max_delay is not None:

                        # add synaptic buffer to the input variable
                        self._add_synaptic_buffer(node_name, op_name, var_name, idx=j, buffer_length=max_delay,
                                                  edge=edges[j], net_config=net_config)

                    elif n_inputs > 1:

                        # add synaptic input collector to the input variable
                        self._add_synaptic_input_collector(node_name, op_name, var_name, idx=j, edge=edges[j],
                                                           net_config=net_config)

        return net_config

    def _vectorize_nodes(self,
                         nodes: List[str],
                         net_config: MultiDiGraph
                         ) -> str:
        """Combines all nodes in list to a single node and adds node to new net config.

        Parameters
        ----------
        nodes
            Names of the nodes to be combined.
        net_config
            Networkx MultiDiGraph with nodes and edges as defined in init.

        Returns
        -------
        str
            name of the new node that was added to the net config.

        """

        n_nodes = len(nodes)

        # instantiate new node
        ######################

        node_ref = nodes.pop()
        new_node = net_config.nodes[node_ref]
        if node_ref not in self._net_config_map.keys():
            self._net_config_map[node_ref] = {}

        # define new node's name
        node_idx = 0
        node_ref_tmp = node_ref.split('/')[-1] if '/'in node_ref else node_ref
        node_ref_tmp = node_ref_tmp.split('.')[0] if '.' in node_ref_tmp else node_ref_tmp
        if all([node_ref_tmp in node_name for node_name in nodes]):
            node_name = f'{node_ref_tmp}_all'
        else:
            node_name = node_ref_tmp
            for n in nodes:
                n_tmp = n.split('/')[-1] if '/' in n else n
                n_tmp = n_tmp.split('.')[0] if '.' in n_tmp else n_tmp
                node_name += f'_{n_tmp}'
                test_names = []
                for test_name in nodes:
                    test_name_tmp = test_name.split('/')[-1] if '/' in test_name else test_name
                    test_name_tmp = test_name_tmp.split('.')[0] if '.' in test_name_tmp else test_name_tmp
                    test_names.append(test_name_tmp)
                if all([test_name in node_name for test_name in test_names]):
                    break
            node_name += '_all'

        while True:
            if f"{node_name}.{node_idx}" in net_config.nodes.keys():
                node_idx += 1
            else:
                node_name += f".{node_idx}"
                break

        # add new node to net config
        net_config.add_node(node_name, **new_node)

        # go through node attributes and store their value and shape
        arg_vals, arg_shapes = {}, {}
        for op_name in self._get_node_attr(node_ref, 'op_graph', net_config=net_config).nodes.keys():

            arg_vals[op_name] = {}
            arg_shapes[op_name] = {}
            if op_name not in self._net_config_map[node_ref].keys():
                self._net_config_map[node_ref][op_name] = {}

            for arg_name, arg in self._get_op_attr(node_ref, op_name, 'variables', net_config=net_config).items():

                # extract argument value and shape
                if arg['vtype'] == 'raw':
                    if type(arg['value']) is list or type(arg['value']) is tuple:
                        arg_vals[op_name][arg_name] = list(arg['value'])
                    else:
                        arg_vals[op_name][arg_name] = arg['value']
                else:
                    if len(arg['shape']) == 2:
                        raise ValueError(f"Automatic optimization of the graph (i.e. method `vectorize` cannot be "
                                         f"applied to networks with variables of 2 or more dimensions. Variable "
                                         f"{arg_name} has shape {arg['shape']}. Please turn of the `vectorize` option "
                                         f"or change the dimensionality of the argument.")
                    if len(arg['shape']) == 1:
                        if type(arg['value']) is float:
                            arg['value'] = np.zeros(arg['shape']) + arg['value']
                        arg_vals[op_name][arg_name] = list(arg['value'])
                        arg_shapes[op_name][arg_name] = tuple(arg['shape'])
                    else:
                        arg_vals[op_name][arg_name] = [arg['value']]
                        arg_shapes[op_name][arg_name] = (1,)

                # save index of original node's attributes in new nodes attributes
                self._net_config_map[node_ref][op_name][arg_name] = (node_name, op_name, arg_name, 0)

        # go through rest of the nodes and extract their argument shapes and values
        ###########################################################################

        for i, node in enumerate(nodes):

            if node not in self._net_config_map.keys():
                self._net_config_map[node] = {}

            # go through arguments
            for op_name, op in arg_vals.items():

                if op_name not in self._net_config_map[node].keys():
                    self._net_config_map[node][op_name] = {}

                for arg_name in op.keys():

                    arg = self._get_node_attr(node, arg_name, op=op_name, net_config=net_config)

                    # extract value and shape of argument
                    if arg['vtype'] == 'raw':
                        if type(arg['value']) is list or type(arg['value']) is tuple:
                            for j, val in enumerate(arg['value']):
                                arg_vals[op_name][arg_name][j] += val
                    else:
                        if len(arg['shape']) == 0:
                            arg_vals[op_name][arg_name].append(arg['value'])
                        else:
                            if type(arg['value']) is float:
                                arg['value'] = np.zeros(arg['shape']) + arg['value']
                            arg_vals[op_name][arg_name] += list(arg['value'])
                            if arg['shape'][0] > arg_shapes[op_name][arg_name][0]:
                                arg_shapes[op_name][arg_name] = tuple(arg['shape'])

                    # save index of original node's attributes in new nodes attributes
                    self._net_config_map[node][op_name][arg_name] = (node_name, op_name, arg_name, i + 1)

        # go through new arguments and update shape and values
        for op_name in self._get_node_attr(node_name, 'op_graph', net_config=net_config).nodes.keys():
            for arg_name, arg in self._get_op_attr(node_name, op_name, 'variables', net_config=net_config).items():
                if 'value' in arg.keys():
                    arg.update({'value': arg_vals[op_name][arg_name]})
                if 'shape' in arg.keys():
                    arg.update({'shape': (n_nodes,) + arg_shapes[op_name][arg_name]})

        return node_name

    def _vectorize_edges(self,
                         source_name: str,
                         target_name: str,
                         net_config: MultiDiGraph
                         ) -> MultiDiGraph:
        """Combines edges in list and adds a new edge to the new net config.

        Parameters
        ----------
        source_name
        target_name
        net_config
            Networkx MultiDiGraph containing the old, non-vectorized backend configuration.

        Returns
        -------
        MultiDiGraph
            Updated backend graph (with new, vectorized edges added).

        """

        # get edges between source and target
        edges = self._get_edges_between(source_name, target_name)

        # extract edges that connect the same variables on source and target
        ####################################################################

        while edges:

            source_tmp, target_tmp, edge_tmp = edges.pop(0)

            # get source and target variable
            source_var = self._get_edge_attr(source_tmp, target_tmp, edge_tmp, 'source_var')
            target_var = self._get_edge_attr(source_tmp, target_tmp, edge_tmp, 'target_var')

            # get edges with equal source and target variables between source and target node
            edges_tmp = []
            for n, source_tmp, target_tmp, edge_tmp in enumerate(edges):
                if self._get_edge_attr(source_tmp, target_tmp, edge_tmp,
                                       'source_var') == source_var and \
                        self._get_edge_attr(source_tmp, target_tmp, edge_tmp,
                                            'target_var') == target_var:
                    edge_tmp.append(edges[n])

            # vectorize those edges
            #######################

            n_edges = len(edges_tmp)

            if n_edges > 0:

                # go through edges and extract weight and delay
                weight_col = []
                delay_col = []
                old_svar_idx = []
                old_tvar_idx = []

                for source, target, idx in edges_tmp:

                    weight, delay = self._get_edge_conn(source, target, idx)

                    # add weight, delay and variable indices to collector lists
                    if delay:
                        delay_col.append(delay)
                    if weight:
                        weight_col.append(weight)
                    idx = self._get_edge_var_idx(source, target, idx, 'source')
                    if idx:
                        old_svar_idx += idx
                    idx = self._get_edge_var_idx(source, target, idx, 'target')
                    if idx:
                        old_tvar_idx += idx

                # create new, vectorized edge
                #############################

                # extract edge
                edge_ref = edges_tmp[0]
                new_edge = self.net_config.edges[edge_ref]

                # change delay and weight attributes
                new_edge['delay'] = np.concatenate(delay_col, axis=0) if delay_col else None
                new_edge['weight'] = np.concatenate(weight_col, axis=0) if weight_col else None
                new_edge['source_idx'] = old_svar_idx
                new_edge['target_idx'] = old_tvar_idx

                # add new edge to new net config
                net_config.add_edge(source_name, target_name, **new_edge)

            # delete vectorized edges from list
            for edge in edge_tmp:
                edges.pop(edges.index(edge))

        return net_config

    def _vectorize_ops(self,
                       op_key: str,
                       nodes: list,
                       new_node: dict,
                       net_config: MultiDiGraph
                       ) -> None:
        """Vectorize all instances of an operation across nodes and put them into a single-node backend.

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

        if node_name_tmp not in self._net_config_map.keys():
            self._net_config_map[node_name_tmp] = {}

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

                if arg['vtype'] == 'raw':

                    if type(arg['value']) is list:

                        self._net_config_map[node_name_tmp][key] = [self.key, key, (0, len(arg['value']))]

                        for node_name in nodes:

                            if node_name not in self._net_config_map.keys():

                                self._net_config_map[node_name] = {}

                                new_val = net_config.nodes[node_name]['operator_args'][key]['value']
                                old_idx = len(arg['value'])
                                arg['value'].append(new_val)

                                self._net_config_map[node_name_tmp][key] = [self.key,
                                                                            key,
                                                                            (old_idx, arg['value'].shape[0])]

                else:

                    if not hasattr(arg['value'], 'shape'):
                        arg['value'] = np.array(arg['value'])
                    if arg['value'].shape != arg['shape']:
                        try:
                            arg['value'] = np.reshape(arg['value'], arg['shape'])
                        except ValueError:
                            arg['value'] = np.zeros(arg['shape']) + arg['value']

                    # go through nodes and extract shape and value information for arg
                    if nodes:

                        self._net_config_map[node_name_tmp][key] = [self.key, key, (0, arg['value'].shape[0])]

                        for node_name in nodes:

                            if node_name not in self._net_config_map.keys():
                                self._net_config_map[node_name] = {}

                            # extract node arg value and shape
                            arg_tmp = net_config.nodes[node_name]['operator_args'][key]
                            val = arg_tmp['value']
                            dims = tuple(arg_tmp['shape'])

                            # append value to argument dictionary
                            if len(dims) > 0 and type(val) is float:
                                val = np.zeros(dims) + val
                            else:
                                val = np.reshape(np.array(val), dims)
                            if len(dims) == 0:
                                idx_old = 0
                            else:
                                idx_old = arg['value'].shape[0]
                            arg['value'] = np.append(arg['value'], val, axis=0)
                            idx_new = arg['value'].shape[0]
                            idx = (idx_old, idx_new)

                            # add information to _net_config_map
                            self._net_config_map[node_name][key] = [self.key, key, idx]

                    else:

                        # add information to _net_config_map
                        self._net_config_map[node_name_tmp][key] = [self.key, key, (0, arg['value'].shape[0])]

                    # change shape of argument
                    arg['shape'] = arg['value'].shape

                # add argument to new node's argument dictionary
                op_args[key] = arg

        # add operator information to new node
        ######################################

        # add operator
        new_node['operators'][op_key] = op
        new_node['operator_order'].append(op_key)

        # add operator args
        for key, arg in op_args.items():
            new_node['operator_args'][key] = arg
