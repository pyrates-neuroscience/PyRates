"""This module provides the network class that should be used to set-up any model. It creates a tensorflow graph that
manages all computations/operations and a networkx graph that represents the network structure (nodes + edges).
"""

# external imports
import tensorflow as tf
from tensorflow.contrib import graph_editor
from typing import Optional, Tuple
from pandas import DataFrame
import time as t
import numpy as np
from networkx import MultiDiGraph

# pyrates imports
from pyrates.node import Node
from pyrates.edge import Edge

# for the vectorization (tmp-library)
import re

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


class Network(MultiDiGraph):
    """Network level class used to set up and simulate networks of nodes defined by a set of operators.

    Parameters
    ----------

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
                 node_dict: dict,
                 connection_dict: dict,
                 dt: float = 1e-3,
                 tf_graph: Optional[tf.Graph] = None,
                 key: Optional[str] = None
                 ) -> None:
        """Instantiation of network.
        """

        # call of super init
        ####################

        super().__init__()

        # additional object attributes
        ##############################

        self.key = key if key else 'net0'
        self.dt = dt
        self.tf_graph = tf_graph if tf_graph else tf.get_default_graph()
        self.states = []

        vectorize = True

        if vectorize:

            node_dict, connection_dict = self.Vectorization(connection_dict, node_dict)


        # initialize nodes
        ##################

        with self.tf_graph.as_default():

            with tf.variable_scope(self.key):

                # initialize every node in node_dict
                ####################################

                node_updates = []

                for node_name, node_info in node_dict.items():

                    node_ops = dict()
                    node_args = dict()
                    node_args['dt'] = {'variable_type': 'constant',
                                       'name': 'dt',
                                       'shape': (),
                                       'data_type': 'float32',
                                       'initial_value': self.dt}

                    # split dictionary keys into operators and variables
                    for key, val in node_info.items():
                        if 'operator' in key:
                            node_ops[key] = val
                        else:
                            node_args[key] = val

                    # add node to network
                    node = self.add_node(n=node_name,
                                         node_ops=node_ops,
                                         node_args=node_args)

                    # collect update operation of node
                    node_updates.append(node.update)

                # group the update operations of all nodes
                self.update = tf.tuple(node_updates, name='update')

        # initialize edges
        ##################

        if connection_dict:

            # collect connectivity information from connection_dict
            coupling_ops = connection_dict['coupling_operators']
            coupling_op_args = connection_dict['coupling_operator_args']
            sources = connection_dict['sources']
            targets = connection_dict['targets']

            # pass integration step-size to coupling op arguments
            coupling_op_args['dt'] = {'variable_type': 'constant',
                                      'name': 'dt',
                                      'shape': (),
                                      'data_type': 'float32',
                                      'initial_value': self.dt}

            # check dimensionality of coupling_ops
            if len(coupling_ops) < len(sources):
                coupling_ops = [coupling_ops[0] for _ in range(len(sources))]

        else:

            sources, targets, coupling_ops, coupling_op_args = ([], [], [], [])

        with self.tf_graph.as_default():

            with tf.variable_scope(self.key):

                with tf.control_dependencies(self.update):

                    projections = []
                    for i, (source, target, op) in enumerate(zip(sources,
                                                                 targets,
                                                                 coupling_ops)):

                        # create edge argument dictionary
                        edge_dict = {}
                        for key, val in coupling_op_args.items():
                            edge_dict[key] = val[i] if type(val) is list else val

                        # add edge to network
                        edge = self.add_edge(u=source,
                                             v=target,
                                             coupling_op=op,
                                             coupling_op_args=edge_dict,
                                             key=f"edge_{i}")

                        # collect project operation of edge
                        projections.append(edge.project)

                    # group project operations of all edges
                    if len(projections) > 0:
                        self.step = tf.tuple(projections, name='step')
                    else:
                        self.step = self.update

    def add_node(self, n, node_ops, node_args, **attr):

        # instantiate node
        node = Node(operations=node_ops,
                    operation_args=node_args,
                    key=n,
                    tf_graph=self.tf_graph)

        # call super method
        super().add_node(n, handle=node, **attr)

        return node

    def add_edge(self, u, v, coupling_op, coupling_op_args, key=None, **attr):

        # define edge key
        if key is None:
            key = f'{u}_{v}'

        # create edge
        edge = Edge(source=self.nodes[u]['handle'],
                    target=self.nodes[v]['handle'],
                    coupling_op=coupling_op,
                    coupling_op_args=coupling_op_args,
                    tf_graph=self.tf_graph,
                    key=key)

        # call super method
        super().add_edge(u, v, key, handle=edge, **attr)

        return edge

    def update_node(self, n, node_ops, node_args, **attr):

        # create updated node on separate graph
        #######################################

        gr_tmp = tf.Graph()
        #node_new = Node(operations=node_ops,
        #                operation_args=node_args,
        #                key=n,
        #                tf_graph=gr_tmp)
        node_new = self.nodes[n]['handle']

        # collect tensors to replace in tensorflow graph
        ################################################

        node_old = self.nodes[n]['handle']
        replace_vars = {}
        for key, val in vars(node_old).items():
            if isinstance(val, tf.Tensor) or isinstance(val, tf.Variable):
                replace_vars[val.name] = getattr(node_new, key)

        # replace old node with new in graph
        ####################################

        graph_def = self.tf_graph.as_graph_def()

        with gr_tmp.as_default():
            tf.import_graph_def(graph_def=graph_def, input_map=replace_vars)

        with self.tf_graph.as_default():
            tf.reset_default_graph()

        self.tf_graph = gr_tmp

        # update fields on networkx node
        ################################

        self.nodes[n]['handle'] = node_new
        for key, val in attr.items():
            self.nodes[n][key] = val

    def update_edge(self, u, v, coupling_op, coupling_op_args, key=None, **attr):

        # create updated edge on separate graph
        #######################################

        gr_tmp = tf.Graph()
        edge_new = Edge(source=self.nodes[u]['handle'],
                        target=self.nodes[v]['handle'],
                        coupling_op=coupling_op,
                        coupling_op_args=coupling_op_args,
                        tf_graph=gr_tmp,
                        key=key)

        # replace old edge with new in graph
        ####################################

        graph_def = self.tf_graph.as_graph_def()

        with gr_tmp.as_default():
            tf.import_graph_def(graph_def=graph_def, input_map={key: edge_new})

        with self.tf_graph.as_default():
            tf.reset_default_graph()

        self.tf_graph = gr_tmp

        # update fields on networkx node
        ################################

        self.edges[key]['handle'] = edge_new
        for key_tmp, val in attr.items():
            self.edges[key][key_tmp] = val

    def remove_node(self, n):
        super().remove_node(n)
        # implement node removal for tf graph. Needs:
        #   a) storage of graph as graphdef,
        #   b) detachment of node subgraph from graph
        #   c) remove_training_nodes + extract_sub_graph

    def Vectorization(self, connection_dict, node_dict):

        n_jrcs = len(node_dict.keys())/3
        jrcsID = 0
        len_prev_opr = []
        operationID = 0
        opr_dict = dict()
        node_args = dict()

        for node_name, node_info in node_dict.items():

            # Reseting jrcsID and operationID
            if jrcsID == n_jrcs:
                jrcsID = 0
                operationID = 0

            jrcsID += 1

            # split dictionary keys into operators and variables
            for key, val in node_info.items():

                # 1- Checking for the "_Operation" keyword
                # 2- Calculating the length of the Operation list, if its more than 1.
                # 3- Either: add the name of the operation with the operation in the dict
                # 3- Or: append the dict if the name is already there.
                # 4- IF: there is two similar operations in same node rearrange them,
                # 4- as how the similar nodes in different circuits is arranged.
                ################################################################

                if 'operator' in key:
                    operationID += 1
                    if operationID > len(len_prev_opr):
                        len_prev_opr.append(0)
                    else:
                        pass

                    # Append similar operation together is a list
                    if not key in opr_dict:
                        if len(val) > 1:
                            opr_dict[key] = val
                        else:
                            opr_dict[key] = [val]
                        len_prev_opr[operationID - 1] += len(val)
                    else:
                        if len(val) > 1:
                            for i, elems in enumerate(val):
                                prev_operations = 0
                                for prev_i in range(0, operationID):
                                    prev_operations += len_prev_opr[prev_i]
                                insertion_idx = (i * (jrcsID)) + (jrcsID - 1)
                                opr_dict[key][insertion_idx:insertion_idx] = [elems]
                                # opr_dict[key].append(elems)
                        else:
                            opr_dict[key].append(val)

                # if its not an Operation it would be then save in the node_args dict
                # with adding the node name to its key name in the dict.
                ####################################################################

                else:
                    node_args[key + node_name] = val
                    val['node_name'] = node_name

            operationID = 0

            # At the end of each node after saving the operations in each node and the variable(args) names,
            # we replace the old names in the operation with the new ones.
            # That's why we need the input args be written with spaces in each operation (limitation 2).
            ############################################################################################

            for key, val in node_info.items():

                if 'operator' in key:
                    pass
                else:
                    for k in opr_dict:
                        for j in range(0, len(opr_dict[k])):
                            opr_dict[k][j] = [w.replace(' ' + key + ' ', ' ' + key + node_name + ' ') for w in
                                              opr_dict[k][j]]

        batched = dict()

        # Batching parameters from inputr dictionary
        for i_dict, oper_list in enumerate(sorted(opr_dict.values())):

            stripped_par_list = []
            for each_oper in oper_list:
                par_list = re.findall(r"[\w]+", str(each_oper).strip('[]'))
                stripped_par_list.append(par_list)
            list_len = len(par_list)

            for first_itr in range(0, list_len):

                variable_found = False
                for second_itr, list_enum in enumerate(stripped_par_list):
                    poped_symbol = list_enum.pop()

                    if poped_symbol in node_args:

                        # To check if its not already in batched.
                        for k, v in batched.items():
                            if f'{poped_symbol}_' in k:
                                variable_found = True
                            else:
                                pass

                        if not variable_found:
                            if second_itr == 0:
                                if not poped_symbol in batched:
                                    batched[poped_symbol] = {'name': poped_symbol,
                                                             'par_name': [f'{poped_symbol}'],
                                                             'variable_type': 'state_variable',
                                                             'data_type': node_args[poped_symbol]['data_type'],
                                                             'nodes': [node_args[poped_symbol]['node_name']],
                                                             'initial_value': [node_args[poped_symbol]['initial_value']],
                                                             'shape': [1, 1],
                                                             }
                                    batched[f'{poped_symbol}_'] = batched.pop(poped_symbol)
                                    pop_par_init = f'{poped_symbol}_'
                                else:
                                    print("Is this an Error? How was it in this case !?!?!?")
                            else:
                                batched[pop_par_init]['initial_value'].append(node_args[poped_symbol]['initial_value'])
                                batched[pop_par_init]['par_name'].append(f'{poped_symbol}')
                                batched[pop_par_init]['shape'] = [len(batched[pop_par_init]['initial_value']), 1]
                                batched[pop_par_init]['nodes'].append(node_args[poped_symbol]['node_name'])
                                batched[pop_par_init]['name'] = f'{pop_par_init}{poped_symbol}_'
                                batched[f'{pop_par_init}{poped_symbol}_'] = batched.pop(pop_par_init)
                                pop_par_init = f'{pop_par_init}{poped_symbol}_'
                        else:
                            pass

        batch_opr_dict = dict()

        # batch_opr_dict:
        #               A dictionary which have only one Operation of each Operation with the
        #               new batched collected names and a slicing for which indices the operations is done on.

        index_start = None
        start_bool = True

        # Create a dictionary with related collected variable names in the batched dictionary to be parsed.
        # A strip of the operations and checking if each variable(argument) is in any of the batched dict keys and
        # then checking in the par_name.
        # popping each variable in opr_dict and replace it with the new variable name form batched dictionary keys
        # and add the slicing required for each operation.
        #############################################################################################################

        for dict1_k in opr_dict:
            stripped_par_list = []
            for expr in opr_dict[dict1_k]:
                par_list = re.findall(r"[\w]+", str(expr).strip('[]'))
                stripped_par_list.append(par_list)

            list_len = len(par_list)

            for first_itr in range(0, list_len):
                for i, expr in enumerate(stripped_par_list):
                    popped_par = expr.pop()
                    for batched_k in batched:
                        if popped_par in batched[batched_k]['par_name']:
                            if popped_par != 'dt':
                                if not dict1_k in batch_opr_dict:
                                    batch_opr_dict[dict1_k] = opr_dict[dict1_k][0]
                                else:
                                    pass
                                if start_bool:
                                    index_start = batched[batched_k]['par_name'].index(popped_par)
                                    start_bool = False
                                    replace_val = popped_par
                                else:
                                    pass

                                ind = batched[batched_k]['par_name'].index(popped_par)
                                index_end = ind + 1
                                if popped_par != 'd':
                                    batch_opr_dict[dict1_k] = [w.replace(' ' + replace_val + ' ',
                                                                         ' ' + batched_k + f"[{index_start}:{index_end}]" + ' ')
                                                               for w in batch_opr_dict[dict1_k]]
                                    replace_val = batched_k + f"[{index_start}:{index_end}]"
                start_bool = True

        batch_opr_list = []

        # I am doing it in this way as it the operations lost their order in the process. (limitation 5)
        # This problem can be solved by adding a dependency in the input dictionary or in the first loop.
        #################################################################################################
        batched.update({'operator_rtp_syn1': batch_opr_dict['operator_rtp_syn1']})
        batched.update({'operator_rtp_syn2': batch_opr_dict['operator_rtp_syn2']})
        batched.update({'operator_rtp_soma_pc': batch_opr_dict['operator_rtp_soma_pc']})
        batched.update({'operator_rtp_soma': batch_opr_dict['operator_rtp_soma']})
        batched.update({'operator_ptr': batch_opr_dict['operator_ptr']})

        conn = []

        for i in range(0, len(connection_dict['coupling_operator_args']['c'])):
            conn.append(
                connection_dict['coupling_operator_args']['c'][i][
                    'initial_value'])  # To add it in the batched dict

        # C_batched:
        #           Dictionary of batched [[input = output * c]] for each index of connections.

        C_batched = dict()
        C_op = []
        k_output_found = False  # A Flag
        k_input_found = False  # A Flag

        for i_batched, k_batched in enumerate(batched):
            if not k_output_found:
                for k_output in connection_dict['coupling_operator_args']['output']:
                    if k_output['name'] in k_batched:
                        batched_source_key = k_batched
                        k_output_found = True
                        break

            if not k_input_found:
                for k_input in connection_dict['coupling_operator_args']['input']:
                    if k_input['name'] in k_batched:
                        batched_target_key = k_batched
                        k_input_found = True
                        break
        new_eq = [w.replace('output', 'mapping @ ((mapping2@output)') for [w] in connection_dict['coupling_operators']]
        for exp in new_eq:
            new_eq = exp.replace(exp, exp+')')

        n_edges = len(connection_dict['coupling_operator_args']['c'])
        n_inp = len(batched[batched_target_key]['initial_value'])
        n_out = len(batched[batched_source_key]['initial_value'])

        # create mapping
        mapping = np.zeros((n_inp, n_edges), dtype=np.float32)
        mapping2 = np.zeros((n_edges, n_out), dtype=np.float32)
        scatter_indices = []
        for conn_i in range(0, len(connection_dict['coupling_operator_args']['c'])):

            Source_idx = batched[batched_source_key]['par_name'].index(
                connection_dict['coupling_operator_args']['output'][conn_i]['name'])

            Target_idx = batched[batched_target_key]['par_name'].index(
                connection_dict['coupling_operator_args']['input'][conn_i]['name'])

            scatter_indices.append([Source_idx])

            # C_op_target.append(f'batched_target_key[{Target_idx}] = new_batched_source[{conn_i}]')

            mapping[Target_idx, conn_i] = 1.
            mapping2[conn_i, Source_idx] = 1.

        #C_op.append(f'output_new = scatter(scatter_idxs, output, shape_scatter_out)')

        #C_op.append(new_eq)

        # targets = unique(edge_targets)
        # for i, t in enumerate(targets):
        #     for j, et in enumerate(edge_targets):
        #         if t == et:

        C_batched['coupling_operators'] = [[new_eq]]
        C_batched['coupling_operator_args'] = {'c': {'name': 'c',
                                                     'variable_type': 'state_variable',
                                                     'data_type': 'float32',
                                                     'shape': [n_edges, 1],
                                                     'initial_value': conn},
                                               'input': {'variable_type': 'target_var', 'name': batched_target_key},
                                               'output': {'variable_type': 'source_var', 'name': batched_source_key},
                                               'mapping': {'variable_type': 'constant',
                                                           'name': 'mapping',
                                                           'shape': [n_inp, n_edges],
                                                           'data_type': 'float32',
                                                           'initial_value': mapping},
                                               'mapping2': {'variable_type': 'constant',
                                                            'name': 'mapping2',
                                                            'shape': [n_edges, n_out],
                                                            'data_type': 'float32',
                                                            'initial_value': mapping2},
                                               'shape_scatter_out':{'variable_type': 'raw',
                                                                    'variable': [n_edges, 1]},
                                               'scatter_idxs': {'name': 'scatter_idxs',
                                                                'shape':[n_edges, 1],
                                                                'data_type': 'int32',
                                                                'variable_type': 'constant',
                                                                'initial_value': scatter_indices}
                                               }

        C_batched['sources'] = ['BNode' for _ in range(len(C_batched['coupling_operators']))]
        C_batched['targets'] = ['BNode' for _ in range(len(C_batched['coupling_operators']))]

        batched['operator_rtp_syn1'].append(batched.pop('operator_rtp_syn2')[0])
        batched['operator_rtp_soma'].append(batched.pop('operator_rtp_soma_pc')[0])

        return({'BNode':batched}, C_batched)

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

        with tf.Session(graph=self.tf_graph) as sess:

            #writer = tf.summary.FileWriter('/tmp/log/', graph=self.tf_graph)

            # initialize all variables
            sess.run(tf.global_variables_initializer())

            results = []
            t_start = t.time()

            # simulate network behavior for each time-step
            for step in range(sim_steps):

                sess.run(self.step, inp[step])

                if step % sampling_steps == 0:
                    results.append([np.squeeze(var.eval()) for var in outputs.values()])

            t_end = t.time()

            # display simulation time
            if simulation_time:
                print(f"{simulation_time}s of network behavior were simulated in {t_end - t_start} s given a "
                      f"simulation resolution of {self.dt} s.")
            else:
                print(f"Network computations finished after {t_end - t_start} seconds.")
            #writer.close()

            # store results in pandas dataframe
            time = 0.
            times = []
            for i in range(sim_steps):
                time += sampling_step_size
                times.append(time)

            results = np.array(results)
            if len(results.shape) > 2:
                n_steps, n_vars, n_nodes = results.shape
            else:
                n_steps, n_vars = results.shape
                n_nodes = 1
            results = np.reshape(results, (n_steps, n_nodes*n_vars))
            columns = []
            for var in outputs.keys():
                columns += [f"{var}_{i}" for i in range(n_nodes)]
            results_final = DataFrame(data=results, columns=columns, index=times)

        return results_final, (t_end - t_start)
