"""This module provides the backend class that should be used to set-up any backend. It creates a tensorflow graph that
manages all computations/operations and a networkx graph that represents the backend structure (nodes + edges).
"""

# external imports
import tensorflow as tf
from typing import Optional, Tuple, List
from pandas import DataFrame
import numpy as np
from networkx import MultiDiGraph, find_cycle, NetworkXNoCycle, DiGraph, is_isomorphic
from copy import copy, deepcopy

# pyrates imports
from pyrates.backend.parser import parse_equation_list, parse_dict
from pyrates.backend.backend_wrapper import TensorflowBackend
from pyrates import PyRatesException

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


class ComputeGraph(MultiDiGraph):
    """Creates an RNN cell that contains all nodes in the network plus their recurrent connections.

    Parameters
    ----------
    net_config
    dt
    vectorize
    name
    build_in_place

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

        super().__init__(name=name if name else 'net.0')

        # instantiate the backend
        self.backend = TensorflowBackend()

        # pre-process the network configuration
        self.dt = dt
        self._net_config_map = {}
        self.net_config = self._preprocess_net_config(net_config) if build_in_place \
            else self._preprocess_net_config(deepcopy(net_config))
        self.net_config = self._vectorize(net_config=deepcopy(self.net_config), vectorization_mode=vectorize)

        # set time constant of the network
        self._dt = parse_dict({'dt': {'vtype': 'constant', 'dtype': 'float32', 'shape': (), 'value': self.dt}},
                              backend=self.backend)['dt']

        # parse node operations
        #######################

        self.node_updates = []

        for node_name, node in self.net_config.nodes.items():

            op_graph = self._get_node_attr(node_name, 'op_graph')

            # check operators for cyclic relationships
            try:
                find_cycle(op_graph)
            except NetworkXNoCycle:
                pass
            else:
                raise PyRatesException("Found cyclic operator graph. "
                                       "Cycles are not allowed for operators within one node.")

            graph = op_graph.copy()  # type: DiGraph

            # first, parse operators that have no dependencies on other operators
            # noinspection PyTypeChecker
            primary_ops = [op for op, in_degree in graph.in_degree if in_degree == 0]
            op_updates = self._add_ops(primary_ops, True, node_name)

            # remove parsed operators from graph
            graph.remove_nodes_from(primary_ops)

            # now, pass all other operators on the node
            while graph.nodes:

                # get all operators that have no dependencies on other operators
                # noinspection PyTypeChecker
                secondary_ops = [op for op, in_degree in graph.in_degree if in_degree == 0]
                op_updates = self._add_ops(secondary_ops, False, node_name, updates=op_updates)

                # remove parsed operators from graph
                graph.remove_nodes_from(secondary_ops)

        # move unconnected node-operator updates to node updates
        for updates in op_updates.values():
            self.node_updates += updates

        # collect output variables
        edges = []
        source_vars = []
        source_vars_idx = []
        for source_node, target_node, edge_idx in self.net_config.edges:
            svar = self._get_edge_attr(source_node, target_node, edge_idx, 'source_var')
            if svar in self.node_updates:
                self.node_updates.pop(self.node_updates.index(svar))
            if svar not in source_vars:
                source_vars.append(svar)
            source_vars_idx.append(source_vars.index(svar))
            edges.append((source_node, target_node, edge_idx))

        # get control dependencies on those output variables
        if source_vars:
            source_vars = self.backend.add_op('tuple', source_vars, name='node_updates_I', scope=self.name)

        # set control dependencies for the remaining node updates
        if self.node_updates:
            node_updates_deps = self.backend.add_op('tuple', self.node_updates, name='node_updates_II', scope=self.name)
            for node_name in self.net_config.nodes:
                op_graph = self._get_node_attr(node_name, 'op_graph')
                for op_name, op in op_graph.nodes.items():
                    for key, var in op['variables'].items():
                        if var in self.node_updates:
                            op['variables'][key] = node_updates_deps[self.node_updates.index(var)]
        else:
            node_updates_deps = []

        # parse edges
        #############

        self.edge_updates = []
        for (source_node, target_node, edge_idx), svar_idx in zip(edges, source_vars_idx):

            # extract edge information
            weight = self._get_edge_attr(source_node, target_node, edge_idx, 'weight')
            delay = self._get_edge_attr(source_node, target_node, edge_idx, 'delay')
            sidx = self._get_edge_attr(source_node, target_node, edge_idx, 'source_idx')
            tidx = self._get_edge_attr(source_node, target_node, edge_idx, 'target_idx')
            tvar = self._get_edge_attr(source_node, target_node, edge_idx, 'target_var', retrieve_from_node=False)

            # get source and target variable
            svar = source_vars[svar_idx]
            op, var = tvar.split('/')
            try:
                tvar = self._get_op_attr(target_node, op, f'{var}_orig')
            except KeyError:
                tvar = self._get_op_attr(target_node, op, var)


            # define target index
            if delay is not None and tidx:
                tidx = [idx + d for idx, d in zip(tidx, delay)]
            elif not tidx:
                tidx = list(delay)

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
            args['vars']['target_var'] = tvar
            args['inputs']['source_var'] = svar

            # parse mapping
            args = parse_equation_list([eq], args, backend=self.backend,
                                       scope=f"{self.name}/{source_node}/{target_node}/{edge_idx}")
            args.pop('lhs_evals')

            # store information in network config
            edge = self.net_config.edges[source_node, target_node, edge_idx]

            # update edge attributes
            edge.update(args['inputs'])
            edge.update(args['vars'])
            edge.update(args['updates'])

            # add projection to edge updates
            self.edge_updates.append(edge['target_var'])

        # add differential equation updates to edges
        for node_name in self.net_config.nodes:
            op_graph = self._get_node_attr(node_name, 'op_graph')
            for op_name, op in op_graph.nodes.items():
                for key_old, var in op['variables'].items():

                    # check if variable is a state variable updated via a differential equation
                    if '_old' in key_old:

                        # extract variable name and values
                        key_new = key_old.replace('_old', '')
                        var_new = op['variables'][key_new]
                        var_old = op['variables'][key_old]

                        # define mapping equation and its arguments
                        eq = f'{key_old} = {key_new}'
                        args = {'inputs': {key_new: var_new}, 'vars': {key_old: var_old}}

                        # parse mapping
                        args = parse_equation_list([eq], args, backend=self.backend,
                                                   scope=f"{self.name}/{node_name}/{op_name}")

                        args.pop('lhs_evals')

                        # add update operation to edge updates and remove it from node updates if necessary
                        self.edge_updates.append(args['updates'][key_old])
                        if var_new in node_updates_deps:
                            node_updates_deps.pop(node_updates_deps.index(var_new))

        update_ops = self.edge_updates + node_updates_deps
        self.step = self.backend.add_op('group', update_ops, scope=self.name, name='network_update')

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
        out_idx = self.backend.add_variable(name='out_var_idx', dtype=tf.int32, shape=(), value=-1,
                                            scope="output_collection")

        # create increment operator for counting index
        out_idx_add = self.backend.add_op('+=', out_idx, 1, scope="output_collection")

        # add collector variables to the graph
        for key, var in outputs_tmp.items():
            shape = [int(sim_steps / sampling_steps)] + list(var.shape)
            output_col[key] = self.backend.add_variable(name=key, dtype=tf.float32, shape=shape, value=np.zeros(shape),
                                                        scope="output_collection")

            # add collect operation to the graph
            store_ops.append(self.backend.add_op('scatter_update', output_col[key], out_idx, var,
                                                 scope="output_collection",
                                                 dependencies=[out_idx_add]))

        sampling_op = self.backend.add_op('group', store_ops, name='output_collection')

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

        output_col, sim_time = self.backend.run(steps=sim_steps, ops=self.step, inputs=inp,
                                                outputs=output_col, sampling_steps=sampling_steps,
                                                sampling_ops=sampling_op, out_dir=out_dir)

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
            out_vars['time'] = np.arange(0., simulation_time, sampling_step_size)

        # display simulation time
        if verbose:
            if simulation_time:
                print(f"{simulation_time}s of backend behavior were simulated in {sim_time} s given a "
                      f"simulation resolution of {self.dt} s.")
            else:
                print(f"ComputeGraph computations finished after {sim_time} seconds.")

        return out_vars, sim_time

    def _add_ops(self, ops, primary_ops, node_name, updates=None):
        """

        Parameters
        ----------
        ops
        primary_ops
        node_name

        Returns
        -------

        """
        if updates is None:
            updates = {}

        for op_name in ops:

            updates[op_name] = []

            # retrieve operator and operator args
            op_args = dict()
            op_args['vars'] = self._get_op_attr(node_name, op_name, 'variables')
            op_args['vars']['dt'] = self._dt
            op_args['inputs'] = {}
            op_info = self._get_op_attr(node_name, op_name, 'operator')

            # handle operator inputs
            dependencies = []
            for var_name, inp in op_info['inputs'].items():

                in_ops_deps = []

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
                            if var in updates[in_op[0]]:
                                updates[in_op[0]].pop(updates[in_op[0]].index(var))
                            in_ops_deps += updates.pop(in_op[0])
                            i += 1

                        elif type(in_op) is tuple:

                            in_ops = in_op[0]
                            reduce_dim = in_op[1]

                            in_ops_tmp = []
                            in_ops_deps_tmp = []
                            for op in in_ops:
                                if primary_ops:
                                    raise ValueError(f'Input found in primary operator {op_name} on node {node_name}. '
                                                     f'This operator should have no node-internal inputs. '
                                                     f'Please move the operator inputs to the node level or change the '
                                                     f'input-output relationships between the node operators.')
                                else:
                                    var = self._get_op_attr(node_name, op, 'output')
                                    if type(var) is str:
                                        raise ValueError(
                                            f'Wrong operator order on node {node_name}. Operator {op_name} '
                                            f'needs input from operator {op} which has not been '
                                            f'processed yet. Please consider changing the operator order '
                                            f'or dependencies.')
                                in_ops_tmp.append(var)
                                if var in updates[op]:
                                    updates[op].pop(updates[op].index(var))
                                in_ops_deps_tmp += updates.pop(op)
                                i += 1

                            in_ops_col.append(self._map_multiple_inputs(in_ops_tmp, reduce_dim,
                                                                        scope=f"{self.name}/{node_name}/{op_name}",
                                                                        dependencies=in_ops_deps_tmp))

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
                            if var in updates[in_op]:
                                updates[in_op].pop(updates[in_op].index(var))
                            in_ops_deps += updates.pop(in_op)
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
                        in_ops = self._map_multiple_inputs(in_ops, inp['reduce_dim'],
                                                           scope=f"{self.name}/{node_name}/{op_name}",
                                                           dependencies=in_ops_deps)
                        in_ops_deps.clear()

                    # for a single input variable
                    else:
                        in_ops = in_ops_col[0]

                    # add input variable to dictionary
                    op_args['inputs'][var_name] = in_ops
                    if var_name in op_args['vars'].keys():
                        op_args['vars'].pop(var_name)

                dependencies += in_ops_deps

            # parse equations into tensorflow
            op_args = parse_equation_list(op_info['equations'], op_args, backend=self.backend,
                                          scope=f"{self.name}/{node_name}/{op_name}",
                                          dependencies=dependencies)

            # store operator variables in net config
            op_vars = self._get_op_attr(node_name, op_name, 'variables')
            op_vars.update(op_args['inputs'])
            op_vars.update(op_args['vars'])
            for key, var in op_args['updates'].items():
                if key in op_vars:
                    op_vars[f'{key}_orig'] = op_vars.pop(key)
                    op_vars[key] = var
                else:
                    op_vars[key] = var

            # store the update operations of op
            if self._get_node_attr(node_name, 'op_graph').out_degree(op_name) == 0:

                # if the operator does not project to others, store them in node updates
                for var in list(set(op_args['lhs_evals'])):
                    self.node_updates.append(op_args['updates'][var])

            else:

                # store them in node update collector
                for var in list(set(op_args['lhs_evals'])):
                    updates[op_name].append(op_args['updates'][var])

        return updates

    def _map_multiple_inputs(self, inputs, reduce_dim, **kwargs):
        """
        """

        inp = self.backend.add_op('stack', inputs, **kwargs)
        if reduce_dim:
            return self.backend.add_op('sum', inp, axis=0, **kwargs)
        else:
            return self.backend.add_op('reshape', inp, shape=(inp.shape[0] * inp.shape[1],), **kwargs)

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

    def _get_edge_attr(self, source, target, edge, attr, retrieve_from_node=True, net_config=None):
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
            if 'var' in attr and type(attr_val) is str and retrieve_from_node:
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
        if attr == 'output':
            attr = op['operator']['output']
        if attr in op['variables'].keys():
            return op['variables'][attr]
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

        nodes = []
        for node in net_config.nodes:
            test_val = self._get_node_attr(node, attr, net_config=net_config)
            try:
                if is_isomorphic(val, test_val):
                    nodes.append(node)
            except (TypeError, ValueError):
                if val == test_val:
                    nodes.append(node)

        return nodes

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

        if source in net_config.nodes and target in net_config.nodes:
            return [(source, target, edge) for edge in range(net_config.graph.number_of_edges(source, target))]
        else:
            edges = []
            for source_tmp in self._net_config_map.keys():
                for target_tmp in self._net_config_map.keys():
                    if self._contains_node(source, source_tmp) and self._contains_node(target, target_tmp):
                        edges += [(source_tmp, target_tmp, edge) for edge
                                  in range(net_config.graph.number_of_edges(source_tmp, target_tmp))]
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
            try:
                var = 'source_var' if idx_type == 'source' else 'target_var'
                op, var = self.net_config.edges[source, target, edge][var].split('/')
                _, _, _, idx_new = self._net_config_map[node_to_idx][op][var]
                if type(idx_new) is int:
                    idx_new = [idx_new]
                else:
                    idx_new = list(idx_new)
            except KeyError:
                idx_new = None

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
                             f"{var}_buffer_{idx}[-1] = 0."]
        else:
            eqs_op_read = [f"{var} = {var}_buffer_{idx}[:, 0]"]
            eqs_op_rotate = [f"{var}_buffer_{idx}_tmp = {var}_buffer_{idx}[:, 1:]",
                             f"{var}_buffer_{idx}[:, -1] = 0."]

        # create buffer variable definitions
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
                                    'output': f'{var}_buffer_{idx}_tmp',
                                    'equations': eqs_op_rotate},
                          variables=var_dict)
        op_graph.add_node(f'{op}_{var}_buffer_read_{idx}',
                          operator={'inputs': {f'{var}_buffer_{idx}': {'sources': [f'{op}_{var}_buffer_rotate_{idx}'],
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
                        delay = np.asarray((delay / self.dt) + 1, dtype=np.int32)
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
                    net_config.label_map.pop(n)
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

            source_tmp, target_tmp, edge_tmp = edges[0]

            # get source and target variable
            source_var = self._get_edge_attr(source_tmp, target_tmp, edge_tmp, 'source_var')
            target_var = self._get_edge_attr(source_tmp, target_tmp, edge_tmp, 'target_var')

            # get edges with equal source and target variables between source and target node
            edges_tmp = []
            for n, (source_tmp, target_tmp, edge_tmp) in enumerate(edges):
                if self._get_edge_attr(source_tmp, target_tmp, edge_tmp,
                                       'source_var') == source_var and \
                        self._get_edge_attr(source_tmp, target_tmp, edge_tmp,
                                            'target_var') == target_var:
                    edges_tmp.append(edges[n])

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
                    idx_tmp = self._get_edge_var_idx(source, target, idx, 'source')
                    if idx_tmp:
                        old_svar_idx += idx_tmp
                    idx_tmp = self._get_edge_var_idx(source, target, idx, 'target')
                    if idx_tmp:
                        old_tvar_idx += idx_tmp

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
            for edge in edges_tmp:
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
