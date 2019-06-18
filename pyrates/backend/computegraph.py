# -*- coding: utf-8 -*-
#
#
# PyRates software framework for flexible implementation of neural
# network model_templates and simulations. See also:
# https://github.com/pyrates-neuroscience/PyRates
#
# Copyright (C) 2017-2018 the original authors (Richard Gast and
# Daniel Rose), the Max-Planck-Institute for Human Cognitive Brain
# Sciences ("MPI CBS") and contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#
# CITATION:
#
# Richard Gast and Daniel Rose et. al. in preparation

"""This module provides the backend class that should be used to set up any backend in pyrates.
"""

# external imports
from typing import Optional, Tuple, List, Union, Any
from pandas import DataFrame, MultiIndex
import numpy as np
from networkx import find_cycle, NetworkXNoCycle, DiGraph
from copy import deepcopy

# pyrates imports
from pyrates.backend.parser import parse_equation_list, parse_dict
from pyrates.backend.backend_wrapper import TensorflowBackend, NumpyBackend
from pyrates import PyRatesException
from pyrates.ir.circuit import CircuitIR

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


class ComputeGraph(object):
    """Creates a compute graph that contains all nodes in the network plus their recurrent connections.

    Parameters
    ----------
    net_config
        Intermediate representation of the network configuration. For a more detailed description, see the documentation
        of `pyrates.IR.CircuitIR`.
    dt
        Step-size with which the network should be simulated later on.
        Important for discretizing delays, differential equations, ...
    vectorization
        Defines the mode of automatic parallelization optimization that should be used. Can be `nodes` for lumping all
        nodes together in a vector, `full` for full vectorization of the network, or `None` for no vectorization.
    name
        Name of the network.
    build_in_place
        If False, a copy of the `net_config``will be made before compute graph creation. Should be used, if the
        `net_config` will be re-used for multiple compute graphs.
    use_device
        Can be either `cpu` or `gpu`. Device placement will be soft.

    """

    def __init__(self,
                 net_config: CircuitIR,
                 dt: float = 1e-3,
                 vectorization: str = 'none',
                 name: Optional[str] = 'net0',
                 build_in_place: bool = True,
                 backend: str = 'numpy',
                 float_precision: str = 'float32',
                 **kwargs
                 ) -> None:
        """Instantiates operator.
        """

        # set basic attributes
        ######################

        super().__init__()
        self.name = name
        self._float_precision = float_precision
        net_config = net_config.move_edge_operators_to_nodes(copy_data=False)

        # instantiate the backend and set the backend default_device
        if backend == 'tensorflow':
            backend = TensorflowBackend
        elif backend == 'numpy':
            backend = NumpyBackend
        else:
            raise ValueError(f'Invalid backend type: {backend}. See documentation for supported backends.')
        kwargs['name'] = self.name
        kwargs['float_default_type'] = self._float_precision
        self.backend = backend(**kwargs)

        # pre-process the network configuration
        self.dt = dt
        self._net_config_map = {}
        self.net_config = self._net_config_consistency_check(net_config) if build_in_place \
            else self._net_config_consistency_check(deepcopy(net_config))
        self._vectorize(vectorization_mode=vectorization)

        # set time constant of the network
        self._dt = parse_dict({'dt': {'vtype': 'constant', 'dtype': self._float_precision, 'shape': (),
                                      'value': self.dt}},
                              backend=self.backend)['dt']

        # parse node operations
        #######################

        self.node_updates = []

        for node_name, node in self.net_config.nodes.items():

            self.backend.bottom_layer()
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
            self.node_updates.append(self.backend.layer)
            op_updates = self._add_ops(primary_ops, node_name=node_name, primary_ops=True)
            self.backend.next_layer()

            # remove parsed operators from graph
            graph.remove_nodes_from(primary_ops)

            # now, pass all other operators on the node
            while graph.nodes:

                # get all operators that have no dependencies on other operators
                # noinspection PyTypeChecker
                secondary_ops = [op for op, in_degree in graph.in_degree if in_degree == 0]
                self.node_updates.append(self.backend.layer)
                op_updates = self._add_ops(secondary_ops, node_name=node_name, updates=op_updates, primary_ops=False)
                self.backend.next_layer()

                # remove parsed operators from graph
                graph.remove_nodes_from(secondary_ops)

        # add layer that contains left-hand side equation updates
        ########################################################

        self.backend.top_layer()
        node_names, op_names, var_names, var_updates = [], [], [], []

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

                        # add info to lists
                        node_names.append(node_name)
                        op_names.append(op_name)
                        var_names.append(key_old)
                        var_updates.append(args['vars'][key_old])

        # add layer to node updates
        self.node_updates.append(self.backend.layer)

        # save control dependencies to network config
        for node_name, op_name, var_name, var in zip(node_names, op_names, var_names, var_updates):
            self._set_op_attr(node_name, op_name, var_name, var)

        # parse edges
        #############

        self.backend.top_layer()

        # collect output variables
        source_nodes, target_nodes, edge_indices = [], [], []
        op_names, var_names = [], []
        for source_node, target_node, edge_idx in self.net_config.edges:
            svar = self._get_edge_attr(source_node, target_node, edge_idx, 'source_var', retrieve_from_node=False)
            op, var = svar.split('/')
            svar = self._get_op_attr(source_node, op, var)
            op_names.append(op)
            var_names.append(var)
            source_nodes.append(source_node)
            target_nodes.append(target_node)
            edge_indices.append(edge_idx)

        for source_node, target_node, edge_idx in zip(source_nodes, target_nodes, edge_indices):

            # extract edge information
            weight = self._get_edge_attr(source_node, target_node, edge_idx, 'weight')
            delay = self._get_edge_attr(source_node, target_node, edge_idx, 'delay')
            sidx = self._get_edge_attr(source_node, target_node, edge_idx, 'source_idx')
            tidx = self._get_edge_attr(source_node, target_node, edge_idx, 'target_idx')
            svar = self._get_edge_attr(source_node, target_node, edge_idx, 'source_var')
            tvar = self._get_edge_attr(source_node, target_node, edge_idx, 'target_var', retrieve_from_node=False)
            add_project = self._get_edge_attr(source_node, target_node, edge_idx, 'add_project')

            # get original target variable
            op, var = tvar.split('/')
            try:
                tvar = self._get_op_attr(target_node, op, f'{var}_old')
            except KeyError:
                tvar = self._get_op_attr(target_node, op, var)

            # define target index
            if delay is not None and tidx:
                tidx_tmp = []
                for idx, d in zip(tidx, delay):
                    if type(idx) is list:
                        tidx_tmp.append(idx + [d])
                    else:
                        tidx_tmp.append([idx, d])
                tidx = tidx_tmp
            elif not tidx and delay is not None:
                tidx = list(delay)

            # create mapping equation and its arguments
            d = "[target_idx]" if tidx else ""
            idx = "[source_idx]" if sidx else ""
            assign = '+=' if add_project else '='
            eq = f"target_var{d} {assign} source_var{idx} * weight"
            args = {'vars': {}, 'inputs': {}}
            args['vars']['weight'] = {'vtype': 'constant', 'dtype': svar.dtype, 'value': weight}
            if tidx:
                args['vars']['target_idx'] = {'vtype': 'constant', 'dtype': 'int32',
                                              'value': np.array(tidx, dtype=np.int32)}
            if sidx:
                args['vars']['source_idx'] = {'vtype': 'constant', 'dtype': 'int32',
                                              'value': np.array(sidx, dtype=np.int32)}
            args['inputs']['target_var'] = tvar
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

        self.edge_updates = [self.backend.layer] if self.backend.layers[self.backend.layer] else []

    def run(self,
            simulation_time: Optional[float] = None,
            inputs: Optional[dict] = None,
            outputs: Optional[dict] = None,
            sampling_step_size: Optional[float] = None,
            out_dir: Optional[str] = None,
            verbose: bool = True,
            profile: Optional[str] = None
            ) -> Union[DataFrame, Tuple[DataFrame, float, float]]:
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
        profile
            Can be used to extract information about graph execution time and memory load. Can be:
            - `t` for returning the total graph execution time.
            - `m` for returning the peak memory consumption during graph excecution.
            - `mt` or `tm` for both

        Returns
        -------
        Union[DataFrame, Tuple[DataFrame, float, float]]
            First entry of the tuple contains the output variables in a pandas dataframe, the second contains the
            simulation time in seconds and the third the peak memory consumption. If profiling was not chosen during
            call of the function, only the dataframe will be returned.

        """

        # prepare simulation
        ####################

        if verbose:
            print("Preparing simulation...")

        # basic simulation parameters initialization
        if not simulation_time:
            simulation_time = self.dt
        sim_steps = int(simulation_time / self.dt)

        if not sampling_step_size:
            sampling_step_size = self.dt
        sampling_steps = int(sampling_step_size / self.dt)

        # add output variables to the backend
        #####################################

        # define output variables
        output_col = {}
        output_cols = []
        output_keys = []
        output_shapes = []
        if outputs:
            for key, val in outputs.items():
                var_key, var_val = self.get_var(node=val[0], op=val[1], var=val[2], var_name=f"{key}_col").popitem()
                var_shape = tuple(var_val.shape)
                if var_shape in output_shapes:
                    idx = output_shapes.index(var_shape)
                    output_cols[idx].append(var_val)
                    output_keys[idx].append(var_key)
                else:
                    output_cols.append([var_val])
                    output_keys.append([var_key])
                    output_shapes.append(var_shape)

            # create counting index for collector variables
            output_col.update(self.backend.add_output_layer(outputs=output_cols,
                                                            sampling_steps=int(sim_steps/sampling_steps),
                                                            out_shapes=output_shapes))

        # add input variables to the backend
        ####################################

        if inputs:

            inp_dict = dict()

            # linearize input dictionary
            for key, val in inputs.items():

                if '_combined' in list(self.net_config.nodes.keys())[0]:

                    # fully vectorized case: add vectorized placeholder variable to input dictionary
                    var = self._get_node_attr(node=list(self.net_config.nodes.keys())[0], op=key[1], attr=key[2])
                    inp_dict[var.name] = np.reshape(val, (sim_steps,) + tuple(var.shape))

                elif any(['_all' in key_tmp for key_tmp in self.net_config.nodes.keys()]):

                    # node-vectorized case
                    if key[0] == 'all':

                        # go through all nodes, extract the variable and add it to input dict
                        i = 0
                        for node in self.net_config.nodes:
                            var = self._get_node_attr(node=node, op=key[1], attr=key[2])
                            i_new = var.shape[0] if len(var.shape) > 0 else 1
                            inp_dict[var.name] = np.reshape(val[:, i:i_new], (sim_steps,) + tuple(var.shape))
                            i += i_new

                    elif key[0] in self.net_config.nodes.keys():

                        # add placeholder variable of node(s) to input dictionary
                        var = self._get_node_attr(node=key[0], op=key[1], attr=key[2])
                        inp_dict[var.name] = np.reshape(val, (sim_steps,) + tuple(var.shape))

                    elif any([key[0] in key_tmp for key_tmp in self.net_config.nodes.keys()]) or \
                            any([key[0].split('.')[0] in key_tmp for key_tmp in self.net_config.nodes.keys()]):

                        key_tmp = key[0].split('.')[0] if '.' in key[0] else key[0]

                        # add vectorized placeholder variable of specified node type to input dictionary
                        for node in list(self.net_config.nodes.keys()):
                            if key_tmp in node:
                                break
                        var = self._get_node_attr(node=node, op=key[1], attr=key[2])
                        inp_dict[var.name] = np.reshape(val, (sim_steps,) + tuple(var.shape))

                else:

                    # non-vectorized case
                    if key[0] == 'all':

                        # go through all nodes, extract the variable and add it to input dict
                        for i, node in enumerate(self.net_config.nodes.keys()):
                            var = self._get_node_attr(node=node, op=key[1], attr=key[2])
                            inp_dict[var.name] = np.reshape(val[:, i], (sim_steps,) + tuple(var.shape))

                    elif any([key[0] in key_tmp for key_tmp in self.net_config.nodes.keys()]):

                        # extract variables from nodes of specified type
                        i = 0
                        for node in self.net_config.nodes.keys():
                            if key[0] in node:
                                var = self._get_node_attr(node=node, op=key[1], attr=key[2])
                                inp_dict[var.name] = np.reshape(val[:, i], (sim_steps,) + tuple(var.shape))
                                i += 1

            # add inputs to graph
            self.backend.add_layer(to_beginning=True)

            # create counting index for input variables
            in_idx = self.backend.add_var(vtype='state_var', name='in_var_idx', dtype='int32', shape=(1,), value=0,
                                          scope="network_inputs")

            for key, var in inp_dict.items():
                var_name = f"{var.short_name}_inp" if hasattr(var, 'short_name') else "var_inp"
                in_var = self.backend.add_var(vtype='state_var', name=var_name, scope="network_inputs", value=var)
                in_var_idx = self.backend.add_op('index', in_var, in_idx, scope="network_inputs")
                self.backend.add_op('=', self.backend.vars[key], in_var_idx, scope="network_inputs")

            # create increment operator for counting index
            self.backend.add_op('+=', in_idx, np.ones((1,), dtype='int32'), scope="network_inputs")

        # run simulation
        ################

        if verbose:
            print("Running simulation...")

        if profile is None:
            output_col = self.backend.run(steps=sim_steps, outputs=output_col, sampling_steps=sampling_steps,
                                          out_dir=out_dir, profile=profile)
        else:
            output_col, time, memory = self.backend.run(steps=sim_steps, outputs=output_col, out_dir=out_dir,
                                                        profile=profile, sampling_steps=sampling_steps)

        if verbose and profile:
            if simulation_time:
                print(f"{simulation_time}s of backend behavior were simulated in {time} s given a "
                      f"simulation resolution of {self.dt} s.")
            else:
                print(f"ComputeGraph computations finished after {time} seconds.")

        # store output variables in data frame
        ######################################

        # ungroup grouped output variables
        outputs = {}
        for names, group_key in zip(output_keys, output_col.keys()):
            out_group = output_col[group_key]
            for i, key in enumerate(names):
                outputs[key] = out_group[:, i]

        out_var_vals = []
        out_var_names = []
        for key in list(outputs):

            var = outputs.pop(key)
            if len(var.shape) > 1 and var.shape[1] > 1:
                for i in range(var.shape[1]):
                    var_tmp = var[:, i]
                    if len(var.shape) > 1:
                        var_tmp = np.squeeze(var_tmp)
                    out_var_vals.append(var_tmp)
                    key_split = key.split('/')
                    var_name = key_split[-1]
                    var_name = var_name[:var_name.find('_col')]
                    node_name = ""
                    for key_tmp in key_split[:-1]:
                        node_name += key_tmp
                    out_var_names.append((var_name, f'{node_name}_{i}'))
            else:
                if len(var.shape) > 1:
                    var = np.squeeze(var)
                out_var_vals.append(var)
                key_split = key.split('/')
                var_name = key_split[-1]
                var_name = var_name[:var_name.find('_col')]
                node_name = ""
                for key_tmp in key_split[:-1]:
                    node_name += key_tmp
                out_var_names.append((var_name, node_name))

        # create multi-index
        index = MultiIndex.from_tuples(out_var_names, names=['var', 'node'])

        # create dataframe
        if out_var_vals:
            data = np.asarray(out_var_vals).T
            if len(data.shape) > 2:
                data = data.squeeze()
            idx = np.arange(0., simulation_time, sampling_step_size)[-data.shape[0]:]
            out_vars = DataFrame(data=data[0:len(idx), :],
                                 index=idx,
                                 columns=index)
        else:
            out_vars = DataFrame()

        # return results
        ################

        if profile:
            return out_vars, time, memory
        return out_vars

    def get_var(self, node: str, op: str, var: str, var_name: Optional[str] = None, **kwargs) -> dict:
        """Extracts a variable from the graph.

        Parameters
        ----------
        node
            Name of the node(s), the variable exists on. Can be 'all' for all nodes, or a sub-string that defines a
            class of nodes or a specific node name.
        op
            Name of the operator the variable belongs to.
        var
            Name of the variable.
        var_name
            Name under which the variable should be returned
        kwargs
            Additional keyword arguments that may be used to pass arguments for the backend like name scopes.

        Returns
        -------
        dict
            Dictionary with all variables found in the network that match the provided signature.

        """

        if not var_name:
            var_name = var
        var_col = {}

        if node == 'all':

            # collect output variable from every node in backend
            for node in self.net_config.nodes.keys():
                var_col[f'{node}/{var_name}'] = self._get_node_attr(node=node, op=op, attr=var)

        elif node in self.net_config.nodes.keys() or node in self._net_config_map.keys():

            # get output variable of specific backend node
            var_col[f'{node}/{var_name}'] = self._get_node_attr(node=node, op=op, attr=var, **kwargs)

        elif any([node in key for key in self.net_config.nodes.keys()]):

            # get output variable from backend nodes of a certain type
            for node_tmp in self.net_config.nodes.keys():
                if node in node_tmp:
                    var_col[f'{node}/{var_name}'] = self._get_node_attr(node=node_tmp, op=op, attr=var, **kwargs)
        else:

            # get output variable of specific, vectorized backend node
            for node_tmp in self._net_config_map.keys():
                if node in node_tmp and '_all' in node_tmp:
                    var_col[f'{node}/{var_name}'] = self._get_node_attr(node=node_tmp, op=op, attr=var, **kwargs)

        return var_col

    def clear(self):
        """Clears the backend graph from all operations and variables.
        """
        self.backend.clear()

    def _add_ops(self, ops: List[str], node_name: str, updates: Optional[dict] = None, primary_ops: bool = False
                 ) -> dict:
        """Adds a number of operations to the backend graph.

        Parameters
        ----------
        ops
            Names of the operators that should be parsed into the graph.
        node_name
            Name of the node that the operators belong to.
        updates
            Key-value pairs of operations already parsed into the backend graph.
        primary_ops
            Indicates, whether the operations are the first to be parsed into the graph or not.

        Returns
        -------
        dict
            Key-value pairs of all operations currently parsed into the backend graph.

        """

        # set up update operation collector variable
        if updates is None:
            updates = {}
        updates_new = {}

        # add operations of same hierarchical lvl to compute graph
        ############################################################

        for op_name in ops:

            updates_new[op_name] = {}

            # retrieve operator and operator args
            op_args = dict()
            op_args['vars'] = self._get_op_attr(node_name, op_name, 'variables')
            op_args['vars']['dt'] = self._dt
            op_args['inputs'] = {}
            op_info = self._get_op_attr(node_name, op_name, 'operator')

            # handle operator inputs
            for var_name, inp in op_info['inputs'].items():

                # go through inputs to variable
                if inp['sources']:

                    in_ops_col = []
                    reduce_inputs = inp['reduce_dim'] if type(inp['reduce_dim']) is bool else False

                    for i, in_op in enumerate(inp['sources']):

                        if type(in_op) is list and len(in_op) > 1:

                            reduce_dim = inp['reduce_dim'][i]
                            in_ops_tmp = []

                            # collect multiple inputs to op
                            for op in in_op:
                                in_ops_tmp.append(self._get_op_input(node_name, op_name, op, primary_ops))

                            # map those inputs correctly
                            in_ops_col.append(self._map_multiple_inputs(in_ops_tmp, reduce_dim,
                                                                        scope=f"{self.name}/{node_name}/{op_name}")
                                              )

                        else:

                            if type(in_op) is list and len(in_op) == 1:
                                in_op = in_op[0]

                            # collect single input to op
                            in_ops_col.append(self._get_op_input(node_name, op_name, in_op, primary_ops))

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
                                for i in range(multiplier):
                                    in_ops.append(op[i * min_shape:(i + 1) * min_shape])
                            else:
                                in_ops.append(op)

                        # map inputs to target
                        in_ops = self._map_multiple_inputs(in_ops, reduce_inputs,
                                                           scope=f"{self.name}/{node_name}/{op_name}")

                    # for a single input variable
                    else:
                        in_ops = in_ops_col[0]

                    # add input variable to dictionary
                    op_args['inputs'][var_name] = in_ops

            # parse equations into tensorflow
            op_args = parse_equation_list(op_info['equations'], op_args, backend=self.backend,
                                          scope=f"{self.name}/{node_name}/{op_name}")

            # store operator variables in net config
            op_vars = self._get_op_attr(node_name, op_name, 'variables')
            for var_key, var_val in op_args['vars'].items():
                if var_key not in op_vars:
                    op_vars[var_key] = var_val

            # store update ops in node update collector
            for var in list(set(op_args['lhs_evals'])):
                updates_new[op_name][var] = op_args['updates'][var]

        return updates

    def _get_op_input(self, node: str, op: str, in_op: str, primary_ops: bool = False
                      ) -> Any:
        """Extracts inputs of operator.

        Parameters
        ----------
        node
            Name of the node the operators belong to.
        op
            Name of the target node.
        in_op
            Name of the source node.

        Returns
        -------
        Any
            Backend variable.

        """

        if primary_ops:
            raise ValueError(f'Input found in primary operator {op} on node {node}. '
                             f'This operator should have no node-internal inputs. '
                             f'Please move the operator inputs to the node level or change the '
                             f'input-output relationships between the node operators.')
        else:
            var_name = self._get_op_attr(node, in_op, 'output', retrieve=False)
            var = self._get_op_attr(node, in_op, var_name)
            if type(var) is str:
                raise ValueError(f'Wrong operator order on node {node}. Operator {op} '
                                 f'needs input from operator {in_op} which has not been '
                                 f'processed yet. Please consider changing the operator order '
                                 f'or dependencies.')
        return var

    def _map_multiple_inputs(self, inputs: list, reduce_dim: bool, **kwargs) -> Any:
        """Creates mapping between multiple input variables and a single output variable.

        Parameters
        ----------
        inputs
            Input variables.
        reduce_dim
            If true, input variables will be summed up, if false, they will be concatenated.
        kwargs
            Additional keyword arguments for the backend.

        Returns
        -------
        Any
            Summed up or concatenated input variables.

        """

        inp = self.backend.stack_vars(inputs, **kwargs)
        self.backend.next_layer()
        if reduce_dim:
            inp_transform = self.backend.add_op('sum', inp, 0)
        else:
            inp_transform = self.backend.add_op('reshape', inp, (inp.shape[0] * inp.shape[1],), **kwargs) \
                if len(inp.shape) > 1 else inp
        self.node_updates.append(self.backend.layer)
        return inp_transform

    def _get_node_attr(self, node: str, attr: str, op: Optional[str] = None, **kwargs) -> Any:
        """Extract attribute from node of the network.

        Parameters
        ----------
        node
            Name of the node.
        attr
            Name of the attribute on the node.
        op
            Name of the operator. Only needs to be provided for operator variables.

        Returns
        -------
        Any
            Node attribute.

        """

        if op:
            return self._get_op_attr(node, op, attr, **kwargs)
        try:
            return self.net_config[node][attr]
        except KeyError:
            vals = []
            for op in self.net_config[node]:
                vals.append(self._get_op_attr(node, op, attr, **kwargs))
            return vals

    def _set_node_attr(self, node: str, attr: str, val: Any, op: Optional[str] = None) -> Any:
        """Sets attribute on a node.

        Parameters
        ----------
        node
            Name of the node.
        attr
            Name of the attribute.
        op
            Name of the operator, the attribute belongs to. Does not need to be provided for attributes that do not
            belong to an operator.

        Returns
        -------
        Any
            New node attribute value.

        """

        if op:
            return self._set_op_attr(node, op, attr, val)

        try:
            self.net_config.nodes[node]['node'][attr] = val
        except KeyError:
            vals_updated = []
            for op in self.net_config.nodes[node]['node']['op_graph'].nodes:
                vals_updated.append(self._set_op_attr(node, op, attr, val))
            return vals_updated

    def _get_edge_attr(self, source: str, target: str, edge: Union[str, int], attr: str,
                       retrieve_from_node: bool = True) -> Any:
        """Extracts attribute from edge.

        Parameters
        ----------
        source
            Name of the source node.
        target
            Name of the target node.
        edge
            Index or name of the edge.
        attr
            Name of the attribute
        retrieve_from_node
            If true, the attribute will be retrieved from the source or target node. Only relevant if the attribute is
            'source_var' or 'target_var'. Else, the value of the attribute on the edge will be returned.

        Returns
        -------
        Any
            Edge attribute.

        """

        try:
            attr_val = self.net_config.edges[source, target, edge][attr]
            if 'var' in attr and type(attr_val) is str and retrieve_from_node:
                op, var = attr_val.split('/')
                if 'source' in attr:
                    attr_val = self._get_op_attr(source, op, var)
                else:
                    attr_val = self._get_op_attr(target, op, var)
        except KeyError:
            attr_val = None

        return attr_val

    def _set_edge_attr(self, source: str, target: str, edge: Union[int, str], attr: str, val: Any) -> Any:
        """Sets value of an edge attribute.

        Parameters
        ----------
        source
            Name of the source node.
        target
            Name of the target node.
        edge
            Name or index of the edge.
        attr
            Name of the attribute.
        val
            New value of the attribute.

        Returns
        -------
        Any
            Value of the updated edge attribute.

        """

        self.net_config.edges[source, target, edge][attr] = val
        return self.net_config.edges[source, target, edge][attr]

    def _get_op_attr(self, node: str, op: str, attr: str, retrieve: bool = True, **kwargs) -> Any:
        """Extracts attribute of an operator.

        Parameters
        ----------
        node
            Name of the node.
        op
            Name of the operator on the node.
        attr
            Name of the attribute of the operator on the node.
        retrieve
            If attribute is output, this can be set to True to receive the handle to the output variable, or to false
            to receive the name of the output variable.

        Returns
        -------
        Any
            Operator attribute.

        """

        if node in self._net_config_map and op in self._net_config_map[node] and attr in self._net_config_map[node][op]:
            node, op, attr, attr_idx = self._net_config_map[node][op][attr]
            idx = f"{list(attr_idx)}" if type(attr_idx) is tuple else attr_idx
            return self.backend.apply_idx(self._get_op_attr(node, op, attr), idx) if retrieve else attr_idx
        elif node in self.net_config:
            op = self.net_config[node]['op_graph'].nodes[op]
        else:
            raise ValueError(f'Node with name {node} is not part of this network.')

        if attr == 'output' and retrieve:
            attr = op['operator']['output']

        if attr in op['variables'].keys():
            attr_val = op['variables'][attr]
        elif hasattr(op['operator'], 'keys') and attr in op['operator'].keys():
            attr_val = op['operator'][attr]
        elif hasattr(op['operator'], attr):
            attr_val = getattr(op['operator'], attr)
        else:
            try:
                attr_val = op[attr]
            except KeyError as e:
                try:
                    attr_val = getattr(op, attr)
                except AttributeError:
                    raise e

        return attr_val

    def _set_op_attr(self, node: str, op: str, attr: str, val: Any) -> Any:
        """Sets value of an operator attribute.

        Parameters
        ----------
        node
            Name of the node.
        op
            Name of the operator on the node.
        attr
            Name of the operator attribute on the node.
        val
            New value of the operator attribute.

        Returns
        -------
        Any
            Updated attribute value.

        """

        op = self.net_config[node]['op_graph'].nodes[op]
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

    @staticmethod
    def _apply_idx(var: Any, idx: Optional[Union[int, tuple]] = None) -> Any:
        """Applies index to graph variable. Only used for observing graph variables,
        not for graph internal computations.

        Parameters
        ----------
        var
            Graph variable.
        idx
            Index to the variable dimensions.

        Returns
        -------
        Any
            Indexed variable.

        """

        if idx is None:
            return var
        elif type(idx) is tuple:
            return var[idx[0]:idx[1]]
        else:
            return var[idx]

    def _get_nodes_with_attr(self, attr: str, val: Any) -> list:
        """Extracts nodes from graph for which a certain attribute takes on a certain value.

        Parameters
        ----------
        attr
            Attribute of interest on the nodes.
        val
            Attribute value that selected nodes should have.

        Returns
        -------
        list
            Nodes for which the attribute takes on the indicated value.

        """

        nodes = []
        for node in self.net_config.nodes:
            test_val = self._get_node_attr(node, attr)
            if hasattr(val, 'nodes') and val.nodes.keys() == test_val.nodes.keys():
                nodes.append(node)
            elif val == test_val:
                nodes.append(node)

        return nodes

    def _contains_node(self, node: str, target_node: str) -> bool:
        """Checks whether a vectorized graph node contains an original, non-vectorized target node or not.

        Parameters
        ----------
        node
            Vectorized graph node.
        target_node
            Less vectorized node.

        Returns
        -------
        bool
            Indicates whether target node is contained in the vectorized node or not.

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

    def _get_edges_between(self, source: str, target: str) -> List[tuple]:
        """Extract all edges between a source and a target node.

        Parameters
        ----------
        source
            Name of the source node.
        target
            Name of the target node.

        Returns
        -------
        List[tuple]
            Collection of all edges between source and target.

        """

        if ('_all' in source and '_all' in target) or ('_combined' in source and '_combined' in target):
            edges = []
            for source_tmp in self._net_config_map:
                for target_tmp in self._net_config_map:
                    if self._contains_node(source, source_tmp) and self._contains_node(target, target_tmp):
                        edges += [(source_tmp, target_tmp, edge) for edge
                                  in range(self.net_config.graph.number_of_edges(source_tmp, target_tmp))]
            return edges
        else:
            return [(source, target, edge) for edge in range(self.net_config.graph.number_of_edges(source, target))]

    def _get_edge_conn(self, source: str, target: str, edge: Union[int, str]) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts weight and delay vectors of an edge.

        Parameters
        ----------
        source
            Name of the source node.
        target
            Name of the target node.
        edge
            Name of index of the edge.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Weight and delay of the edge (both are arrays).

        """

        # check delay of edge
        delay = self._get_edge_attr(source, target, edge, 'delay')
        if delay is not None and delay.ndim > 1:
            raise ValueError(f"Automatic optimization of the graph (i.e. method `vectorize`"
                             f" cannot be applied to networks with variables of 2 or more"
                             f" dimensions. Delay of edge {edge} between {source} and {target} has shape"
                             f" {delay.shape}. Please turn of the `vectorize` option or change"
                             f" the edges' dimensionality.")

        # check weight of edge
        weight = self._get_edge_attr(source, target, edge, 'weight')
        if weight is not None and weight.ndim > 1:
            raise ValueError(f"Automatic optimization of the graph (i.e. method `vectorize`"
                             f" cannot be applied to networks with variables of 2 or more"
                             f" dimensions. Weight of edge {edge} between {source} and {target} has shape"
                             f" {weight.shape}. Please turn of the `vectorize` option or"
                             f" change the edges' dimensionality.")

        # match dimensionality of delay and weight
        if delay is None or weight is None:
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

    def _get_edge_var_idx(self, source: str, target: str, edge: Union[int, str], idx_type: str) -> Union[list, None]:
        """Extracts indices of an edge that map vectorized weights and delays to the original edges.

        Parameters
        ----------
        source
            Name of the source node.
        target
            Name of the target node.
        edge
            Index or name of the edge.

        Returns
        -------
        Union[list, None]
            List of indices.

        """

        if idx_type == 'source':
            var = 'source_idx'
            node_var = 'source_var'
            node_to_idx = source
        elif idx_type == 'target':
            var = 'target_idx'
            node_var = 'target_var'
            node_to_idx = target
        else:
            raise ValueError('Wrong `idx_type`. Please, choose either `source` or `target`.')

        var_idx = self._get_edge_attr(source, target, edge, var)
        if var_idx:
            op, var = self.net_config.edges[source, target, edge][node_var].split('/')
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
                op, var = self.net_config.edges[source, target, edge][node_var].split('/')
                _, _, _, idx_new = self._net_config_map[node_to_idx][op][var]
                if type(idx_new) is int:
                    idx_new = [idx_new]
                else:
                    idx_new = list(idx_new)
            except KeyError:
                idx_new = None

        return idx_new

    def _sort_edges(self, edges: List[tuple], attr: str) -> dict:
        """Sorts edges according to the given edge attribute.

        Parameters
        ----------
        edges
            Collection of edges of interest.
        attr
            Name of the edge attribute.

        Returns
        -------
        dict
            Key-value pairs of the different values the attribute can take on (keys) and the list of edges for which
            the attribute takes on that value (value).

        """

        edges_new = {}
        edge_idx = {}
        for edge in edges:
            if len(edge) == 3:
                source, target, edge = edge
            else:
                source, target = edge
                if (source, target) not in edge_idx:
                    edge_idx[(source, target)] = 0
                edge = edge_idx[(source, target)]
                edge_idx[(source, target)] += 1
            edge_info = self.net_config.edges[source, target, edge]
            if edge_info[attr] not in edges_new.keys():
                edges_new[edge_info[attr]] = [(source, target, edge)]
            else:
                edges_new[edge_info[attr]].append((source, target, edge))

        return edges_new

    def _add_edge_buffer(self, node: str, op: str, var: str, idx: int, buffer_length: int, edge: tuple) -> None:
        """Adds a buffer variable to an edge.

        Parameters
        ----------
        node
            Name of the target node of the edge.
        op
            Name of the target operator of the edge.
        var
            Name of the target variable of the edge.
        idx
            Index of the buffer variable for that specific edge.
        buffer_length
            Length of the time-buffer that should be added to realize edge delays.
        edge
            Edge identifier (source_name, target_name, edge_idx).

        Returns
        -------
        None

        """

        target_shape = self._get_op_attr(node, op, var)['shape']
        op_graph = self._get_node_attr(node, 'op_graph')

        # create buffer variable definitions
        if len(target_shape) < 1 or (len(target_shape) == 1 and target_shape[0] == 1):
            buffer_shape = (buffer_length + 1,)
            buffer_shape_reset = (1,)
        else:
            buffer_shape = (target_shape[0], buffer_length + 1)
            buffer_shape_reset = (target_shape[0], 1)
        var_dict = {f'{var}_buffer_{idx}': {'vtype': 'state_var',
                                            'dtype': self._float_precision,
                                            'shape': buffer_shape,
                                            'value': 0.
                                            },
                    f'{var}_buffer_{idx}_reset': {'vtype': 'constant',
                                                  'dtype': self._float_precision,
                                                  'shape': buffer_shape_reset,
                                                  'value': 0.
                                                  }
                    }

        # create buffer equations
        if len(target_shape) < 1 or (len(target_shape) == 1 and target_shape[0] == 1):
            eqs_op_read = [f"{var} = {var}_buffer_{idx}[0]"]
            eqs_op_rotate = [f"{var}_buffer_{idx} = concat(({var}_buffer_{idx}[1:], {var}_buffer_{idx}_reset), 0)"]
        else:
            eqs_op_read = [f"{var} = {var}_buffer_{idx}[:, 0]"]
            eqs_op_rotate = [f"{var}_buffer_{idx} = concat(({var}_buffer_{idx}[:, 1:], {var}_buffer_{idx}_reset), 1)"]

        # add buffer operators to operator graph
        op_graph.add_node(f'{op}_{var}_buffer_rotate_{idx}',
                          operator={'inputs': {},
                                    'output': f'{var}_buffer_{idx}',
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
        inputs = self._get_op_attr(node, op, 'inputs')
        if var in inputs.keys():
            inputs[var]['sources'].append(f'{op}_{var}_buffer_read_{idx}')
        else:
            inputs[var] = {'sources': [f'{op}_{var}_buffer_read_{idx}'],
                           'reduce_dim': True}

        # update edge information
        s, t, e = edge
        self._set_edge_attr(s, t, e, 'target_var', f'{op}_{var}_buffer_rotate_{idx}/{var}_buffer_{idx}')
        self._set_edge_attr(s, t, e, 'add_project', val=True)

    def _add_edge_input_collector(self, node: str, op: str, var: str, idx: int, edge: tuple) -> None:
        """Adds an input collector variable to an edge.

        Parameters
        ----------
        node
            Name of the target node of the edge.
        op
            Name of the target operator of the edge.
        var
            Name of the target variable of the edge.
        idx
            Index of the input collector variable on that edge.
        edge
            Edge identifier (source_name, target_name, edge_idx).

        Returns
        -------
        None

        """

        target_shape = self._get_op_attr(node, op, var)['shape']
        op_graph = self._get_node_attr(node, 'op_graph')

        # create collector equation
        eqs = [f"{var} = {var}_col_{idx}"]

        # create collector variable definition
        var_dict = {f'{var}_col_{idx}': {'vtype': 'state_var',
                                         'dtype': self._float_precision,
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
        op_inputs = self._get_op_attr(node, op, 'inputs')
        if var in op_inputs.keys():
            op_inputs[var]['sources'].append(f'{op}_{var}_col_{idx}')
        else:
            op_inputs[var] = {'sources': [f'{op}_{var}_col_{idx}'],
                              'reduce_dim': True}

        # update edge target information
        s, t, e = edge
        self._set_edge_attr(s, t, e, 'target_var', f'{op}_{var}_col_{idx}/{var}_col_{idx}')

    def _net_config_consistency_check(self, net_config: CircuitIR) -> CircuitIR:
        """Checks whether the passed network configuration follows the expected intermediate representation structure.

        Parameters
        ----------
        net_config
            Intermediate representation of the network configuration that should be translated into the backend.

        Returns
        -------
        CircuitIR
            The checked (and maybe slightly altered) network configuration.

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
                                             'dtype': self._float_precision,
                                             'shape': (1,),
                                             },
                               'constant': {'value': KeyError,
                                            'shape': (1,),
                                            'dtype': self._float_precision},
                               'placeholder': {'dtype': self._float_precision,
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
                        _ = var['value']
                    except KeyError:
                        if var_defs['value'] is KeyError:
                            raise KeyError(f'Field `value` not found in variable definition of variable {var_name} of '
                                           f'operator {op_name} on node {node_name}, but it is needed for variables of '
                                           f'type {vtype}.')
                        elif var_defs['value'] is not None:
                            var['value'] = var_defs['value']

                    # check the shape of the variable
                    try:
                        shape = var['shape']
                        if len(shape) == 0:
                            var['shape'] = (1,)
                    except KeyError:
                        var['shape'] = var_defs['shape']

                    # check the data type of the variable
                    try:
                        _ = var['dtype']
                    except KeyError:
                        var['dtype'] = var_defs['dtype']

                    if 'value' in var.keys() and not hasattr(var['value'], 'shape'):
                        var['value'] = np.zeros(var['shape'], dtype=var['dtype']) + var['value']

                # check whether the equations, inputs and output fields exist on the operator field
                op_fields = ['equations', 'inputs', 'output']
                for field in op_fields:
                    try:
                        _ = getattr(op_info, field)
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
                _ = edge['source_var']
            except KeyError:
                raise KeyError(f'Field `source_var` not found on edge {idx} between {source} and {target}. Every edge '
                               f'needs information about the variable on the source node it should access that needs '
                               f'to be provided at this position.')
            try:
                _ = edge['target_var']
            except KeyError:
                raise KeyError(f'Field `target_var` not found on edge {idx} between {source} and {target}. Every edge '
                               f'needs information about the variable on the target node it should access that needs '
                               f'to be provided at this position.')

            # check weight of edge
            try:
                weight = edge['weight']
                if not hasattr(weight, 'shape') or not weight.shape:
                    weight = np.ones((1,), dtype=self._float_precision) * weight
                elif 'float' not in str(weight.dtype):
                    weight = np.asarray(weight, dtype=self._float_precision)
                edge['weight'] = weight
            except KeyError:
                edge['weight'] = np.ones((1,), dtype=self._float_precision)

            # check delay of edge
            try:
                delay = edge['delay']
                if delay is not None:
                    if not hasattr(delay, 'shape') or not delay.shape:
                        delay = np.zeros((1,)) + delay
                    if 'float' in str(delay.dtype):
                        delay = np.asarray((delay / self.dt) + 1, dtype=np.int32)
                    edge['delay'] = delay
            except KeyError:
                edge['delay'] = None

        return net_config

    def _vectorize(self, vectorization_mode: Optional[str] = 'nodes') -> None:
        """Method that goes through the nodes and edges dicts and vectorizes those that are governed by the same
        operators/equations.

        Parameters
        ----------
        vectorization_mode
            Can be 'none' for no vectorization, 'nodes' for vectorization over nodes and 'ops' for vectorization over
            nodes and operations.

        Returns
        -------
        None

        """

        # First stage: Vectorize over nodes
        ###################################

        if vectorization_mode in 'nodesfull':

            nodes = list(self.net_config.nodes.keys())

            # go through each node in net config and vectorize it with nodes that have the same operator structure
            ######################################################################################################

            while nodes:

                # get nodes with same operators
                op_graph = self._get_node_attr(nodes[0], 'op_graph')
                nodes_tmp = self._get_nodes_with_attr('op_graph', op_graph)

                # vectorize those nodes
                self._vectorize_nodes(list(nodes_tmp))

                # delete vectorized nodes from list
                for n in nodes_tmp:
                    nodes.pop(nodes.index(n))

            # adjust edges accordingly
            ##########################

            # go through new nodes
            for source in self.net_config.nodes.keys():
                for target in self.net_config.nodes.keys():
                    if '_all' in source and '_all' in target:
                        self._vectorize_edges(source, target)

            # save changes to net config
            for node in list(self.net_config.nodes):
                if '_all' not in node:
                    self.net_config.graph.remove_node(node)

        # Second stage: Vectorize over operators
        ########################################

        if vectorization_mode == 'full':

            # create dictionary of operators of each node that will be used to check whether they have been vectorized
            vec_info = {}
            for node in self.net_config.nodes:
                vec_info[node] = {}
                for op in self._get_node_attr(node, 'op_graph').nodes:
                    vec_info[node][op] = False

            # add new node to net config
            new_node = DiGraph()
            new_node_name = f'{self.name}_combined'
            self.net_config.add_node(new_node_name, node={'op_graph': new_node})
            new_node_name += '.0'

            # go through nodes and vectorize over their operators
            node_keys = list(vec_info)
            node_idx = 0
            node_key = node_keys[node_idx]
            while not all([all(node.values()) for node in vec_info.values()]):

                changed_node = False
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

                            for node_key_tmp in vec_info:

                                if node_key_tmp != node_key and op_key in vec_info[node_key_tmp]:

                                    # check if operator dependencies are vectorized already
                                    deps_vectorized = True
                                    op_graph_tmp = self._get_node_attr(node_key_tmp, 'op_graph')
                                    for op_key_tmp in op_graph_tmp.nodes:
                                        if op_key_tmp == op_key:
                                            for edge in op_graph_tmp.in_edges(op_key_tmp):
                                                if not vec_info[node_key_tmp][edge[0]]:
                                                    node_key = node_key_tmp
                                                    deps_vectorized = False

                                    if deps_vectorized:
                                        nodes_to_vec.append(node_key_tmp)
                                    else:
                                        changed_node = True
                                        break

                            if nodes_to_vec and not changed_node:

                                # vectorize op
                                nodes_to_vec.append(node_key)
                                self._vectorize_ops(op_key=op_key,
                                                    nodes=nodes_to_vec.copy(),
                                                    new_node=new_node,
                                                    new_node_name=new_node_name)

                                # indicate where vectorization was performed
                                for node_key_tmp in nodes_to_vec:
                                    vec_info[node_key_tmp][op_key] = True

                                # remove vectorized operation from graph
                                op_graph.remove_node(op_key)

                            elif changed_node:
                                break
                            else:

                                # add operation to new net configuration
                                self._vectorize_ops(op_key=op_key,
                                                    nodes=[node_key],
                                                    new_node=new_node,
                                                    new_node_name=new_node_name)

                                # mark operation on node as checked
                                vec_info[node_key][op_key] = True

                                # remove vectorized operation from graph
                                op_graph.remove_node(op_key)

                        else:

                            # remove operation from graph
                            op_graph.remove_node(op_key)

                    if changed_node:
                        break

                # increment node
                node_idx = node_idx + 1 if node_idx < len(vec_info) - 1 else 0
                node_key = node_keys[node_idx]

            # add dependencies between operators
            for target_op in self._get_node_attr(new_node_name, 'op_graph'):
                op = self._get_node_attr(new_node_name, 'operator', target_op)
                for inp in op['inputs'].values():
                    for source_op in inp['sources']:
                        if type(source_op) is list:
                            for sop in source_op:
                                new_node.add_edge(sop, target_op)
                        else:
                            new_node.add_edge(source_op, target_op)

            # adjust edges accordingly
            ##########################

            self._vectorize_edges(new_node_name, new_node_name)

            # delete vectorized nodes from graph
            for n in list(self.net_config.nodes):
                if new_node_name not in n:
                    self.net_config.graph.remove_node(n)

        # Third Stage: Finalize edge vectorization
        ##########################################

        # go through nodes and create mapping for their inputs
        for node_name, node in self.net_config.nodes.items():

            node_inputs = self.net_config.graph.in_edges(node_name)
            node_inputs = self._sort_edges(node_inputs, 'target_var')

            # loop over input variables of node
            for i, (in_var, edges) in enumerate(node_inputs.items()):

                # extract info for input variable connections
                n_inputs = len(edges)
                op_name, var_name = in_var.split('/')
                delays = []
                for s, t, e in edges:
                    d = self._get_edge_attr(s, t, e, 'delay')
                    if d is not None:
                        delays.append(d)
                max_delay = np.max(delays) if delays else None

                # loop over different input sources
                for j in range(n_inputs):

                    if max_delay is not None:

                        # add synaptic buffer to the input variable
                        self._add_edge_buffer(node_name, op_name, var_name, idx=j, buffer_length=max_delay,
                                              edge=edges[j])

                    elif n_inputs > 1:

                        # add synaptic input collector to the input variable
                        self._add_edge_input_collector(node_name, op_name, var_name, idx=j, edge=edges[j])

    def _vectorize_nodes(self,
                         nodes: Union[List[str]]
                         ) -> str:
        """Combines all nodes in list to a single node and adds node to new net config.

        Parameters
        ----------
        nodes
            Names of the nodes to be combined.

        Returns
        -------
        str
            name of the new node that was added to the net config.

        """

        n_nodes = len(nodes)

        # instantiate new node
        ######################

        node_ref = nodes.pop()
        new_node = deepcopy(self.net_config.nodes[node_ref])
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
            if f"{node_name}.{node_idx}" in self.net_config.nodes.keys():
                node_idx += 1
            else:
                node_name += f".{node_idx}"
                break

        # add new node to net config
        self.net_config.add_node(node_name, **new_node)

        # go through node attributes and store their value and shape
        arg_vals, arg_shapes = {}, {}
        for op_name in self._get_node_attr(node_ref, 'op_graph').nodes.keys():

            arg_vals[op_name] = {}
            arg_shapes[op_name] = {}
            if op_name not in self._net_config_map[node_ref].keys():
                self._net_config_map[node_ref][op_name] = {}

            for arg_name, arg in self._get_op_attr(node_ref, op_name, 'variables').items():

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

                    arg = self._get_node_attr(node, arg_name, op=op_name)

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
        for op_name in self._get_node_attr(node_name, 'op_graph').nodes.keys():
            for arg_name, arg in self._get_op_attr(node_name, op_name, 'variables').items():
                if 'value' in arg.keys():
                    arg.update({'value': arg_vals[op_name][arg_name]})
                if 'shape' in arg.keys():
                    arg.update({'shape': (n_nodes,) + arg_shapes[op_name][arg_name]})

        return node_name

    def _vectorize_edges(self, source_name: str, target_name: str) -> None:
        """Combines edges in list and adds a new edge to the new net config.

        Parameters
        ----------
        source_name
            Name of the source node
        target_name
            Name of the target node

        Returns
        -------
        None

        """

        # get edges between source and target
        edges = self._get_edges_between(source_name, target_name)

        # extract edges that connect the same variables on source and target
        ####################################################################

        while edges:

            source_tmp, target_tmp, edge_tmp = edges[0]

            # get source and target variable
            source_var = self._get_edge_attr(source_tmp, target_tmp, edge_tmp, 'source_var', retrieve_from_node=False)
            target_var = self._get_edge_attr(source_tmp, target_tmp, edge_tmp, 'target_var', retrieve_from_node=False)

            # get edges with equal source and target variables between source and target node
            edges_tmp = []
            for n, (source_tmp, target_tmp, edge_tmp) in enumerate(edges):
                if self._get_edge_attr(source_tmp, target_tmp, edge_tmp,
                                       'source_var', retrieve_from_node=False) == source_var and \
                        self._get_edge_attr(source_tmp, target_tmp, edge_tmp,
                                            'target_var', retrieve_from_node=False) == target_var:
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
                    if delay is not None:
                        delay_col.append(delay)
                    if weight is not None:
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
                self.net_config.graph.add_edge(source_name, target_name, **new_edge)

            # delete vectorized edges from list
            for edge in edges_tmp:
                edges.pop(edges.index(edge))

    def _vectorize_ops(self,
                       op_key: str,
                       nodes: list,
                       new_node: DiGraph,
                       new_node_name: str
                       ) -> None:
        """Vectorize all instances of an operation across nodes and put them into a single-node backend.

        Parameters
        ----------
        op_key
            Name of the operator that should be vectorized.
        nodes
            Collection of node names on which the operator exists.
        new_node
            Name of the new node that should be added to the graph.

        Returns
        -------
        None

        """

        # extract operation in question
        node_name_tmp = nodes.pop(0)
        op_inputs = self._get_node_attr(node_name_tmp, 'inputs', op_key)

        if node_name_tmp not in self._net_config_map:
            self._net_config_map[node_name_tmp] = {}
        if op_key not in self._net_config_map[node_name_tmp]:
            self._net_config_map[node_name_tmp][op_key] = {}

        # collect input dependencies of all nodes
        #########################################

        nodes_tmp = [node_name_tmp] + nodes
        if len(nodes_tmp) > 1:
            for node in nodes_tmp:
                op_inputs_tmp = self._get_node_attr(node, 'inputs', op_key)
                for key, arg in op_inputs_tmp.items():
                    if type(op_inputs[key]['reduce_dim']) is bool:
                        op_inputs[key]['reduce_dim'] = [arg['reduce_dim']]
                        op_inputs[key]['sources'] = [arg['sources']]
                    else:
                        if arg['sources'] not in op_inputs[key]['sources']:
                            op_inputs[key]['sources'].append(arg['sources'])
                            op_inputs[key]['reduce_dim'].append(arg['reduce_dim'])

        # extract operator-specific arguments and change their shape and value
        ######################################################################

        op_args = {}
        for key, arg in self._get_op_attr(node_name_tmp, op_key, 'variables').items():

            arg = deepcopy(arg)

            if arg['vtype'] == 'raw':

                if type(arg['value']) is list:

                    self._net_config_map[node_name_tmp][op_key][key] = [new_node_name, op_key, key,
                                                                        (0, len(arg['value']))]

                    for node_name in nodes:

                        if node_name not in self._net_config_map:
                            self._net_config_map[node_name] = {}
                        if op_key not in self._net_config_map[node_name]:
                            self._net_config_map[node_name][op_key] = {}

                        new_val = self._get_op_attr(node_name, op_key, key)['value']
                        old_idx = len(arg['value'])
                        arg['value'].append(new_val)

                        self._net_config_map[node_name][op_key][key] = [new_node_name, op_key, key,
                                                                        (old_idx, arg['value'].shape[0])]

                else:

                    # add information to _net_config_map
                    self._net_config_map[node_name_tmp][op_key][key] = [new_node_name, op_key, key, None]

            else:

                if not hasattr(arg['value'], 'shape'):
                    arg['value'] = np.array(arg['value'])
                if arg['value'].shape != arg['shape']:
                    try:
                        arg['value'] = np.reshape(arg['value'], arg['shape'])
                    except ValueError:
                        arg['value'] = np.zeros(arg['shape'], dtype=self._float_precision) + arg['value']

                # go through nodes and extract shape and value information for arg
                if nodes:

                    self._net_config_map[node_name_tmp][op_key][key] = [new_node_name, op_key, key,
                                                                        (0, arg['value'].shape[0])]

                    for node_name in nodes:

                        if node_name not in self._net_config_map:
                            self._net_config_map[node_name] = {}
                        if op_key not in self._net_config_map[node_name]:
                            self._net_config_map[node_name][op_key] = {}

                        # extract node arg value and shape
                        arg_tmp = self._get_op_attr(node_name, op_key, key)
                        val = arg_tmp['value']
                        dims = tuple(arg_tmp['shape'])

                        # append value to argument dictionary
                        if len(dims) > 0 and type(val) is float:
                            val = np.zeros(dims, dtype=self._float_precision) + val
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
                        self._net_config_map[node_name][op_key][key] = [new_node_name, op_key, key, idx]

                else:

                    # add information to _net_config_map
                    self._net_config_map[node_name_tmp][op_key][key] = [new_node_name, op_key, key,
                                                                        (0, arg['value'].shape[0])]

                # change shape of argument
                arg['shape'] = arg['value'].shape

                # add argument to new node's argument dictionary
                op_args[key] = arg

        # add operator to node
        new_node.add_node(op_key,
                          operator=self._get_op_attr(node_name_tmp, op_key, 'operator'),
                          variables=op_args)
