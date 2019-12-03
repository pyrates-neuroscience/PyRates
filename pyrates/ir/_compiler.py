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
from warnings import filterwarnings

# pyrates imports
from pyrates.backend.numpy_backend import NumpyBackend as _NumpyBackend
from pyrates.backend.parser import parse_equation_system, parse_dict
from pyrates.ir.circuit import CircuitIR
from pyrates.frontend import CircuitTemplate
from pyrates.backend.parser import replace

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


class Compiler:
    """Helper class that contains all methods to compile a full circuit IR into its backend representation.

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
    backend
        Backend in which to build the compute graph.
    solver
        Numerical solver to use for differential equations.

    """

    def __init__(self,
                 dt: float = 1e-3,
                 backend: Any = _NumpyBackend,
                 solver: str = 'euler',
                 float_precision: str = 'float32',
                 label: str = "circuit",
                 **kwargs
                 ) -> None:
        """Instantiates operator.
        """

        # filterwarnings("ignore", category=FutureWarning)

        # set basic attributes
        ######################

        self._float_precision = float_precision

        # instantiate the backend and set the backend default_device

        self.backend = backend
        self.solver = solver

        # pre-process the network configuration
        self.dt = dt
        # self._net_config_map = {}

        # set time constant of the network
        self.dt_parsed = parse_dict({'step_size': {'vtype': 'constant', 'dtype': self._float_precision, 'shape': (),
                                             'value': self.dt}},
                                    backend=self.backend)['step_size']

    def compile(self, circuit):
        """Transform CiruitIR instance into a form suitable for computation."""

        pass

    def run(self,
            simulation_time: Optional[float] = None,
            inputs: Optional[dict] = None,
            outputs: Optional[dict] = None,
            sampling_step_size: Optional[float] = None,
            out_dir: Optional[str] = None,
            verbose: bool = True,
            profile: Optional[str] = None,
            **kwargs
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

        filterwarnings("ignore", category=FutureWarning)

        # prepare simulation
        ####################

        if verbose:
            print("Preparing the simulation...")

        if not self._first_run:
            self.backend.remove_layer(0)
            self.backend.remove_layer(self.backend.top_layer())
        else:
            self._first_run = False

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
                val_split = val.split('/')
                node, op, var = "/".join(val_split[:-2]), val_split[-2], val_split[-1]
                for var_key, var_val in self.get_var(node=node, op=op, var=var, var_name=f"{key}_col").items():
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
                                                                sampling_steps=int(sim_steps / sampling_steps),
                                                                out_shapes=output_shapes))

        # add input variables to the backend
        ####################################

        if inputs:

            inp_dict = dict()

            # linearize input dictionary
            for key, val in inputs.items():

                key_split = key.split('/')
                node, op, attr = "/".join(key_split[:-2]), key_split[-2], key_split[-1]

                if '_combined' in list(self.net_config.nodes.keys())[0]:

                    # fully vectorized case: add vectorized placeholder variable to input dictionary
                    var = self._get_node_attr(node=list(self.net_config.nodes.keys())[0], op=op, attr=attr)
                    inp_dict[var.name] = np.reshape(val, (sim_steps,) + tuple(var.shape))

                elif any(['_all' in key_tmp for key_tmp in self.net_config.nodes.keys()]):

                    # node-vectorized case
                    if node == 'all':

                        # go through all nodes, extract the variable and add it to input dict
                        i = 0
                        for node_tmp in self.net_config.nodes:
                            var = self._get_node_attr(node=node_tmp, op=op, attr=attr)
                            i_new = var.shape[0] if len(var.shape) > 0 else 1
                            inp_dict[var.name] = np.reshape(val[:, i:i_new], (sim_steps,) + tuple(var.shape))
                            i += i_new

                    elif node in self.net_config.nodes.keys():

                        # add placeholder variable of node(s) to input dictionary
                        var = self._get_node_attr(node=node, op=op, attr=attr)
                        inp_dict[var.name] = np.reshape(val, (sim_steps,) + tuple(var.shape))

                    elif any([node in key_tmp for key_tmp in self.net_config.nodes.keys()]) or \
                            any([node.split('.')[0] in key_tmp for key_tmp in self.net_config.nodes.keys()]):

                        node_tmp = node.split('.')[0] if '.' in node else node

                        # add vectorized placeholder variable of specified node type to input dictionary
                        for node_tmp2 in list(self.net_config.nodes.keys()):
                            if node_tmp in node_tmp2:
                                break
                        var = self._get_node_attr(node=node_tmp2, op=op, attr=attr)
                        inp_dict[var.name] = np.reshape(val, (sim_steps,) + tuple(var.shape))

                else:

                    # non-vectorized case
                    if node == 'all':

                        # go through all nodes, extract the variable and add it to input dict
                        for i, node_tmp in enumerate(self.net_config.nodes.keys()):
                            var = self._get_node_attr(node=node_tmp, op=op, attr=attr)
                            inp_dict[var.name] = np.reshape(val[:, i], (sim_steps,) + tuple(var.shape))

                    elif any([node in key_tmp for key_tmp in self.net_config.nodes.keys()]):

                        # extract variables from nodes of specified type
                        i = 0
                        for node_tmp in self.net_config.nodes.keys():
                            if node in node_tmp:
                                var = self._get_node_attr(node=node_tmp, op=op, attr=attr)
                                inp_dict[var.name] = np.reshape(val[:, i], (sim_steps,) + tuple(var.shape))
                                i += 1

            self.backend.add_input_layer(inputs=inp_dict)

        # run simulation
        ################

        if verbose:
            print("Running the simulation...")

        if profile is None:
            output_col = self.backend.run(steps=sim_steps, outputs=output_col, dts=sampling_steps,
                                          out_dir=out_dir, profile=profile, **kwargs)
        else:
            output_col, time, memory = self.backend.run(steps=sim_steps, outputs=output_col, out_dir=out_dir,
                                                        profile=profile, dts=sampling_steps, **kwargs)

        if verbose and profile:
            if simulation_time:
                print(f"{simulation_time}s of backend behavior were simulated in {time} s given a "
                      f"simulation resolution of {self.dt} s.")
            else:
                print(f"ComputeGraph computations finished after {time} seconds.")
        elif verbose:
            print('finished!')

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
                    node_name = "/".join(key_split[:-1])
                    out_var_names.append((var_name, f'{node_name}_{i}'))
            else:
                if len(var.shape) > 1:
                    var = np.squeeze(var)
                out_var_vals.append(var)
                key_split = key.split('/')
                var_name = key_split[-1]
                var_name = var_name[:var_name.find('_col')]
                node_name = "/".join(key_split[:-1])
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
                var_col[f'{node}/{op}/{var_name}'] = self._get_node_attr(node=node, op=op, attr=var)

        elif node in self.net_config.nodes.keys() or node in self._net_config_map.keys():

            # get output variable of specific backend node
            var_col[f'{node}/{op}/{var_name}'] = self._get_node_attr(node=node, op=op, attr=var, **kwargs)

        elif any([node in key for key in self.net_config.nodes.keys()]):

            # get output variable from backend nodes of a certain type
            for node_tmp in self.net_config.nodes.keys():
                if node in node_tmp:
                    var_col[f'{node}/{op}/{var_name}'] = self._get_node_attr(node=node_tmp, op=op, attr=var, **kwargs)
        else:

            # get output variable of specific, vectorized backend node
            i = 0
            for node_tmp in self._net_config_map.keys():
                if node in node_tmp:
                    var_col[f'{node}/{op}/{var_name}_{i}'] = self._get_node_attr(node=node_tmp, op=op, attr=var,
                                                                                 **kwargs)
                    i += 1

        return var_col

    def clear(self):
        """Clears the backend graph from all operations and variables.
        """
        self.backend.clear()

    def _net_config_consistency_check(self, circuit: CircuitIR) -> CircuitIR:
        """Checks whether the passed network configuration follows the expected intermediate representation structure.

        Parameters
        ----------
        circuit
            Intermediate representation of the network configuration that should be translated into the backend.

        Returns
        -------
        CircuitIR
            The checked (and maybe slightly altered) network configuration.

        """

        # check node attributes
        #######################

        # go through each node in the network  config
        for node_name, data in circuit.nodes(data=True):
            node = data["node"]
            # check whether an operation graph exists
            try:
                op_graph = node.op_graph
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
            for op_name, data in op_graph.nodes(data=True):
                op = data["operator"]
                # check whether the variable field exists on the operator
                try:
                    variables = op.variables
                except KeyError:
                    raise KeyError(f'Key `variables` not found in operator {op_name} of node {node_name}. Every '
                                   f'operator on a node needs a field `variables` under which all necessary '
                                   'variables are defined via key-value pairs.')

                # check whether the operator field exists on the operator
                # try:
                #     op_info = op['operator']
                # except KeyError:
                #     raise KeyError(f'Key `operator` not found in operator {op_name} of node {node_name}. Every '
                #                    f'operator on a node needs a field `operator` under which all the equations, '
                #                    'inputs and output are defined.')

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
                        vtype = var.vtype
                    except KeyError:
                        raise KeyError(f'Field `vtype` not found in variable definition of variable {var_name} of '
                                       f'operator {op_name} on node {node_name}. Each variable needs a field `vtype`'
                                       f'that indicates whether the variable is a `state_var`, `constant` or '
                                       f'`placeholder`.')
                    var_defs = var_def[vtype]

                    # check the value of the variable
                    # try:
                    #     _ = node.values[op_name][var_name]
                    # except KeyError:
                    #     if var_defs['value'] is KeyError:
                    #         raise KeyError(f'Field `value` not found in variable definition of variable {var_name} of '
                    #                        f'operator {op_name} on node {node_name}, but it is needed for variables of '
                    #                        f'type {vtype}.')
                    #     elif var_defs['value'] is not None:
                    #         var['value'] = var_defs['value']

                    # check the shape of the variable
                    try:
                        shape = var.shape
                        if len(shape) == 0:
                            var['shape'] = (1,)
                    except KeyError:
                        var['shape'] = var_defs['shape']

                    # check the data type of the variable
                    try:
                        _ = var.dtype
                    except KeyError:
                        var['dtype'] = var_defs['dtype']

                    val = node.values[op_name][var_name]
                    if not hasattr(val, "shape"):
                        node.values[op_name][var_name] = np.zeros(var.shape, dtype=var.dtype) + val

                # check whether the equations, inputs and output fields exist on the operator field
                op_fields = ['equations', 'inputs', 'output']
                for field in op_fields:
                    try:
                        _ = getattr(op, field)
                    except KeyError:
                        if field == 'equations':
                            raise KeyError(f'Field {field} not found in `operators` field of operator {op_name} on '
                                           f'node {node_name}. Each operator should follow a list of equations that '
                                           f'needs to be provided at this position.')
                        elif field == 'inputs':
                            op['inputs'] = {}
                        else:
                            raise KeyError(f'Field {field} not found in `operators` field of operator {op_name} on '
                                           f'node {node_name}. Each operator should have an output, the name of which '
                                           f'needs to be provided at this position.')

        # check edge attributes
        #######################

        # go through edges
        for source, target, idx in circuit.edges:

            edge = circuit.edges[source, target, idx]

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

        return circuit

    def _collect_ops(self, ops: List[str], node_name: str) -> tuple:
        """Adds a number of operations to the backend graph.

        Parameters
        ----------
        ops
            Names of the operators that should be parsed into the graph.
        node_name
            Name of the node that the operators belong to.

        Returns
        -------
        tuple
            Collected and updated operator equations and variables

        """

        # set up update operation collector variable
        equations = []
        variables = {}

        # add operations of same hierarchical lvl to compute graph
        ############################################################

        for op_name in ops:

            # retrieve operator and operator args
            op_args = self._get_op_attr(node_name, op_name, 'variables')
            op_args['inputs'] = {}
            op_info = self._get_op_attr(node_name, op_name, 'operator')

            if getattr(op_info, 'collected', False):
                break

            # handle operator inputs
            in_ops = {}
            for var_name, inp in op_info['inputs'].items():

                # go through inputs to variable
                if inp['sources']:

                    in_ops_col = {}
                    reduce_inputs = inp['reduce_dim'] if type(inp['reduce_dim']) is bool else False
                    in_node = inp['node'] if 'node' in inp else node_name

                    for i, in_op in enumerate(inp['sources']):

                        # collect single input to op
                        in_var = self._get_op_attr(in_node, in_op, 'output', retrieve=False)
                        try:
                            in_val = self._get_op_attr(in_node, in_op, 'output', retrieve=True)
                        except KeyError:
                            in_val = None
                        in_ops_col[f"{in_node}/{in_op}/{in_var}"] = in_val

                    if len(in_ops_col) > 1:
                        in_ops[var_name] = self._map_multiple_inputs(in_ops_col, reduce_inputs)
                    else:
                        key, _ = in_ops_col.popitem()
                        in_node, in_op, in_var = key.split("/")
                        in_ops[var_name] = (in_var, {in_var: key})

            # replace input variables with input in operator equations
            for var, inp in in_ops.items():
                for i, eq in enumerate(op_info['equations']):
                    op_info['equations'][i] = replace(eq, var, inp[0], rhs_only=True)
                op_args['inputs'].update(inp[1])

            # collect operator variables and equations
            scope = f"{node_name}/{op_name}"
            variables[f"{scope}/inputs"] = {}
            equations += [(eq, scope) for eq in op_info['equations']]
            for key, var in op_args.items():
                full_key = f"{scope}/{key}"
                if key == 'inputs':
                    variables[f"{scope}/inputs"].update(var)
                elif full_key not in variables:
                    variables[full_key] = var
            try:
                setattr(op_info, 'collected', True)
            except AttributeError:
                op_info['collected'] = True

        return equations, variables

    # def _get_node_attr(self, node: str, attr: str, op: Optional[str] = None, **kwargs) -> Any:
    #     """Extract attribute from node of the network.
    #
    #     Parameters
    #     ----------
    #     node
    #         Name of the node.
    #     attr
    #         Name of the attribute on the node.
    #     op
    #         Name of the operator. Only needs to be provided for operator variables.
    #
    #     Returns
    #     -------
    #     Any
    #         Node attribute.
    #
    #     """
    #
    #     if op:
    #         return self._get_op_attr(node, op, attr, **kwargs)
    #     try:
    #         return self.net_config[node][attr]
    #     except KeyError:
    #         vals = []
    #         for op in self.net_config[node]:
    #             vals.append(self._get_op_attr(node, op, attr, **kwargs))
    #         return vals

    # def _set_node_attr(self, node: str, attr: str, val: Any, op: Optional[str] = None) -> Any:
    #     """Sets attribute on a node.
    #
    #     Parameters
    #     ----------
    #     node
    #         Name of the node.
    #     attr
    #         Name of the attribute.
    #     op
    #         Name of the operator, the attribute belongs to. Does not need to be provided for attributes that do not
    #         belong to an operator.
    #
    #     Returns
    #     -------
    #     Any
    #         New node attribute value.
    #
    #     """
    #
    #     if op:
    #         return self._set_op_attr(node, op, attr, val)
    #
    #     try:
    #         self.net_config.nodes[node]['node'][attr] = val
    #     except KeyError:
    #         vals_updated = []
    #         for op in self.net_config.nodes[node]['node']['op_graph'].nodes:
    #             vals_updated.append(self._set_op_attr(node, op, attr, val))
    #         return vals_updated

    # def _get_edge_attr(self, source: str, target: str, edge: Union[str, int], attr: str,
    #                    retrieve_from_node: bool = True) -> Any:
    #     """Extracts attribute from edge.
    #
    #     Parameters
    #     ----------
    #     source
    #         Name of the source node.
    #     target
    #         Name of the target node.
    #     edge
    #         Index or name of the edge.
    #     attr
    #         Name of the attribute
    #     retrieve_from_node
    #         If true, the attribute will be retrieved from the source or target node. Only relevant if the attribute is
    #         'source_var' or 'target_var'. Else, the value of the attribute on the edge will be returned.
    #
    #     Returns
    #     -------
    #     Any
    #         Edge attribute.
    #
    #     """
    #
    #     try:
    #         attr_val = self.net_config.edges[source, target, edge][attr]
    #         if 'var' in attr and type(attr_val) is str and retrieve_from_node:
    #             op, var = attr_val.split('/')
    #             if 'source' in attr:
    #                 attr_val = self._get_op_attr(source, op, var)
    #             else:
    #                 attr_val = self._get_op_attr(target, op, var)
    #             return op, var, attr_val
    #     except KeyError:
    #         attr_val = None
    #
    #     return attr_val

    # def _set_edge_attr(self, source: str, target: str, edge: Union[int, str], attr: str, val: Any) -> Any:
    #     """Sets value of an edge attribute.
    #
    #     Parameters
    #     ----------
    #     source
    #         Name of the source node.
    #     target
    #         Name of the target node.
    #     edge
    #         Name or index of the edge.
    #     attr
    #         Name of the attribute.
    #     val
    #         New value of the attribute.
    #
    #     Returns
    #     -------
    #     Any
    #         Value of the updated edge attribute.
    #
    #     """
    #
    #     self.net_config.edges[source, target, edge][attr] = val
    #     return self.net_config.edges[source, target, edge][attr]

    # def _get_op_attr(self, node: str, op: str, attr: str, retrieve: bool = True, **kwargs) -> Any:
    #     """Extracts attribute of an operator.
    #
    #     Parameters
    #     ----------
    #     node
    #         Name of the node.
    #     op
    #         Name of the operator on the node.
    #     attr
    #         Name of the attribute of the operator on the node.
    #     retrieve
    #         If attribute is output, this can be set to True to receive the handle to the output variable, or to false
    #         to receive the name of the output variable.
    #
    #     Returns
    #     -------
    #     Any
    #         Operator attribute.
    #
    #     """
    #
    #     if node in self._net_config_map and op in self._net_config_map[node] and attr in self._net_config_map[node][op]:
    #         node, op, attr, attr_idx = self._net_config_map[node][op][attr]
    #         idx = f"{list(attr_idx)}" if type(attr_idx) is tuple else attr_idx
    #         return self.backend.apply_idx(self._get_op_attr(node, op, attr), idx) if retrieve else attr_idx
    #     elif node in self.net_config:
    #         op = self.net_config[node]['op_graph'].nodes[op]
    #     else:
    #         raise ValueError(f'Node with name {node} is not part of this network.')
    #
    #     if attr == 'output' and retrieve:
    #         attr = op['operator']['output']
    #
    #     if attr in op['variables'].keys():
    #         attr_val = op['variables'][attr]
    #     elif hasattr(op['operator'], 'keys') and attr in op['operator'].keys():
    #         attr_val = op['operator'][attr]
    #     elif hasattr(op['operator'], attr):
    #         attr_val = getattr(op['operator'], attr)
    #     else:
    #         try:
    #             attr_val = op[attr]
    #         except KeyError as e:
    #             try:
    #                 attr_val = getattr(op, attr)
    #             except AttributeError:
    #                 raise e
    #
    #     return attr_val

    # def _set_op_attr(self, node: str, op: str, attr: str, val: Any) -> Any:
    #     """Sets value of an operator attribute.
    #
    #     Parameters
    #     ----------
    #     node
    #         Name of the node.
    #     op
    #         Name of the operator on the node.
    #     attr
    #         Name of the operator attribute on the node.
    #     val
    #         New value of the operator attribute.
    #
    #     Returns
    #     -------
    #     Any
    #         Updated attribute value.
    #
    #     """
    #
    #     op = self.net_config[node]['op_graph'].nodes[op]
    #     if attr in op['variables'].keys():
    #         op['variables'][attr] = val
    #         return op['variables'][attr]
    #     elif attr == 'output':
    #         op['variables'][op[attr]] = val
    #         return op['variables'][op[attr]]
    #     elif hasattr(op['operator'], attr):
    #         setattr(op['operator'], attr, val)
    #         return getattr(op['operator'], attr)
    #     else:
    #         try:
    #             op[attr] = val
    #             return op[attr]
    #         except KeyError:
    #             return None

    def _collect_op_layers(self, circuit, layers: list, exclude: Optional[bool] = False, op_identifier: Optional[str] = None
                           ) -> tuple:
        """

        Parameters
        ----------
        layers
        exclude
        op_identifier

        Returns
        -------

        """

        equations = []
        variables = {}

        for node_name, data in circuit.nodes(data=True):

            node = data["node"]
            op_graph = node.op_graph
            graph = op_graph.copy()  # type: DiGraph

            # go through all operators on node and pre-process + extract equations and variables
            i = 0
            while graph.nodes:

                # get all operators that have no dependencies on other operators
                # noinspection PyTypeChecker
                ops = [op for op, in_degree in graph.in_degree if in_degree == 0]

                if (i in layers and not exclude) or (i not in layers and exclude):

                    if op_identifier:
                        ops_tmp = [op for op in ops if op_identifier not in op] if exclude else \
                            [op for op in ops if op_identifier in op]
                    else:
                        ops_tmp = ops
                    op_eqs, op_vars = self._collect_ops(ops_tmp, node_name=node_name)

                    # collect primary operator equations and variables
                    if i == len(equations):
                        equations.append(op_eqs)
                    else:
                        equations[i] += op_eqs
                    for key, var in op_vars.items():
                        if key not in variables:
                            variables[key] = var

                # remove parsed operators from graph
                graph.remove_nodes_from(ops)
                i += 1

        return equations, variables

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

    @staticmethod
    def _sort_edges(circuit, edges: List[tuple], attr: str) -> dict:
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
            value = circuit.edges[source, target, edge][attr]
            if value not in edges_new.keys():
                edges_new[value] = [(source, target, edge)]
            else:
                edges_new[value].append((source, target, edge))

        return edges_new

    def _add_edge_buffer(self, circuit, node, op: str, var: str, idx: int, buffer_length: int, edge: tuple) -> None:
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

        var_path = f"{op}/{var}"
        target_shape = node[var_path].shape
        op_graph = node.op_graph

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
        inputs = op_graph.nodes[op]["inputs"]
        if var in inputs.keys():
            inputs[var]['sources'].add(f'{op}_{var}_buffer_read_{idx}')
        else:
            inputs[var] = {'sources': {f'{op}_{var}_buffer_read_{idx}'},
                           'reduce_dim': True}

        # update edge information
        edge_dict = circuit.edges[edge]
        edge_dict['target_var'] = f'{op}_{var}_buffer_rotate_{idx}/{var}_buffer_{idx}'
        edge_dict['add_project'] = True

    def _add_edge_input_collector(self, circuit, node, op: str, var: str, idx: int, edge: tuple) -> None:
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

        var_path = f"{op}/{var}"
        target_shape = node[var_path].shape
        op_graph = node.op_graph

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
        op_inputs = op_graph.nodes[op]["inputs"]
        if var in op_inputs.keys():
            op_inputs[var]['sources'].add(f'{op}_{var}_col_{idx}')
        else:
            op_inputs[var] = {'sources': {f'{op}_{var}_col_{idx}'},
                              'reduce_dim': True}

        # update edge target information
        edge_dict = circuit.edges[edge]
        edge_dict["target_var"] = f'{op}_{var}_col_{idx}/{var}_col_{idx}'

    def go_through_nodes_and_create_mapping_for_their_inputs(self, circuit):

        # (Previously was) Third Stage: Finalize edge vectorization
        ##########################################

        # go through nodes and create mapping for their inputs
        for node_name, data in circuit.nodes(data=True):
            node = data["node"]

            incoming_edges = circuit.graph.in_edges(node_name)
            incoming_edges = self._sort_edges(circuit, incoming_edges, 'target_var')

            # loop over input variables of node
            for in_var, edges in incoming_edges.items():

                # extract info for input variable connections
                n_inputs = len(edges)
                op_name, var_name = in_var.split('/')
                delays = []
                for s, t, e in edges:
                    d = circuit.edges[(s, t, e)]['delay']
                    if d is not None:
                        delays.append(d)

                if delays:
                    max_delay = np.max(delays)
                else:
                    max_delay = None

                # loop over different input sources
                for j in range(n_inputs):

                    if max_delay is not None:

                        # add synaptic buffer to the input variable
                        self._add_edge_buffer(circuit, node, op_name, var_name, idx=j, buffer_length=max_delay,
                                              edge=edges[j])

                    elif n_inputs > 1:

                        # add synaptic input collector to the input variable
                        self._add_edge_input_collector(circuit, node, op_name, var_name, idx=j, edge=edges[j])

    @staticmethod
    def _map_multiple_inputs(inputs: dict, reduce_dim: bool) -> tuple:
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
        tuple
            Summed up or concatenated input variables and the mapping to the respective input variables

        """

        inputs_unique = []
        input_mapping = {}
        for key, var in inputs.items():
            node, in_op, in_var = key.split('/')
            i = 0
            inp = in_var
            while inp in inputs_unique:
                i += 1
                if inp[-2:] == f"_{i - 1}":
                    inp[:-2] = f"_{i}"
                else:
                    inp = f"{inp}_{i}"
            inputs_unique.append(inp)
            input_mapping[inp] = key

        if reduce_dim:
            inputs_unique = f"sum(({','.join(inputs_unique)}), 0)"
        else:
            idx = 0
            var = inputs[input_mapping[inputs_unique[idx]]]
            while not hasattr(var, 'shape'):
                idx += 1
                var = inputs[input_mapping[inputs_unique[idx]]]
            shape = var['shape']
            if len(shape) > 0:
                inputs_unique = f"reshape(({','.join(inputs_unique)}), ({len(inputs_unique) * shape[0],}))"
            else:
                inputs_unique = f"stack({','.join(inputs_unique)})"
        return inputs_unique, input_mapping


def sort_equations(edge_eqs: list, node_eqs: list) -> list:
    """

    Parameters
    ----------
    edge_eqs
    node_eqs

    Returns
    -------

    """

    from pyrates.backend.parser import is_diff_eq

    # clean up equations
    for i, layer in enumerate(edge_eqs.copy()):
        if not layer:
            edge_eqs.pop(i)
    for i, layer in enumerate(node_eqs.copy()):
        if not layer:
            node_eqs.pop(i)

    # re-order node equations
    eqs_new = []
    for node_layer in node_eqs.copy():
        if not any([is_diff_eq(eq) for eq, _ in node_layer]):
            eqs_new.append(node_layer)
            node_eqs.pop(node_eqs.index(node_layer))

    eqs_new += edge_eqs
    eqs_new += node_eqs

    return eqs_new
