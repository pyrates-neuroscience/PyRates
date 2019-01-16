
# -*- coding: utf-8 -*-
#
#
# PyRates software framework for flexible implementation of neural 
# network models and simulations. See also: 
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
"""Functions for performing parameter grid simulations with pyrates models.
"""

# external imports
import pandas as pd
import numpy as np
import paramiko

# system imports
import os
import json
import getpass
from datetime import date
from threading import Thread

# pyrates internal imports
from pyrates.backend import ComputeGraph
from pyrates.frontend import CircuitTemplate
from pyrates.ir.circuit import CircuitIR

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


def cluster_grid_search(circuit_template, param_grid, param_map, dt, simulation_time, inputs, outputs, hostnames,
                        sampling_step_size=None, permute_grid=False, load_config_file=None,
                        save_config_file=f'{os.getcwd()}/CGS_config_{str(date.today())}_default.json',
                        **kwargs):
    """

    Parameters
    ----------
    hostnames
    circuit_template
    param_grid
    param_map
    dt
    simulation_time
    inputs
    outputs
    sampling_step_size
    permute_grid
    load_config_file
    save_config_file
    kwargs

    Returns
    -------

    """
    # TODO: Implement proper check weather save_config_file or load_config_file are specified
    print(save_config_file)
    if load_config_file is not None:
        print(f'Loading config file: {load_config_file}')
        config_fp = load_config_file
    else:
        print(f'No load_config_file found')
        print(f'Creating default config_file: {save_config_file}')

        # linearize parameter grid if necessary
        if type(param_grid) is dict:
            # convert linear_grid from dict to pandas.DataFrame. Add status_flag later
            param_grid = linearize_grid(param_grid, permute_grid, add_status_flag=False)

        create_config_file(save_config_file, circuit_template, param_grid, param_map, dt, simulation_time, inputs,
                           outputs, hostnames, sampling_step_size, permute_grid, **kwargs)

    # TODO: Load param_grid from config_file and start thread/scheduler

    # TODO: Long-term: Implement asynchronous computation instead of multiple threads
    # for host in hostnames:
    #     spawn_thread(host, circuit_template, param_grid, param_map, dt, simulation_time, inputs, outputs, results,
    #             sampling_step_size=None, **kwargs)
    #
    # return results


def spawn_thread(host, config_file):
    t = Thread(
        target=thread_master,
        args=(host, config_file)
    )
    t.start()
    t.join()


def thread_master(host, config_file):

    env = '/data/u_salomon_software/anaconda3/envs/PyRates/bin/python'
    workerfile = '/data/hu_salomon/PycharmProjects/PyRates/pyrates/utility/cluster_worker.py'
    command = env + ' ' + workerfile

    # create SSH Client/Channel
    # TODO: Implement connection with key-files and no password

    client = create_ssh_connection(host,
                                   username=getpass.getuser(),
                                   password=getpass.getpass(
                                       prompt='Enter password:', stream=None)
                                   )

    # Check if create_ssh_connection() didn't return 0
    # if client:

    # If needed, insert a function to copy all necessary files (environments, worker files, log files) here
    # -> Change paths of env and workerfile respectively

    # Check if 'status'-key is present in param_grid
    if not fetch_param_idx(param_grid, set_status=False).isnull():
        # TODO: Call exec_command only with the config_file as command line argument
        # TODO: Call exec_command only once and communicate with it via stdin inside the while loop
        print(f'\'{host}\': Starting computation')

        # Check for available parameters to fetch
        while not fetch_param_idx(param_grid, set_status=False).empty:

            param_idx = fetch_param_idx(param_grid, num_params=4)
            param_grid = param_grid.iloc[param_idx]

            # - All input to the remote script needs to be sent as command line arguments
            # - Dictionaries have to be parsed as string using "" -> f' "{dict}"'
            # - To parse a DataFrame convert it do a dict first using DataFrame.to_dict()
            # - Beware not to use JSON-like strings or dicts, since JSON is based on double quotas, which are
            #   eliminated by the shell during the parcing process
            # stdin, stdout, stderr = client.exec_command(command +
            #                                             f' --circuit_template="{circuit_template}"'
            #                                             f' --param_grid="{param_grid.to_dict()}"'
            #                                             f' --param_map="{param_map}"'
            #                                             f' --inputs="{inputs}"'
            #                                             f' --outputs="{outputs}"'
            #                                             f' --sampling_step_size={sampling_step_size}'
            #                                             f' --dt={dt}'
            #                                             f' --simulation_time={simulation_time}',
            #                                             get_pty=True)
            #
            # exit_status = stdout.channel.recv_exit_status()
            #
            # for line in iter(stdout.readline, ""):
            #     print(line, end="")

            # TODO: Create result file and concatenate the intermediate results directly to this file
            #
            # result = pd.read_csv(stdout)


    else:
        print("No key named 'status' in param_grid")

    client.close()
    # return result


def create_config_file(config_fp, circuit_template, param_grid, param_map, dt, simulation_time, inputs,
                       outputs, hostnames, sampling_step_size, permute_grid, **kwargs):
    config_dict = {
        "circuit_template": circuit_template,
        "param_grid": param_grid.to_dict(),
        "param_map": param_map,
        "dt": dt,
        "simulation_time": simulation_time,
        "inputs": {str(*inputs.keys()): list(*inputs.values())},
        "outputs": outputs,
        "hostnames": hostnames,
        "sampling_step_size": sampling_step_size,
        "permute_grid": permute_grid,
        "kwargs": kwargs
    }

    with open(config_fp, "w") as f:
        json.dump(config_dict, f, indent=2)


def create_ssh_connection(host, username, password):
    """Connect to a host via SSH

    Parameters
    ----------
    host
        Name or IP-address of the host to connect to
    username
    password

    Returns
    -------
    paramiko.SSHClient()
        Throws exception and returns 0 if connection fails. See Paramiko documentation


    """
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f'Attempting to connect to host \'{host}\'...')
    try:
        client.connect(host, username=username, password=password)
        print(f'\'{host}\': Connection established')
        return client
        # return client.invoke_shell()
    except paramiko.AuthenticationException:
        print(f'\'{host}\': Couldn\'t establish connection to host \'{host}\'. Authentication failed')
        return 0
    except IOError:
        print(f'\'{host}\': Couldn\'t establish connection to host \'{host}\'. No such host available')
        return 0


def fetch_param_idx(param_grid, num_params=1, set_status=True):
    """Fetch a pandas.Index([index_list]) with the indices of the first num_params rows of param_grid who's
    'status'-key equals 'unsolved'

    Parameters
    ----------
    param_grid
        Linearized parameter grid of type pandas.DataFrame.
    num_params
        Number of indices to fetch from param_grid. Is 1 by default.
    set_status
        If True, sets 'status' key of the fetched rows to 'pending', to exclude them from future calls.
        Can be used to check param_grid for fetchable or existend keys without changing their 'status' key.
        Is True by default.

    Returns
    -------
    pandas.Index([index_list])
        Is empty if there are no row indices to be fetched.
        Is np.nan if param_grid has no key named 'status'.
        Contains all remaining indices if num_params is higher than fetchable row indices.


    """
    try:
        # Get the first num_params row indices of lin_grid who's 'status' keys equal 'unsolved'
        param_idx = param_grid.loc[param_grid['status'] == 'unsolved'].index[:num_params]
    except KeyError:
        # print("DataFrame doesn't contain a key named 'status'")
        return pd.Index([np.nan])

    if set_status:
        param_grid.at[param_idx, 'status'] = 'pending'

    return param_idx
    # To access the selected data use fetched_params = lin_grid.iloc[param_idx]


def grid_search(circuit_template, param_grid, param_map, dt, simulation_time, inputs, outputs,
                sampling_step_size=None, permute_grid=False, **kwargs):
    """

    Parameters
    ----------
    circuit_template
    param_grid
    param_map
    dt
    simulation_time
    inputs
    outputs
    sampling_step_size
    permute_grid
    kwargs

    Returns
    -------

    """

    # linearize parameter grid if necessary
    if type(param_grid) is dict:
        param_grid = linearize_grid(param_grid, permute_grid)

    # assign parameter updates to each circuit and combine them to unconnected network
    circuit = CircuitIR()
    circuit_names = []
    param_info = []
    param_split = "--"
    val_split = "-"
    comb = "_"
    for n in range(param_grid.shape[0]):
        circuit_tmp = CircuitTemplate.from_yaml(circuit_template).apply()
        circuit_names.append(f'{circuit_tmp.label}_{n}')
        circuit_tmp = adapt_circuit(circuit_tmp, param_grid.iloc[n, :], param_map)
        circuit.add_circuit(circuit_names[-1], circuit_tmp)
        param_names = list(param_grid.columns.values)
        param_info_tmp = [f"{param_names[i]}{val_split}{val}" for i, val in enumerate(param_grid.iloc[n, :])]
        param_info.append(param_split.join(param_info_tmp))

    # create backend graph
    net = ComputeGraph(circuit, dt=dt, **kwargs)

    # adjust input of simulation to combined network
    for inp_key, inp in inputs.items():
        inputs[inp_key] = np.tile(inp, (1, len(circuit_names)))

    # adjust output of simulation to combined network
    nodes = list(CircuitTemplate.from_yaml(circuit_template).apply().nodes)
    out_names = list(outputs.keys())
    for out_key, out in outputs.copy().items():
        outputs.pop(out_key)
        if out[0] in nodes:
            for i, name in enumerate(param_info):
                out_tmp = list(out)
                out_tmp[0] = f'{circuit_names[i]}/{out_tmp[0]}'
                outputs[f'{name}{param_split}out_var{val_split}{out_key}'] = tuple(out_tmp)
        elif out[0] == 'all':
            out_names = []
            for node in nodes:
                for i, name in enumerate(param_info):
                    out_tmp = list(out)
                    out_tmp[0] = f'{circuit_names[i]}/{node}'
                    outputs[f'{name}{param_split}out_var{val_split}{out_key}{comb}{node}'] = tuple(out_tmp)
                    out_names.append(f'{out_key}{comb}{node}')
            out_names = list(set(out_names))
        else:
            node_found = False
            out_names = []
            for node in nodes:
                if out[0] in node:
                    node_found = True
                    for i, name in enumerate(param_info):
                        out_tmp = list(out)
                        out_tmp[0] = f'{circuit_names[i]}/{node}'
                        outputs[f'{name}{param_split}out_var{val_split}{out_key}{comb}{node}'] = tuple(out_tmp)
                        out_names.append(f'{out_key}{comb}{node}')
            out_names = list(set(out_names))
            if not node_found:
                raise ValueError(f'Invalid output identifier in output: {out_key}. '
                                 f'Node {out[0]} is not part of this network')

    # simulate the circuits behavior
    results = net.run(simulation_time=simulation_time,
                      inputs=inputs,
                      outputs=outputs,
                      sampling_step_size=sampling_step_size)

    # transform results into long-form dataframe with changed parameters as columns
    multi_idx = [param_grid[key].values for key in param_grid.keys()]
    n_iters = len(multi_idx[0])
    outs = []
    for out_name in out_names:
        outs += [out_name] * n_iters
    multi_idx = [list(idx) * len(out_names) for idx in multi_idx]
    multi_idx.append(outs)
    index = pd.MultiIndex.from_arrays(multi_idx,
                                      names=list(param_grid.keys()) + ['out_var'])
    results_final = pd.DataFrame(columns=index, data=np.zeros_like(results.values))
    for col in results.keys():
        params = col[0].split(param_split)
        indices = [None] * len(results_final.columns.names)
        for param in params:
            var, val = param.split(val_split)
            idx = list(results_final.columns.names).index(var)
            try:
                indices[idx] = float(val)
            except ValueError:
                indices[idx] = val
        results_final.loc[:, tuple(indices)] = results[col].values

    return results_final.dropna()


def linearize_grid(grid: dict, permute=False, add_status_flag=False):
    """

    Parameters
    ----------
    grid
    permute
    add_status

    Returns
    -------

    """

    arg_lengths = [len(arg) for arg in grid.values()]

    if len(list(set(arg_lengths))) == 1 and not permute:
        df = pd.DataFrame(grid)
        if not add_status_flag:
            return df
        else:
            # Add status key to each entry
            df['status'] = 'unsolved'
            return df
    else:
        vals, keys = [], []
        for key, val in grid.items():
            vals.append(val)
            keys.append(key)
        new_grid = np.stack(np.meshgrid(*tuple(vals)), -1).reshape(-1, len(grid))
        df = pd.DataFrame(new_grid, columns=keys)
        if not add_status_flag:
            return df
        else:
            # Add a status key to each entry
            df['status'] = 'unsolved'
            return df


def adapt_circuit(circuit, params, param_map):
    """

    Parameters
    ----------
    circuit
    params
    param_map

    Returns
    -------

    """

    for key in params.keys():
        val = params[key]
        for op, var in param_map[key]['var']:
            nodes = param_map[key]['nodes'] if 'nodes' in param_map[key] else []
            edges = param_map[key]['edges'] if 'edges' in param_map[key] else []
            for node in nodes:
                if op in circuit.nodes[node]['node'].op_graph.nodes:
                    circuit.nodes[node]['node'].op_graph.nodes[op]['variables'][var]['value'] = float(val)
            for source, target, edge in edges:
                if op in circuit.edges[source, target, edge]:
                    circuit.edges[source, target, edge][op][var] = float(val)
                elif var in circuit.edges[source, target, edge]:
                    circuit.edges[source, target, edge][var] = float(val)

    return circuit
